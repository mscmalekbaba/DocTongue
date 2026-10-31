"""Tests for the offline evaluation harness.

Everything here runs against the keyless ``stub``/``local`` providers forced by
the autouse ``zero_key_providers`` fixture in ``conftest.py``, so no test makes
a live API call. Judge behaviour is exercised with stubs rather than a real
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.evaluation import CaseResult, EvaluationCase, JudgeMetric, RetrievalMetrics
from app.models.schemas import ChatResponse, Citation
from app.services.evaluation import (
    EvaluationRunner,
    aggregate,
    is_refusal,
    load_testset,
    score_retrieval,
)


CORPUS_DIR = Path(__file__).resolve().parents[1] / "app" / "evaluation" / "corpus"
TESTSET_PATH = Path(__file__).resolve().parents[1] / "app" / "evaluation" / "testset.json"


def _citation(document_id: str, excerpt: str = "excerpt") -> Citation:
    return Citation(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        excerpt=excerpt,
        page_number=1,
        score=0.9,
    )


def _case(**overrides: object) -> EvaluationCase:
    defaults: dict[str, object] = {
        "id": "case-1",
        "question": "What is the deductible on the Standard plan?",
        "expected_answer": "£750.",
        "expected_sources": ["eval-shield-standard"],
        "expected_refusal": False,
        "tags": ["confusable"],
    }
    defaults.update(overrides)
    return EvaluationCase.model_validate(defaults)


def _result(**overrides: object) -> CaseResult:
    defaults: dict[str, object] = {
        "case_id": "case-1",
        "question": "q",
        "expected_answer": "a",
        "answer": "answer",
        "contexts": ["excerpt"],
        "retrieval": RetrievalMetrics(
            source_hit=True,
            first_hit_rank=1,
            reciprocal_rank=1.0,
            refused=False,
            refusal_correct=True,
            grounded=True,
            citation_count=1,
            cited_sources=["eval-shield-standard"],
        ),
        "judge": [],
        "latency_seconds": 0.1,
        "tags": ["confusable"],
    }
    defaults.update(overrides)
    return CaseResult.model_validate(defaults)


# --- test set and corpus integrity -------------------------------------------------


def test_testset_loads_and_corpus_files_exist() -> None:
    testset = load_testset(TESTSET_PATH)

    assert testset.cases, "test set must not be empty"
    assert len({case.id for case in testset.cases}) == len(testset.cases), "case ids must be unique"
    for document in testset.corpus:
        assert (CORPUS_DIR / document.filename).exists(), f"missing corpus file {document.filename}"


def test_expected_sources_reference_known_corpus_documents() -> None:
    testset = load_testset(TESTSET_PATH)
    known = {document.document_id for document in testset.corpus}

    for case in testset.cases:
        assert set(case.expected_sources) <= known, f"{case.id} references an unknown document"
        if case.expected_refusal:
            assert not case.expected_sources, f"negative case {case.id} must expect no sources"
        else:
            assert case.expected_sources, f"answerable case {case.id} must name a source"


def test_testset_contains_negative_cases() -> None:
    testset = load_testset(TESTSET_PATH)

    negatives = [case for case in testset.cases if case.expected_refusal]
    assert len(negatives) >= 2, "test set needs negatives to catch over-eager answering"


# --- deterministic metrics ---------------------------------------------------------


def test_is_refusal_matches_pipeline_no_evidence_responses() -> None:
    assert is_refusal("I could not find an answer in the indexed documents.")
    assert is_refusal(
        "I could not find enough supporting evidence in the indexed documents to answer that."
    )
    assert not is_refusal("The Standard plan has a £750 deductible. [1]")


# Verbatim openings of real nemotron-3.5-super-vl answers to the three negative
# cases. An earlier canned-string-only matcher scored all three as failures.
REAL_MODEL_REFUSALS = [
    "I could not find information about a Platinum plan in the indexed documents. "
    "The sources only mention three ShieldCare plan tiers: Basic, Standard, and Premium.",
    "I could not find any mention of pet insurance or bundling discounts in the provided "
    "ShieldCare documents. The sources cover plan tiers, coverage features and exclusions.",
    "I could not find the name of the chief medical officer of ShieldCare in the provided "
    "document excerpts. None of the sources mention this role.",
]


@pytest.mark.parametrize("answer", REAL_MODEL_REFUSALS)
def test_is_refusal_detects_natural_language_refusals(answer: str) -> None:
    """A model refuses in its own words, not by echoing the canned strings."""
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        # A negative *fact* is an answer, not a refusal. These are the cases a
        # looser matcher would wrongly credit as refusals.
        "Routine dental treatment is not included under the Basic plan. [1]",
        "The Basic plan does not provide an annual health assessment. [1]",
        "Physiotherapy is not available on the Basic plan, but the Premium plan covers 24 "
        "sessions per year. [1][2]",
        "The nurse advice line is not available to Basic plan members. [1]",
    ],
)
def test_is_refusal_does_not_fire_on_negative_facts(answer: str) -> None:
    assert not is_refusal(answer)


GENERATION_FAILURE_ANSWER = (
    "I found relevant excerpts in the indexed documents, but I could not generate "
    "a final response from the configured LLM provider."
)


def test_generation_failure_is_not_scored_as_a_refusal() -> None:
    """A provider outage is neither an answer nor a decline.

    This exact fallback appeared when the judge-enabled run's answer call timed
    out. Reading it as a refusal penalised the pipeline for an outage.
    """
    from app.services.evaluation import is_generation_failure

    assert is_generation_failure(GENERATION_FAILURE_ANSWER)
    assert not is_refusal(GENERATION_FAILURE_ANSWER)


def test_aggregate_excludes_errored_cases_from_retrieval_rates() -> None:
    good = _result(case_id="ok")
    failed = _result(
        case_id="llm-down",
        error="LLM generation failed; pipeline returned its evidence-only fallback.",
        retrieval=RetrievalMetrics(
            source_hit=False,
            refused=False,
            refusal_correct=False,
            grounded=False,
            citation_count=2,
        ),
    )

    totals = aggregate([good, failed])

    # An outage must not look like a retrieval regression.
    assert totals.source_hit_rate == 1.0
    assert totals.mean_reciprocal_rank == 1.0
    assert totals.grounded_rate == 1.0
    assert totals.errored_cases == 1


def test_score_retrieval_ranks_first_correct_citation() -> None:
    response = ChatResponse(
        answer="The deductible is £750. [1]",
        citations=[_citation("eval-shield-basic"), _citation("eval-shield-standard")],
        grounded=True,
    )

    metrics = score_retrieval(_case(), response)

    assert metrics.source_hit is True
    assert metrics.first_hit_rank == 2
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.refused is False
    assert metrics.refusal_correct is True
    assert metrics.cited_sources == ["eval-shield-basic", "eval-shield-standard"]


def test_score_retrieval_reports_miss_when_wrong_document_cited() -> None:
    """The failure the previous per-answer judge could not see."""
    response = ChatResponse(
        answer="The deductible is £250. [1]",
        citations=[_citation("eval-shield-premium")],
        grounded=True,
    )

    metrics = score_retrieval(_case(), response)

    assert metrics.source_hit is False
    assert metrics.first_hit_rank is None
    assert metrics.reciprocal_rank == 0.0


def test_score_retrieval_credits_correct_refusal_on_negative_case() -> None:
    case = _case(id="neg", expected_sources=[], expected_refusal=True, tags=["negative"])
    response = ChatResponse(
        answer="I could not find an answer in the indexed documents.",
        citations=[],
        grounded=False,
    )

    metrics = score_retrieval(case, response)

    assert metrics.refused is True
    assert metrics.refusal_correct is True
    assert metrics.source_hit is False


def test_score_retrieval_penalises_answering_a_negative_case() -> None:
    case = _case(id="neg", expected_sources=[], expected_refusal=True, tags=["negative"])
    response = ChatResponse(
        answer="The Platinum plan costs £140 per month. [1]",
        citations=[_citation("eval-shield-premium")],
        grounded=True,
    )

    metrics = score_retrieval(case, response)

    assert metrics.refused is False
    assert metrics.refusal_correct is False


# --- aggregation -------------------------------------------------------------------


def test_aggregate_excludes_negatives_from_retrieval_rates() -> None:
    hit = _result(case_id="hit")
    negative = _result(
        case_id="neg",
        tags=["negative"],
        retrieval=RetrievalMetrics(
            source_hit=False,
            refused=True,
            refusal_correct=True,
            grounded=False,
            citation_count=0,
        ),
    )

    totals = aggregate([hit, negative])

    # A negative correctly citing nothing must not drag the hit rate down.
    assert totals.source_hit_rate == 1.0
    assert totals.mean_reciprocal_rank == 1.0
    assert totals.refusal_accuracy == 1.0
    assert totals.case_count == 2


def test_aggregate_counts_errored_cases() -> None:
    errored = _result(
        case_id="boom",
        error="provider exploded",
        retrieval=RetrievalMetrics(
            source_hit=False,
            refused=False,
            refusal_correct=False,
            grounded=False,
            citation_count=0,
        ),
    )

    totals = aggregate([_result(case_id="ok"), errored])

    assert totals.errored_cases == 1
    assert totals.case_count == 2


def test_aggregate_averages_only_scored_judge_metrics() -> None:
    first = _result(
        case_id="a",
        judge=[
            JudgeMetric(name="faithfulness", score=1.0, passed=True, status="scored"),
            JudgeMetric(name="answer_relevancy", status="skipped", reason="disabled"),
        ],
    )
    second = _result(
        case_id="b",
        judge=[JudgeMetric(name="faithfulness", score=0.5, passed=False, status="scored")],
    )

    totals = aggregate([first, second])

    assert totals.judge_means == {"faithfulness": 0.75}


# --- runner ------------------------------------------------------------------------


def test_runner_uses_the_evaluation_collection_not_the_live_one() -> None:
    runner = EvaluationRunner(Settings(eval_collection_name="documents_eval"))

    assert runner._settings.chroma_collection_name == "documents_eval"


def test_runner_marks_judge_skipped_with_a_reason_when_disabled() -> None:
    """Skipping must be explicit, not a silent fallback to a weaker score."""
    runner = EvaluationRunner(Settings(eval_judge_enabled=False))
    cases = [_result()]

    runner._apply_judge(cases)

    assert cases[0].judge, "every case should carry judge metric placeholders"
    assert {metric.status for metric in cases[0].judge} == {"skipped"}
    assert all(metric.score is None for metric in cases[0].judge)
    assert all("disabled" in (metric.reason or "") for metric in cases[0].judge)


def test_runner_marks_judge_skipped_when_judge_cannot_be_built() -> None:
    """A judge that cannot run must skip loudly, never score silently.

    The reason text is deliberately not asserted verbatim: the judge can be
    unavailable because deepeval is absent or because no key is configured, and
    either way the requirement is the same — every metric unscored, and every
    one carrying an explanation.
    """
    # Null the keys explicitly: a developer's backend/.env would otherwise make
    # this a live (billed) judge run.
    runner = EvaluationRunner(Settings(eval_judge_enabled=True, openrouter_api_key=None, llm_api_key=None))
    cases = [_result()]

    runner._apply_judge(cases)

    assert {metric.status for metric in cases[0].judge} == {"skipped"}
    assert all(metric.score is None for metric in cases[0].judge)
    assert all(metric.passed is None for metric in cases[0].judge)
    assert all("unavailable" in (metric.reason or "").lower() for metric in cases[0].judge)


def test_build_judge_refuses_without_credentials_or_deepeval() -> None:
    """``build_judge`` signals unavailability as a typed error, not a bare crash."""
    from app.services.eval_judge import JudgeUnavailableError, build_judge

    with pytest.raises(JudgeUnavailableError):
        build_judge(Settings(eval_judge_enabled=True, openrouter_api_key=None, llm_api_key=None))


def test_judge_json_loads_tolerates_markdown_fences() -> None:
    """Judges wrap JSON in code fences unbidden; the adapter must cope."""
    from app.services.eval_judge import judge_json_loads

    assert judge_json_loads('```json\n{"score": 1}\n```') == {"score": 1}
    assert judge_json_loads('Here you go: {"score": 0.5} - hope that helps') == {"score": 0.5}


def test_full_run_against_the_real_pipeline_writes_a_result(tmp_path: Path) -> None:
    """End-to-end run under the keyless providers, with no network access."""
    settings = Settings(eval_results_dir=tmp_path / "evaluations")
    run = EvaluationRunner(settings).run()

    assert run.status == "completed", run.error
    assert run.aggregates is not None
    assert run.aggregates.case_count == len(load_testset(TESTSET_PATH).cases)
    assert run.aggregates.errored_cases == 0

    # The config snapshot is what makes two runs comparable.
    assert run.config.llm_provider == "stub"
    assert run.config.embedding_provider == "local"

    written = tmp_path / "evaluations" / f"{run.run_id}.json"
    assert written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert len(payload["cases"]) == run.aggregates.case_count


def test_evaluation_run_does_not_touch_the_chat_audit_log(tmp_path: Path) -> None:
    """A test-set run must not append to the tracked chat audit log."""
    audit_path = tmp_path / "chat_audit.txt"
    settings = Settings(
        chat_audit_log_path=audit_path,
        eval_results_dir=tmp_path / "evaluations",
    )

    run = EvaluationRunner(settings).run(write_result=False)

    assert run.status == "completed", run.error
    assert not audit_path.exists(), "evaluation runs must pass audit=False"
