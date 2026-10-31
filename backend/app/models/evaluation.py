"""Schemas for the offline evaluation harness.

These describe the on-disk test set (:class:`EvaluationCase`) and the result
document a run writes to ``data/evaluations/{run_id}.json``
(:class:`EvaluationRun`).  They are deliberately separate from
``app.models.schemas``, which carries the live HTTP request/response contract —
evaluation is an offline concern and must not leak into the chat path.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    """One fixture PDF making up the evaluation corpus."""

    document_id: str = Field(
        description="Stable id used in place of a UUID so expected_sources stays valid across runs."
    )
    filename: str = Field(description="Filename of the PDF within the corpus directory.")


class EvaluationCase(BaseModel):
    """A single ground-truthed question in the test set."""

    id: str = Field(description="Stable case identifier, unique within the test set.")
    question: str = Field(description="The question put to the retrieval pipeline.")
    expected_answer: str = Field(
        description="Reference answer. Used by the judge metrics as the ground truth."
    )
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Corpus document ids that can legitimately support the answer. Empty for negatives.",
    )
    expected_refusal: bool = Field(
        default=False,
        description="True when the corpus cannot answer the question and a refusal is the correct behaviour.",
    )
    tags: list[str] = Field(default_factory=list, description="Free-form labels for slicing results.")


class TestSet(BaseModel):
    """The complete on-disk test set document."""

    version: int = Field(description="Schema version of the test set file.")
    corpus: list[CorpusDocument] = Field(description="Fixture documents the cases are written against.")
    cases: list[EvaluationCase] = Field(description="All ground-truthed cases.")


class RetrievalMetrics(BaseModel):
    """Deterministic, judge-free metrics for one case.

    These need no LLM, so they run under the keyless ``stub``/``local``
    providers and are the primary regression signal.
    """

    source_hit: bool = Field(description="True when any expected source appears among the citations.")
    first_hit_rank: int | None = Field(
        default=None,
        description="1-based rank of the first correct citation, or null when none matched.",
    )
    reciprocal_rank: float = Field(
        default=0.0, description="1 / first_hit_rank, or 0.0 when no expected source was cited."
    )
    refused: bool = Field(description="True when the pipeline returned a no-evidence response.")
    refusal_correct: bool = Field(
        description="True when refusal behaviour matched the case's expected_refusal."
    )
    grounded: bool = Field(description="The grounded flag returned by the pipeline.")
    citation_count: int = Field(description="Number of citations returned.")
    cited_sources: list[str] = Field(
        default_factory=list, description="Document ids cited, in the order returned."
    )


class JudgeMetric(BaseModel):
    """One LLM-as-judge metric outcome for a case."""

    name: str = Field(description="Metric identifier, e.g. 'faithfulness'.")
    score: float | None = Field(default=None, description="Score in [0, 1], or null when not run.")
    passed: bool | None = Field(default=None, description="Whether the score met eval_threshold.")
    status: str = Field(description="One of 'scored', 'skipped' or 'error'.")
    reason: str | None = Field(default=None, description="Judge rationale, or why the metric did not run.")


class CaseResult(BaseModel):
    """Everything recorded for one evaluated case."""

    case_id: str
    question: str
    expected_answer: str
    answer: str = Field(description="The answer the pipeline actually produced.")
    contexts: list[str] = Field(description="Excerpts cited, as passed to the judge metrics.")
    retrieval: RetrievalMetrics
    judge: list[JudgeMetric] = Field(default_factory=list)
    latency_seconds: float
    tags: list[str] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Set when the pipeline raised for this case rather than answering."
    )


class RunConfigSnapshot(BaseModel):
    """The configuration a run was executed under.

    Without this two runs are not comparable, which defeats the purpose of
    having a fixed test set at all.
    """

    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    max_answer_citations: int
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    judge_enabled: bool
    judge_model: str | None = None
    eval_threshold: float


class RunAggregates(BaseModel):
    """Run-level roll-up across all cases."""

    case_count: int
    errored_cases: int = 0
    source_hit_rate: float = Field(description="Fraction of answerable cases citing an expected source.")
    mean_reciprocal_rank: float = Field(description="MRR over answerable cases.")
    refusal_accuracy: float = Field(description="Fraction of all cases with correct refusal behaviour.")
    grounded_rate: float = Field(description="Fraction of answerable cases reported as grounded.")
    mean_latency_seconds: float
    judge_means: dict[str, float] = Field(
        default_factory=dict, description="Mean score per judge metric over the cases that scored."
    )


class EvaluationRun(BaseModel):
    """The complete result document for one evaluation run."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = Field(description="One of 'running', 'completed' or 'failed'.")
    testset_version: int
    config: RunConfigSnapshot
    aggregates: RunAggregates | None = None
    cases: list[CaseResult] = Field(default_factory=list)
    error: str | None = None
