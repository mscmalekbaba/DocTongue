"""CLI entry point for an offline evaluation run.

Usage, from the ``backend`` directory::

    python -m app.evaluation.run                 # deterministic metrics only
    EVAL_JUDGE_ENABLED=1 python -m app.evaluation.run   # adds judge metrics

Writes the full result document to ``data/evaluations/{run_id}.json`` and
prints a summary table.  Judge metrics need an OpenRouter key and cost roughly
eight API calls per case; without one they are reported as skipped rather than
quietly replaced with a weaker score.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.core.config import get_settings
from app.models.evaluation import EvaluationRun
from app.services.evaluation import EvaluationRunner


def _format_summary(run: EvaluationRun) -> str:
    """Render *run* as a human-readable summary block."""
    lines: list[str] = []
    config = run.config
    lines.append(f"run {run.run_id} - {run.status}")
    lines.append(
        f"  config: llm={config.llm_provider}:{config.llm_model} "
        f"embeddings={config.embedding_provider}:{config.embedding_model} "
        f"k={config.retrieval_k} chunk={config.chunk_size}/{config.chunk_overlap}"
    )
    lines.append(
        f"  judge: {'enabled ' + (config.judge_model or '') if config.judge_enabled else 'disabled'}"
    )

    if run.error:
        lines.append(f"  error: {run.error}")

    totals = run.aggregates
    if totals is not None:
        lines.append("")
        lines.append(f"  cases                {totals.case_count} ({totals.errored_cases} errored)")
        lines.append(f"  source hit rate      {totals.source_hit_rate:.2%}")
        lines.append(f"  mean reciprocal rank {totals.mean_reciprocal_rank:.3f}")
        lines.append(f"  refusal accuracy     {totals.refusal_accuracy:.2%}")
        lines.append(f"  grounded rate        {totals.grounded_rate:.2%}")
        lines.append(f"  mean latency         {totals.mean_latency_seconds:.3f}s")
        for name, value in sorted(totals.judge_means.items()):
            lines.append(f"  judge {name:<14} {value:.3f}")

    lines.append("")
    lines.append(f"  {'case':<34} {'hit':<5} {'rank':<5} {'refuse':<7} cites")
    for case in run.cases:
        retrieval = case.retrieval
        marker = "err" if case.error else ("yes" if retrieval.source_hit else "no")
        rank = "-" if retrieval.first_hit_rank is None else str(retrieval.first_hit_rank)
        refusal = "ok" if retrieval.refusal_correct else "WRONG"
        lines.append(
            f"  {case.case_id:<34} {marker:<5} {rank:<5} {refusal:<7} {retrieval.citation_count}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the test set and print a summary. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run the DocTongue evaluation test set.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not persist the result document; print the summary only.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log ingestion and per-case progress."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    run = EvaluationRunner(settings).run(write_result=not args.no_write)
    print(_format_summary(run))

    if run.status != "completed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
