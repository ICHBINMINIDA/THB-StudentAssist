"""
Answer scoring and aggregation for the model benchmark.

Pure functions only (no models, no network), so they are unit-tested in CI.

Quality is measured with checks that do not need a reference answer:

* answerable questions:   the answer cites at least one source, cites no
                          source number that does not exist, does not fall
                          back to "I don't know", and contains the expected
                          keywords (only if the question lists any)
* unanswerable questions: the answer is the fixed fallback sentence and
                          cites nothing (the model must not invent facts)
* every question:         the answer is in the language of the question
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable

CITATION = re.compile(r"\[(\d+)\]")


def citations(answer: str) -> set[int]:
    return {int(n) for n in CITATION.findall(answer)}


def score_answer(answer: str, *, expect: str, n_sources: int, fallback_text: str,
                 keywords: Iterable[str] = (), language_ok: bool | None = None) -> dict:
    """Score one answer. ``expect`` is "answer" or "fallback"."""
    cited = citations(answer)
    is_fallback = fallback_text[:40].lower() in answer.lower()
    checks: dict[str, bool] = {}

    if expect == "fallback":
        checks["fallback_correct"] = is_fallback and not cited
    elif expect == "answer":
        checks["not_fallback"] = not is_fallback
        checks["cites_source"] = bool(cited)
        checks["citations_valid"] = all(1 <= n <= n_sources for n in cited)
        keywords = [k for k in keywords if k.strip()]
        if keywords:
            hits = sum(k.lower() in answer.lower() for k in keywords)
            checks["keywords"] = hits == len(keywords)
    else:
        raise ValueError(f"expect must be 'answer' or 'fallback', got {expect!r}")

    if language_ok is not None:
        checks["language"] = language_ok

    return {"checks": checks, "quality": sum(checks.values()) / len(checks)}


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate benchmark rows (one per model x question x repeat) per model."""
    models: dict[str, list[dict]] = {}
    for row in rows:
        models.setdefault(row["model"], []).append(row)

    summary = []
    for model, items in models.items():
        ok = [r for r in items if not r.get("error")]

        def rate(check: str) -> float | None:
            values = [r["checks"][check] for r in ok if check in r["checks"]]
            return round(sum(values) / len(values), 3) if values else None

        summary.append({
            "model": model,
            "runs": len(items),
            "errors": len(items) - len(ok),
            "quality": round(statistics.mean(r["quality"] for r in ok), 3) if ok else None,
            "fallback_accuracy": rate("fallback_correct"),
            "citation_rate": rate("cites_source"),
            "language_match": rate("language"),
            "ttft_median_s": _median([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
            "latency_median_s": _median([r["latency_s"] for r in ok]),
            "tokens_per_s_median": _median([r["tokens_per_s"] for r in ok
                                            if r.get("tokens_per_s") is not None]),
            "memory_gb": next((r["memory_gb"] for r in reversed(items) if r.get("memory_gb")), None),
        })
    return sorted(summary, key=lambda s: (s["quality"] is None, -(s["quality"] or 0)))


COLUMNS = [
    ("model", "Model"), ("quality", "Quality"), ("fallback_accuracy", "Says \"don't know\""),
    ("citation_rate", "Cites sources"), ("language_match", "Right language"),
    ("ttft_median_s", "TTFT (s)"), ("latency_median_s", "Latency (s)"),
    ("tokens_per_s_median", "Tokens/s"), ("memory_gb", "Memory (GB)"), ("errors", "Errors"),
]


def to_markdown(summary: list[dict]) -> str:
    def cell(value) -> str:
        if value is None:
            return "–"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = ["| " + " | ".join(title for _, title in COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for row in summary:
        lines.append("| " + " | ".join(cell(row.get(key)) for key, _ in COLUMNS) + " |")
    return "\n".join(lines)
