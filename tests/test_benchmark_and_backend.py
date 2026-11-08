"""Unit tests for the benchmark scoring and the LLM backend factory (no network)."""

import json
from pathlib import Path

import pytest

import config
import llm_backend
from benchmark.scoring import citations, score_answer, summarize, to_markdown

FALLBACK = "Dazu habe ich leider keine genauen Informationen in meinen THB-Quellen."


# ── scoring ───────────────────────────────────────────────────────────────────

def test_citations_are_parsed():
    assert citations("Bis **15. Juli** [1][3], siehe auch [3].") == {1, 3}


def test_good_answer_passes_all_checks():
    result = score_answer("Die Rückmeldung läuft bis 15. Juli [2].", expect="answer",
                          n_sources=8, fallback_text=FALLBACK, language_ok=True)
    assert result["quality"] == 1.0


def test_invented_source_number_is_caught():
    result = score_answer("Siehe [12].", expect="answer", n_sources=8, fallback_text=FALLBACK)
    assert result["checks"]["citations_valid"] is False


def test_answer_without_citation_loses_points():
    result = score_answer("Bis Juli.", expect="answer", n_sources=8, fallback_text=FALLBACK)
    assert result["checks"]["cites_source"] is False
    assert result["quality"] < 1.0


def test_keywords_are_checked_only_when_given():
    no_keywords = score_answer("Ja [1].", expect="answer", n_sources=1, fallback_text=FALLBACK)
    assert "keywords" not in no_keywords["checks"]
    missing = score_answer("Ja [1].", expect="answer", n_sources=1, fallback_text=FALLBACK,
                           keywords=["5 ECTS"])
    assert missing["checks"]["keywords"] is False


def test_unanswerable_question_requires_fallback_without_citations():
    good = score_answer(FALLBACK + " Bitte wende dich an das Prüfungsamt.", expect="fallback",
                        n_sources=8, fallback_text=FALLBACK)
    invented = score_answer("Das Passwort ist thb2026 [1].", expect="fallback",
                            n_sources=8, fallback_text=FALLBACK)
    assert good["quality"] == 1.0
    assert invented["quality"] == 0.0


def test_summary_ranks_models_by_quality_and_ignores_errors():
    rows = [
        {"model": "a", "quality": 0.5, "checks": {"cites_source": False}, "latency_s": 2.0,
         "ttft_s": 0.5, "tokens_per_s": 10.0},
        {"model": "b", "quality": 1.0, "checks": {"cites_source": True}, "latency_s": 4.0,
         "ttft_s": 1.0, "tokens_per_s": 5.0},
        {"model": "b", "quality": 0.0, "checks": {}, "error": "timeout"},
    ]
    summary = summarize(rows)
    assert [s["model"] for s in summary] == ["b", "a"]
    assert summary[0]["errors"] == 1 and summary[0]["quality"] == 1.0
    assert "| b | 1.00 |" in to_markdown(summary)


def test_question_set_is_well_formed():
    path = Path(__file__).resolve().parent.parent / "benchmark" / "questions.jsonl"
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({q["id"] for q in questions}) == len(questions)
    assert {q["expect"] for q in questions} == {"answer", "fallback"}
    assert {q["lang"] for q in questions} == {"de", "en", "ar"}


# ── backend factory ───────────────────────────────────────────────────────────

def test_openai_backend_points_at_the_gateway(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://litellm:4000/v1")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen2.5-7b")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")

    llm = llm_backend.make_llm()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "qwen2.5-7b"
    assert llm_backend.health_request() == (
        "http://litellm:4000/v1/models", {"Authorization": "Bearer sk-test"})


def test_ollama_backend_is_the_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/")
    assert type(llm_backend.make_llm()).__name__ == "ChatOllama"
    assert llm_backend.health_request() == ("http://localhost:11434/api/tags", {})


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        llm_backend.make_llm()
