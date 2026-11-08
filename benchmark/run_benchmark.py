"""
Compare chat models on the real THB StudentAssist task: quality, speed and memory.

Every model gets exactly the same prompt per question (same retrieved sources,
same system prompt), so differences come from the model alone. Models are
called through one OpenAI-compatible endpoint, normally the LiteLLM gateway
from docker-compose.yml, so Ollama and vLLM models can be compared side by side.

Usage (vector store must exist, see README):

    python -m benchmark.run_benchmark --models gemma3-4b,gemma3-12b,llama3.1-8b

    # also record memory use from Ollama: alias=ollama_tag
    python -m benchmark.run_benchmark \
        --models gemma3-4b=gemma3:4b,gemma3-12b=gemma3:12b \
        --ollama-url http://localhost:11434

Results are written to benchmark/results/<timestamp>.jsonl (raw) and .md (table).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from benchmark.scoring import score_answer, summarize, to_markdown  # noqa: E402

ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompts(questions: list[dict]) -> list[dict]:
    """Retrieve sources once per question; the prompt is then fixed for all models."""
    import chatbot  # heavy: loads the embedding model and the vector store

    prepared = []
    for q in questions:
        lang = q.get("lang") or chatbot.detect_language(q["question"])
        docs = chatbot.retrieve(q["question"], [])
        messages = chatbot.build_messages(q["question"], docs, [], lang)
        prepared.append({
            **q, "lang": lang, "n_sources": len(docs),
            "fallback_text": chatbot.FALLBACK[lang],
            "messages": [{"role": ROLES[m.type], "content": m.content} for m in messages],
        })
    return prepared


def chat_stream(client: httpx.Client, base_url: str, model: str, messages: list[dict]) -> dict:
    """One streamed completion; returns the answer and timing measurements."""
    payload = {
        "model": model, "messages": messages, "stream": True,
        "temperature": config.LLM_TEMPERATURE,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token = None
    parts: list[str] = []
    usage: dict = {}
    with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if first_token is None:
                        first_token = time.perf_counter()
                    parts.append(text)
    finished = time.perf_counter()

    tokens = usage.get("completion_tokens")
    generation_time = finished - (first_token or started)
    return {
        "answer": "".join(parts).strip(),
        "ttft_s": round(first_token - started, 3) if first_token else None,
        "latency_s": round(finished - started, 3),
        "completion_tokens": tokens,
        "tokens_per_s": round(tokens / generation_time, 2) if tokens and generation_time > 0 else None,
    }


def ollama_memory_gb(ollama_url: str, tag: str) -> float | None:
    """Memory of a loaded Ollama model (VRAM if offloaded to a GPU, else total size)."""
    try:
        loaded = httpx.get(f"{ollama_url.rstrip('/')}/api/ps", timeout=5).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return None
    for entry in loaded:
        if entry.get("name") == tag or entry.get("model") == tag:
            size = entry.get("size_vram") or entry.get("size")
            return round(size / 1024 ** 3, 2) if size else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True,
                        help="comma-separated gateway model names, optionally alias=ollama_tag")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:4000/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--questions", type=Path, default=Path(__file__).parent / "questions.jsonl")
    parser.add_argument("--repeats", type=int, default=1, help="runs per question and model")
    parser.add_argument("--ollama-url", help="Ollama URL, to record memory use per model")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    models = []
    for spec in args.models.split(","):
        alias, _, tag = spec.strip().partition("=")
        models.append((alias, tag or None))

    questions = build_prompts(load_questions(args.questions))
    from chatbot import detect_language  # already loaded by build_prompts

    print(f"{len(questions)} questions x {len(models)} models x {args.repeats} repeats")

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    rows: list[dict] = []
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        for model, ollama_tag in models:
            # Warm-up: the first request also loads the model; never measure it.
            try:
                chat_stream(client, args.base_url, model, [{"role": "user", "content": "Hallo"}])
            except httpx.HTTPError as exc:
                print(f"[{model}] warm-up failed: {exc}")
            memory = ollama_memory_gb(args.ollama_url, ollama_tag) if args.ollama_url and ollama_tag else None

            for q in questions:
                for repeat in range(args.repeats):
                    row = {"model": model, "id": q["id"], "repeat": repeat, "expect": q["expect"],
                           "lang": q["lang"], "memory_gb": memory}
                    try:
                        row.update(chat_stream(client, args.base_url, model, q["messages"]))
                        row.update(score_answer(
                            row["answer"], expect=q["expect"], n_sources=q["n_sources"],
                            fallback_text=q["fallback_text"], keywords=q.get("keywords", []),
                            language_ok=detect_language(row["answer"], q["lang"]) == q["lang"],
                        ))
                    except (httpx.HTTPError, ValueError) as exc:
                        row.update({"error": str(exc), "checks": {}, "quality": 0.0})
                    rows.append(row)
                    status = row.get("error") or f"quality={row['quality']:.2f} latency={row['latency_s']}s"
                    print(f"[{model}] {q['id']}#{repeat}: {status}")

    summary = summarize(rows)
    table = to_markdown(summary)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with (out_dir / f"{stamp}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"{stamp}.md").write_text(
        f"# Benchmark {stamp}\n\n{len(questions)} questions, {args.repeats} repeat(s), "
        f"endpoint `{args.base_url}`.\n\n{table}\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nWritten to {out_dir / stamp}.jsonl / .md")


if __name__ == "__main__":
    main()
