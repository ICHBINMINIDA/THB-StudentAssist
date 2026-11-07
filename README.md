# THB StudentAssist

**A multilingual RAG chatbot that answers student questions about the Technische Hochschule Brandenburg (THB): re-registration, exams, modules, semester dates and study programs. It runs fully locally and cites its sources.**

[![tests](https://github.com/ICHBINMINIDA/THB-StudentAssist/actions/workflows/tests.yml/badge.svg)](https://github.com/ICHBINMINIDA/THB-StudentAssist/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.1-black)
![LiteLLM](https://img.shields.io/badge/gateway-LiteLLM-6E56CF)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Disclaimer:** This is an independent student project. It is not affiliated with or endorsed by the Technische Hochschule Brandenburg. Always verify deadlines and fees on the official THB website.

<p align="center">
  <img src="docs/screenshots/answer-semesterticket.png" alt="THB StudentAssist answering a question about the Semesterticket refund, with links and cited sources" width="760">
</p>

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/welcome.png" alt="Start screen with suggested questions"></td>
    <td width="50%"><img src="docs/screenshots/answer-exmatrikulation.png" alt="Answer about exmatriculation after the final exam, with numbered citations"></td>
  </tr>
  <tr>
    <td align="center"><sub>Start screen with suggested questions</sub></td>
    <td align="center"><sub>Cited answer: each statement links to its source</sub></td>
  </tr>
</table>

---

## Features

- **Grounded answers with citations.** Every THB-specific statement is backed by a numbered source `[1]`. Citations link to the FAQ page or to the exact page of the PDF.
- **Says "I don't know".** If the sources don't contain the answer, the bot refers students to the Studierendensekretariat or Prüfungsamt instead of guessing.
- **Multilingual.** Ask in German, English or Arabic; the answer comes in the same language, even though all sources are German.
- **Understands follow-ups.** "Wie viele ECTS hat Datenbanken?" followed by "und wie viele SWS?" still finds the right module.
- **Hybrid search.** Semantic vector search combined with BM25 keyword search, so it handles both paraphrases and exact terms (module names, "ECTS", dates).
- **Streaming chat UI.** Answers appear token by token, with Markdown rendering, dark mode, a mobile layout and suggested questions.
- **Runs locally.** Llama 3.1 via Ollama and on-device embeddings; no API keys, no data leaves the machine.

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline data pipeline"]
        S[sources.json<br/>web pages + PDFs] --> P[prepare_sources.py<br/>fetch · clean · structure]
        P --> C[clean_documents/*.json]
        C --> I[ingest.py<br/>chunk · add context headers · embed]
        I --> V[(Chroma<br/>vector store)]
    end

    subgraph Online["Per question"]
        Q[Student question<br/>+ chat history] --> R[Hybrid retriever<br/>vector + BM25 → RRF]
        V --> R
        R --> L[Prompt with numbered sources<br/>Llama 3.1 via Ollama]
        L --> A[Streamed answer<br/>+ cited sources]
    end
```

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 (8B) served by [Ollama](https://ollama.com) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Vector store | Chroma |
| Keyword search | Own BM25 implementation, fused with Reciprocal Rank Fusion |
| Orchestration | LangChain |
| Backend | FastAPI (NDJSON streaming, rate limiting with slowapi) |
| Frontend | Plain HTML/CSS/JS, no build step |

## Getting started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download)

### 1. Install

```bash
git clone https://github.com/ICHBINMINIDA/THB-StudentAssist.git
cd THB-StudentAssist
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1
```

### 2. Configure

```bash
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Set `OLLAMA_BASE_URL` to wherever Ollama runs. Its default is `http://localhost:11434`.

### 3. Get the THB documents

THB documents are not redistributed in this repository. Download these PDFs from the [THB website](https://www.th-brandenburg.de) into `sources/pdf/`, using the file names listed in [`sources/metadata/sources.json`](sources/metadata/sources.json):

| File | Where to find it |
|---|---|
| `Websteite_Rahmentermine_SoSe26-SoSe28.pdf` | [Rahmentermine](https://www.th-brandenburg.de/hochschule/termine-veranstaltungen/rahmentermine/) |
| `Modulhandbuch-FBI-Praesenz-2025-Nr2_SoSe2026.pdf` | Module handbook of the Fachbereich Informatik und Medien |
| `Studiengänge.pdf` | Overview of study programs |

The web pages (FAQs, Rückmeldung) are downloaded automatically. Missing PDFs are skipped with a warning, so you can start with only some of them.

### 4. Build the knowledge base

```bash
python prepare_sources.py      # fetch + clean       -> clean_documents/
python ingest.py               # chunk + embed       -> vectorstore/
python check_db.py "Wann ist die Rückmeldung?"   # optional: inspect search results
```

### 5. Run

```bash
uvicorn app:app --reload       # open http://127.0.0.1:8000
python chatbot.py              # or chat in the terminal
```

## Docker: app + LiteLLM gateway + Ollama

`docker-compose.yml` runs the full stack. The app no longer talks to a model
engine directly: it calls an OpenAI-compatible **LiteLLM** gateway, which routes
each model name to the engine that serves it. Swapping or adding a model is a
change to [`deploy/litellm.yaml`](deploy/litellm.yaml), not to the code.

```mermaid
flowchart LR
    U[Browser] --> A[app<br/>FastAPI + RAG]
    W[Open WebUI<br/>optional] --> G
    B[benchmark] --> G
    A -->|OpenAI API| G[LiteLLM gateway<br/>:4000]
    G --> O[Ollama<br/>GPU server via SSH tunnel<br/>or local container]
    G -.->|GPU override| V[vLLM]
```

The gateway can use **any Ollama server**, set by `OLLAMA_API_BASE` in `.env`.
By default it is a remote GPU server reached through an SSH tunnel on the
host, so nothing heavy runs locally:

```bash
cp .env.example .env                               # set LITELLM_MASTER_KEY
ssh -N -L 18080:localhost:11434 <user>@<gpu-server> # keep this terminal open

docker compose up -d litellm
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" http://localhost:4000/v1/models

# Build the knowledge base inside the app container (PDFs from sources/pdf/)
docker compose run --rm app python prepare_sources.py
docker compose run --rm app python ingest.py

docker compose up -d                               # http://localhost:8000
docker compose --profile ui up -d                  # + Open WebUI on http://localhost:3000
```

**Local Ollama instead of a server:** set `OLLAMA_API_BASE=http://ollama:11434`,
then `docker compose --profile local-ollama up -d` and
`docker compose exec ollama ollama pull llama3.1`.

The models listed in [`deploy/litellm.yaml`](deploy/litellm.yaml) must exist on
that Ollama server (`ollama list`). `num_ctx: 8192` is sent with every request,
so the server's default context window cannot truncate the RAG prompt.

| Service | Image | Purpose |
|---|---|---|
| `ollama` | `ollama/ollama` | Optional local model engine (`--profile local-ollama`) |
| `litellm` | `ghcr.io/berriai/litellm` | One OpenAI-compatible API and key for every model |
| `app` | built from `Dockerfile` | The chatbot (non-root, CPU-only PyTorch for the embeddings) |
| `open-webui` | `ghcr.io/open-webui/open-webui` | Optional general chat UI on the same gateway (`--profile ui`) |

**NVIDIA GPU:** `docker-compose.gpu.yml` gives Ollama the GPU and adds a **vLLM**
server (`Qwen2.5-7B-Instruct`) behind the same gateway as `qwen2.5-7b-vllm`.
It needs the NVIDIA driver and `nvidia-container-toolkit`:

```bash
# with OLLAMA_API_BASE=http://ollama:11434 in .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile local-ollama up -d
```

Without Docker, nothing changes: `LLM_PROVIDER=ollama` (the default) talks to
Ollama directly, as before.

## Benchmark: which model fits this task?

[`benchmark/run_benchmark.py`](benchmark/run_benchmark.py) compares models on
the real StudentAssist task instead of a generic leaderboard. Every model gets
exactly the same prompt per question (same retrieved sources, same system
prompt), so differences come from the model alone.

```bash
pip install -r requirements.txt
python -m benchmark.run_benchmark \
    --models llama3.1-8b=llama3.1,qwen2.5-7b=qwen2.5:7b,gemma3-4b=gemma3:4b \
    --api-key "$LITELLM_MASTER_KEY" --ollama-url http://localhost:18080 --repeats 3
```

It measures per model:

| Metric | How |
|---|---|
| **Quality** | Share of passed checks per answer (below), averaged |
| Says "don't know" | On questions the sources cannot answer, the model must give the fixed fallback and cite nothing |
| Cites sources | Answerable questions must cite at least one source, and never a source number that does not exist |
| Right language | German, English or Arabic, matching the question |
| TTFT / latency / tokens/s | Time to first token, total time, generation speed (median, streamed) |
| Memory | RAM/VRAM of the loaded model, read from Ollama's `/api/ps` |

The first request per model loads it and is never measured. Results go to
`benchmark/results/<timestamp>.md` (table) and `.jsonl` (every answer, for
inspection).

The question set ([`benchmark/questions.jsonl`](benchmark/questions.jsonl))
covers all three languages. The `fallback` questions assume the topic is not in
the indexed sources; check that this still holds when you add sources. Add
`keywords` to a question to also require facts from the official answer.

## Adding sources

Add an entry to `sources/metadata/sources.json` (`type`: `web` or `pdf`), then rerun steps 4 and 5. `ingest.py` is incremental: it only re-embeds changed files and removes their outdated chunks first.

## How it works

Notable decisions in the retrieval pipeline:

1. **Structure-aware chunking.** FAQ pages are split into question/answer pairs, and the 480-page module handbook is grouped into one entry per module (127 modules). Each chunk starts with a context header such as `[Modulbeschreibung | Modul: Datenbanken]`, so even a chunk that only says "ECTS-Leistungspunkte: 5" knows which module it belongs to.
2. **Chunk size matches the embedding model.** The multilingual MiniLM model reads only 128 tokens. With the original 900-character chunks, about half of every chunk was never embedded. Chunks are now 450 characters.
3. **Hybrid retrieval.** Dense and BM25 rankings are merged with Reciprocal Rank Fusion. For short follow-up questions, the previous question is added as a second, higher-weighted query.
4. **Prompting.** System prompt plus chat history as real chat messages, numbered sources, an explicit answer language and a fixed fallback sentence. Only sources the model actually cites are shown. Ollama's context window is raised from its 2048-token default to 8192, so long prompts aren't silently truncated.

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask/stream` | `{"question": "..."}` → NDJSON stream of `token` events, then `done` with sources |
| `POST` | `/ask` | Same, as a single JSON response `{answer, sources}` |
| `POST` | `/clear` | Clear the conversation history of the current session |
| `GET` | `/health` | Liveness check; also reports whether the LLM backend is reachable |
| `GET` | `/pdf/{file}` | Serves source PDFs so citations can link to a page |

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (direct) or `openai` (LiteLLM, vLLM or any OpenAI-compatible API) |
| `OLLAMA_BASE_URL` | `http://localhost:18080` | Ollama server (`ollama` provider) |
| `OLLAMA_MODEL` | `llama3.1` | Chat model (`ollama` provider) |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | `http://localhost:4000/v1` / `llama3.1-8b` / – | Endpoint, model name and key (`openai` provider) |
| `EMBEDDING_MODEL` | multilingual MiniLM | e.g. `intfloat/multilingual-e5-small` for longer inputs (re-ingest required) |
| `TOP_K` | `8` | Chunks given to the LLM |
| `MAX_HISTORY_TURNS` | `6` | Conversation turns kept per session |
| `RATE_LIMIT` | `20/minute` | Requests per IP |

## Project structure

```
├── app.py               FastAPI app (streaming, sessions, rate limiting)
├── chatbot.py           RAG pipeline: retrieval, prompting, sessions, CLI
├── retrieval.py         Hybrid retriever (vector + BM25 + RRF)
├── embeddings.py        Shared embedding model / Chroma factory
├── ingest.py            Chunking and incremental indexing
├── prepare_sources.py   Fetching and cleaning of web pages and PDFs
├── check_db.py          Inspect the index and test queries
├── config.py            Settings from environment / .env
├── llm_backend.py       Chat model factory: Ollama or OpenAI-compatible (LiteLLM / vLLM)
├── templates/index.html Chat UI
├── sources/metadata/    Source definitions
├── benchmark/           Model benchmark: question set, runner, scoring
├── deploy/              LiteLLM gateway configs (CPU and GPU)
├── Dockerfile           App image
├── docker-compose.yml   app + LiteLLM + Ollama (+ Open WebUI)
├── docker-compose.gpu.yml  NVIDIA override: GPU for Ollama + vLLM
└── tests/               Unit tests (pytest)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Roadmap

- [x] Docker Compose setup (app + LiteLLM gateway + Ollama, optional vLLM on GPU)
- [x] Model benchmark on the real task (quality, latency, memory)
- [ ] Grow the question set with real student questions and expected keywords
- [ ] Monitoring: Prometheus metrics from LiteLLM / vLLM and a Grafana dashboard
- [ ] More sources (fees, international office, other departments)
- [ ] Stronger embedding model or a cross-encoder re-ranker

## License

The code is released under the [MIT License](LICENSE). THB documents and web content belong to the Technische Hochschule Brandenburg and are not included.
