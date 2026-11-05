"""
THB StudentAssist: retrieval-augmented chatbot core.

Run directly for a CLI chat:  python chatbot.py
"""

import asyncio
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import config
from embeddings import get_vectorstore
from retrieval import HybridRetriever, tokenize

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chatbot")


class LLMUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached or fails."""


# ── Models (loaded once) ──────────────────────────────────────────────────────

retriever = HybridRetriever(get_vectorstore())

llm = ChatOllama(
    model=config.OLLAMA_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=config.LLM_TEMPERATURE,
    num_ctx=config.LLM_NUM_CTX,
)


# ── Session store (bounded, expiring) ─────────────────────────────────────────

class SessionStore:
    """In-memory chat histories with TTL expiry and an upper bound on sessions."""

    def __init__(self, ttl: int, max_sessions: int, max_turns: int):
        self.ttl, self.max_sessions, self.max_turns = ttl, max_sessions, max_turns
        self._data: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.time()
        while self._data:
            key, (ts, _) = next(iter(self._data.items()))
            if now - ts > self.ttl or len(self._data) > self.max_sessions:
                self._data.pop(key)
            else:
                break

    def get(self, session_id: str) -> list[dict]:
        with self._lock:
            self._evict()
            entry = self._data.get(session_id)
            return list(entry[1]) if entry else []

    def append(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            history = self._data.pop(session_id, (0, []))[1]
            history.append({"user": user, "assistant": assistant})
            self._data[session_id] = (time.time(), history[-self.max_turns:])
            self._evict()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


sessions = SessionStore(config.SESSION_TTL_SECONDS, config.MAX_SESSIONS, config.MAX_HISTORY_TURNS)


# ── Language handling ─────────────────────────────────────────────────────────

GERMAN_HINTS = re.compile(
    r"[äöüß]|\b(ich|wie|wann|wo|was|welche|kann|muss|gibt|der|die|das|und|ist|mein|prüfung|modul|semester)\b",
    re.IGNORECASE,
)
ENGLISH_HINTS = re.compile(
    r"\b(how|what|when|where|which|can|do|does|is|are|the|my|i|you|exam|module|course)\b", re.IGNORECASE
)

LANGUAGE_NAMES = {"de": "German", "en": "English", "ar": "Arabic"}

FALLBACK = {
    "de": "Dazu habe ich leider keine genauen Informationen in meinen THB-Quellen. "
          "Bitte wende dich direkt an das Studierendensekretariat oder das Prüfungsamt.",
    "en": "I don't have precise information about this in my THB sources. "
          "Please contact the Studierendensekretariat or the Prüfungsamt directly.",
    "ar": "لا تتوفر لديّ هذه المعلومات بدقة في مصادر THB. يُرجى التواصل مع مكتب شؤون الطلاب أو مكتب الامتحانات.",
}


def detect_language(text: str, previous: str = "de") -> str:
    if re.search(r"[؀-ۿ]", text):
        return "ar"
    de, en = len(GERMAN_HINTS.findall(text)), len(ENGLISH_HINTS.findall(text))
    if de == en:
        return previous  # e.g. "Danke!" / "ok": keep the conversation language
    return "de" if de > en else "en"


# ── Prompting ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are THB StudentAssist, a friendly assistant for students of the Technische Hochschule Brandenburg (THB).

For questions about THB (enrollment, re-registration/Rückmeldung, fees, exams, modules, deadlines, study programs, campus services):
- Answer ONLY from the numbered THB sources provided in the user message. Never invent THB-specific facts (dates, deadlines, fees, rules, names, e-mail addresses, links).
- Cite the sources you used with their number in square brackets, e.g. [1] or [2][3], right after the statement they support.
- If the sources do not clearly answer the question, reply exactly with: "{fallback}"
- If the sources contain a relevant link, include it.

For greetings, small talk and general study questions (e.g. learning tips), answer briefly from general knowledge without citations.

Style:
- Answer in {language}, even though the sources are in German. Keep German official terms (e.g. Rückmeldung, Prüfungsamt) where helpful.
- Be concise and concrete. Use short bullet lists for steps or several items. Use **bold** for key dates and amounts.
- For deadlines, fees or legal matters, briefly suggest checking the official THB website for the latest information."""


def pdf_url(source_path: str, page) -> str:
    """Map a local PDF path to the URL under which app.py serves it."""
    url = "/pdf/" + quote(os.path.basename(source_path))
    return f"{url}#page={page}" if page not in ("", None) else url


def source_entry(doc: Document, number: int) -> dict:
    meta = doc.metadata
    source = meta.get("source", "")
    page = meta.get("page", "")
    title = meta.get("title", "Quelle")
    if meta.get("module"):
        title = f"{title}: {meta['module']}"
    elif meta.get("question"):
        title = f"{title}: {meta['question']}"
    url = source if source.startswith("http") else pdf_url(source, page) if source.endswith(".pdf") else ""
    entry = {"n": number, "title": title, "url": url}
    if page not in ("", None):
        entry["page"] = page
    return entry


def format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no THB sources found)"
    blocks = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        page_info = f" (S. {page})" if page not in ("", None) else ""
        blocks.append(f"[{i}]{page_info}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_messages(question: str, docs: list[Document], history: list[dict], lang: str) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT.format(
        fallback=FALLBACK[lang], language=LANGUAGE_NAMES[lang],
    ))]
    for turn in history:
        messages.append(HumanMessage(turn["user"]))
        # Old [n] citations refer to that turn's sources; drop them to avoid confusion.
        messages.append(AIMessage(re.sub(r"\s?\[\d+\]", "", turn["assistant"])))
    messages.append(HumanMessage(
        f"THB sources (may or may not contain the answer):\n\n{format_context(docs)}\n\n"
        f"---\nQuestion: {question}\n\n(Answer in {LANGUAGE_NAMES[lang]}.)"
    ))
    return messages


# ── Pipeline ──────────────────────────────────────────────────────────────────

def retrieve(question: str, history: list[dict]) -> list[Document]:
    """
    Follow-up questions ("und bis wann?") carry little meaning on their own, so
    the previous user question is used as an additional, lower-weighted query.
    """
    if not history:
        return retriever.search([(question, 1.0)])
    combined = f"{history[-1]['user']} {question}"
    # Short questions are usually follow-ups: trust the combined query more.
    is_followup = len(tokenize(question)) <= 4
    return retriever.search([(question, 0.5 if is_followup else 1.0),
                             (combined, 1.0 if is_followup else 0.5)])


def select_sources(answer: str, docs: list[Document], lang: str) -> list[dict]:
    """
    Show only the sources the model actually cited. Several chunks of the same
    FAQ entry / PDF page are merged into one entry listing all their numbers.
    """
    if FALLBACK[lang][:40] in answer:
        return []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}

    merged: dict[tuple, dict] = {}
    for i, doc in enumerate(docs, start=1):
        if i not in cited:
            continue
        entry = source_entry(doc, i)
        key = (entry["title"], entry["url"])
        if key in merged:
            merged[key]["refs"].append(i)
        else:
            entry["refs"] = [i]
            merged[key] = entry
    return list(merged.values())


def _prepare(question: str, session_id: str):
    history = sessions.get(session_id)
    previous_lang = detect_language(history[-1]["user"]) if history else "de"
    lang = detect_language(question, previous_lang)
    docs = retrieve(question, history)
    return docs, build_messages(question, docs, history, lang), lang


def answer_question(question: str, session_id: str = "default") -> dict:
    """Blocking variant (CLI and the non-streaming /ask endpoint)."""
    docs, messages, lang = _prepare(question, session_id)
    try:
        answer = llm.invoke(messages).content.strip()
    except Exception as exc:
        logger.exception("LLM call failed")
        raise LLMUnavailableError(str(exc)) from exc

    sessions.append(session_id, question, answer)
    return {"answer": answer, "sources": select_sources(answer, docs, lang)}


async def stream_answer(question: str, session_id: str = "default") -> AsyncIterator[dict]:
    """
    Streaming variant. Yields events:
      {"type": "token", "text": ...}   repeatedly
      {"type": "done", "sources": [...]}
    """
    # Embedding + BM25 are CPU-bound: keep them off the event loop.
    docs, messages, lang = await asyncio.to_thread(_prepare, question, session_id)

    parts: list[str] = []
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                parts.append(chunk.content)
                yield {"type": "token", "text": chunk.content}
    except Exception as exc:
        logger.exception("LLM streaming failed")
        raise LLMUnavailableError(str(exc)) from exc

    answer = "".join(parts).strip()
    sessions.append(session_id, question, answer)
    yield {"type": "done", "sources": select_sources(answer, docs, lang)}


def clear_history(session_id: str) -> None:
    sessions.clear(session_id)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("THB StudentAssist is ready. Type 'exit' to quit, 'clear' to reset history.\n")
    session = "cli-session"

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "clear":
            clear_history(session)
            print("History cleared.\n")
            continue

        try:
            result = answer_question(question, session_id=session)
        except LLMUnavailableError as exc:
            print(f"\n[Error] LLM not reachable at {config.OLLAMA_BASE_URL}: {exc}\n")
            continue

        print("\nTHB StudentAssist:")
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for src in result["sources"]:
                page_info = f" (page {src['page']})" if src.get("page") else ""
                refs = ", ".join(map(str, src["refs"]))
                print(f"  [{refs}] {src['title']}{page_info}: {src['url']}")
        print()
