"""
ingest.py: turn clean_documents/*.json into searchable chunks in Chroma.

- FAQ pages: one document per question/answer pair.
- Module handbook: one document per module (instead of per page).
- Every chunk is prefixed with a short context header (source, module, question)
  so a chunk like "ECTS-Leistungspunkte: 5" still says which module it is about.
- Incremental: only changed files are re-embedded; their old chunks are
  deleted first, and chunks of removed files are deleted too.

Usage:
    python ingest.py           # incremental update
    python ingest.py --force   # full rebuild
"""

import argparse
import hashlib
import json
import logging
import os

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

HASH_FILE = os.path.join(config.VECTOR_DIR, ".ingest_hashes.json")
# Bump when chunking/metadata logic changes so existing stores get rebuilt.
SCHEMA_VERSION = 2
BATCH_SIZE = 100  # Chroma's max batch size is platform dependent (166 on Windows)


# ── Hash bookkeeping ──────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_state() -> dict:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("schema") == SCHEMA_VERSION and state.get("embedding_model") == config.EMBEDDING_MODEL:
            return state
    return {}


def save_state(hashes: dict) -> None:
    os.makedirs(config.VECTOR_DIR, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"schema": SCHEMA_VERSION, "embedding_model": config.EMBEDDING_MODEL, "files": hashes},
            f, indent=2,
        )


# ── Document building ─────────────────────────────────────────────────────────

def sanitize(metadata: dict) -> dict:
    """Chroma accepts only str/int/float/bool metadata values."""
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def build_documents(data: dict) -> list[Document]:
    base = {
        "doc_id": data.get("id", ""),
        "title": data.get("title", "Unknown title"),
        "source": data.get("source", ""),
        "type": data.get("type", "unknown"),
        "category": data.get("category", "unknown"),
        "language": data.get("language", "unknown"),
        "priority": data.get("priority", "normal"),
    }
    structured = data.get("structured_content") or {}
    fmt = structured.get("format")
    docs: list[Document] = []

    if fmt == "faq" and structured.get("qa_pairs"):
        for pair in structured["qa_pairs"]:
            docs.append(Document(
                page_content=f"Frage: {pair['question']}\nAntwort: {pair['answer']}",
                metadata={**base, "question": pair["question"]},
            ))

    elif fmt == "modules" and structured.get("modules"):
        for entry in structured["modules"]:
            docs.append(Document(
                page_content=entry["text"],
                metadata={**base, "module": entry.get("module", ""), "page": entry.get("page", "")},
            ))

    else:
        content = data.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("text", "").strip():
                    docs.append(Document(page_content=item["text"],
                                         metadata={**base, "page": item.get("page", "")}))
        elif isinstance(content, str) and content.strip():
            docs.append(Document(page_content=content, metadata=base))

    return docs


def context_header(meta: dict, text: str) -> str:
    parts = [meta["title"]]
    if meta.get("module"):
        parts.append(f"Modul: {meta['module']}")
    if meta.get("question") and not text.startswith("Frage:"):
        parts.append(f"Frage: {meta['question']}")
    return "[" + " | ".join(parts) + "]"


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )
    chunks = splitter.split_documents(docs)

    counters: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata = sanitize(chunk.metadata)
        chunk.page_content = f"{context_header(chunk.metadata, chunk.page_content)}\n{chunk.page_content}"
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def reset_store() -> None:
    """Delete every collection in the persist directory (incl. the legacy one)."""
    client = chromadb.PersistentClient(path=config.VECTOR_DIR)
    for col in client.list_collections():
        name = col if isinstance(col, str) else col.name
        client.delete_collection(name)
        logger.info("Deleted collection '%s'", name)


def main(force: bool = False) -> None:
    state = {} if force else load_state()
    if not state:
        logger.info("Full rebuild (forced, first run, or schema/model changed).")
        if os.path.exists(config.VECTOR_DIR):
            reset_store()

    old_hashes: dict = state.get("files", {})
    new_hashes: dict = {}
    changed: list[str] = []

    for filename in sorted(os.listdir(config.CLEAN_DIR)):
        if filename.endswith(".json"):
            new_hashes[filename] = file_hash(os.path.join(config.CLEAN_DIR, filename))
            if old_hashes.get(filename) != new_hashes[filename]:
                changed.append(filename)

    removed = [f for f in old_hashes if f not in new_hashes]

    if not changed and not removed:
        logger.info("No changes detected. Vector store is up to date.")
        return

    vectorstore = get_vectorstore()

    # Drop stale chunks of changed/removed files before re-adding.
    for filename in changed + removed:
        if filename in old_hashes:
            vectorstore.delete(where={"doc_id": filename[:-5]})
    for filename in removed:
        logger.info("Removed: %s", filename)

    total = 0
    for filename in changed:
        with open(os.path.join(config.CLEAN_DIR, filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("id", filename[:-5])
        chunks = split_documents(build_documents(data))
        ids = [f"{c.metadata['doc_id']}:{c.metadata['chunk_index']}" for c in chunks]
        for i in range(0, len(chunks), BATCH_SIZE):
            vectorstore.add_documents(chunks[i:i + BATCH_SIZE], ids=ids[i:i + BATCH_SIZE])
        logger.info("Indexed %-36s %4d chunks", filename, len(chunks))
        total += len(chunks)

    save_state(new_hashes)
    logger.info("Done. %d chunks (re)indexed from %d file(s).", total, len(changed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clean documents into the vector store")
    parser.add_argument("--force", action="store_true", help="Force a full rebuild")
    main(force=parser.parse_args().force)
