"""Shared embedding + vector store factory, used by both ingest.py and chatbot.py."""

import logging
import os
from functools import lru_cache

# chromadb 0.6.x + posthog>=6 logs bogus "Failed to send telemetry event" errors
# even when telemetry is disabled. Disable telemetry and mute that logger.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    encode_kwargs = {"normalize_embeddings": True}
    query_kwargs = {"normalize_embeddings": True}

    # E5-family models expect "query: " / "passage: " prefixes.
    if "e5" in config.EMBEDDING_MODEL.lower():
        encode_kwargs["prompt"] = "passage: "
        query_kwargs["prompt"] = "query: "

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_kwargs,
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
