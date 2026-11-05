"""
Hybrid retrieval: dense vector search + BM25 keyword search, merged with
Reciprocal Rank Fusion (RRF).

Why hybrid: the multilingual embedding model is good at paraphrases
("Wie melde ich mich ab?" ~ "Rücktritt von Prüfungen") but weak on exact terms
such as module names, form names, dates or abbreviations ("ECTS", "SPO 2025"),
which BM25 handles well. The BM25 index is built from the documents already
stored in Chroma, so there is a single source of truth.
"""

import logging
import math
import re
from collections import Counter, defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

RRF_K = 60

STOPWORDS = set("""
der die das den dem des ein eine einer eines einem einen und oder aber ist sind war
wird werden wurde kann können muss müssen ich du er sie es wir ihr mich mir dich dir
mein meine meinen sich zu zum zur im in an am auf aus bei mit nach von vor für über
unter um wie was wo wann warum welche welcher welches wer gibt es nicht kein keine
auch noch nur so dass als wenn ob bis hier da dann denn doch ja nein habe hat haben
the a an and or of to in on at for is are was be how what where when why which who
do does can i my me you your it this that with from by as
""".split())

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    """Minimal Okapi BM25 (avoids an extra dependency for ~1.5k chunks)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / len(corpus)) if corpus else 0.0
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(doc))
        n = len(corpus)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self.postings[term].append(i)

    def top_n(self, query_tokens: list[str], n: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.doc_freqs[i][term]
                norm = 1 - self.b + self.b * self.doc_lens[i] / self.avgdl
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return [i for i, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]]


class HybridRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.refresh()

    def refresh(self) -> None:
        """(Re)build the BM25 index from the current Chroma collection."""
        data = self.vectorstore.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs = {
            id_: Document(id=id_, page_content=text, metadata=meta or {})
            for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self.bm25 = BM25([tokenize(text) for text in data["documents"]])
        logger.info("Hybrid retriever ready: %d chunks indexed.", len(self.ids))
        if not self.ids:
            logger.warning("The vector store is empty. Run `python ingest.py` first.")

    def search(self, queries: list[tuple[str, float]], k: int = config.TOP_K) -> list[Document]:
        """
        `queries` is a list of (query, weight). Each query is run through both
        retrievers; all rankings are fused with weighted RRF.
        """
        if not self.ids:
            return []

        fused: dict[str, float] = defaultdict(float)
        for query, weight in queries:
            dense = self.vectorstore.similarity_search(query, k=config.CANDIDATE_K)
            sparse = [self.ids[i] for i in self.bm25.top_n(tokenize(query), config.CANDIDATE_K)]
            for ranking in ([d.id for d in dense], sparse):
                for rank, doc_id in enumerate(ranking):
                    fused[doc_id] += weight / (RRF_K + rank + 1)

        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

        if config.DEBUG_RETRIEVAL:
            logger.info("Retrieval for %r", queries[0][0])
            for doc_id, score in best:
                logger.info("  %.4f  %s  %s", score, doc_id, self.docs[doc_id].page_content[:90].replace("\n", " "))

        return [self.docs[doc_id] for doc_id, _ in best if doc_id in self.docs]
