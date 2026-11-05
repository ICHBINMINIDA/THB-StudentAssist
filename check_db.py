"""
Inspect the vector store.

    python check_db.py                     # chunk counts per source
    python check_db.py "Wie melde ich mich von einer Prüfung ab?"   # test retrieval
"""

import sys
from collections import Counter

from embeddings import get_vectorstore


def main() -> None:
    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["metadatas"])
    counts = Counter(meta.get("title", "?") for meta in data["metadatas"])

    print(f"{len(data['ids'])} chunks total\n")
    for title, n in counts.most_common():
        print(f"{n:5d}  {title}")

    if len(sys.argv) > 1:
        import config
        config.DEBUG_RETRIEVAL = False
        from retrieval import HybridRetriever

        question = " ".join(sys.argv[1:])
        print(f"\nTop results for: {question!r}\n")
        for i, doc in enumerate(HybridRetriever(vectorstore).search([(question, 1.0)]), start=1):
            print(f"[{i}] {doc.id}")
            print("    " + doc.page_content[:200].replace("\n", "\n    ") + "\n")


if __name__ == "__main__":
    main()
