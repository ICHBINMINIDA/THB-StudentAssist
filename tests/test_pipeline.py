"""Fast unit tests for the data pipeline and retrieval (no models or LLM needed)."""

from ingest import build_documents, split_documents
from prepare_sources import clean_text, group_module_pages, html_to_text, structure_faq_text
from retrieval import BM25, tokenize


# ── prepare_sources ───────────────────────────────────────────────────────────

def test_clean_text_keeps_real_words_and_drops_ui_noise():
    raw = "Analyse\nDatenanalyse und Analyse von Algorithmen\n42\nCookie optin by Olli machts\nInhalt"
    assert clean_text(raw) == "Datenanalyse und Analyse von Algorithmen\nInhalt"


def test_html_to_text_keeps_links_inline():
    html = '<main><p>Die Formulare sind <a href="/formulare/">hier</a> zu finden.</p></main>'
    text = html_to_text(html, "https://www.th-brandenburg.de/studium/")
    assert text == "Die Formulare sind hier (https://www.th-brandenburg.de/formulare/) zu finden."


def test_structure_faq_text_skips_table_of_contents():
    text = "Prüfungen an/abmelden\nNoten\nWie melde ich mich ab?\nPer Formular.\nBis 14 Tage vorher.\nWas ist ein Freiversuch?\nEin Versuch."
    assert structure_faq_text(text) == [
        {"question": "Wie melde ich mich ab?", "answer": "Per Formular.\nBis 14 Tage vorher."},
        {"question": "Was ist ein Freiversuch?", "answer": "Ein Versuch."},
    ]


def test_group_module_pages():
    pages = [
        {"page": 1, "text": "Inhaltsverzeichnis"},
        {"page": 2, "text": "Modulbezeichnung: Datenbanken\nECTS: 5"},
        {"page": 3, "text": "Literatur: ..."},
        {"page": 4, "text": "Modulbezeichnung: Mathematik 1"},
    ]
    entries = group_module_pages(pages)
    assert [(e["module"], e["pages"]) for e in entries] == [
        ("", [1]), ("Datenbanken", [2, 3]), ("Mathematik 1", [4]),
    ]


# ── ingest ────────────────────────────────────────────────────────────────────

def test_faq_documents_become_qa_chunks_with_context_header():
    data = {
        "id": "faq", "title": "Prüfungsamt FAQ", "source": "https://example.org/faq",
        "structured_content": {"format": "faq", "qa_pairs": [{"question": "Wann?", "answer": "Im Juli."}]},
    }
    chunks = split_documents(build_documents(data))
    assert len(chunks) == 1
    assert chunks[0].page_content == "[Prüfungsamt FAQ]\nFrage: Wann?\nAntwort: Im Juli."
    assert chunks[0].metadata["doc_id"] == "faq" and chunks[0].metadata["chunk_index"] == 0


def test_module_chunks_name_their_module():
    long_text = "Modulbezeichnung: Datenbanken\n" + "Inhalt der Vorlesung. " * 60
    data = {
        "id": "mod", "title": "Modulbeschreibung", "source": "sources/pdf/x.pdf",
        "structured_content": {"format": "modules", "modules": [{"module": "Datenbanken", "page": 7, "text": long_text}]},
    }
    chunks = split_documents(build_documents(data))
    assert len(chunks) > 1
    assert all(c.page_content.startswith("[Modulbeschreibung | Modul: Datenbanken]") for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ── retrieval ─────────────────────────────────────────────────────────────────

def test_tokenize_removes_stopwords():
    assert tokenize("Wie melde ich mich von der Prüfung ab?") == ["melde", "prüfung", "ab"]


def test_bm25_ranks_matching_document_first():
    corpus = [tokenize(t) for t in [
        "Rückmeldung zum Wintersemester vom 01. Juni bis 25. Juli",
        "Modul Datenbanken hat 5 ECTS-Leistungspunkte",
        "Die CampusCard muss validiert werden",
    ]]
    bm25 = BM25(corpus)
    assert bm25.top_n(tokenize("Wie viele ECTS hat Datenbanken?"), 3)[0] == 1
    assert bm25.top_n(tokenize("Wann ist die Rückmeldung?"), 3)[0] == 0
