"""
prepare_sources.py: fetch, clean and save source documents for ingestion.

Pipeline:  sources/metadata/sources.json  ->  clean_documents/<id>.json

- Web pages: fetched with retries, raw HTML cached in sources/cache/ (so that
  improvements to the cleaning step never require a re-download), main content
  extracted, inline links kept as "text (url)" so the bot can point to them.
- FAQ pages: additionally split into question/answer pairs.
- PDFs: extracted page by page.
- Module handbook: pages grouped into one entry per module.

Usage:
    python prepare_sources.py              # use cached HTML when available
    python prepare_sources.py --no-cache   # re-download all web pages
"""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import CLEAN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_sources")

SOURCES_FILE = "sources/metadata/sources.json"
CACHE_DIR = "sources/cache"

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
USER_AGENT = "THB-StudentAssist-Bot/1.0"


# ── Text cleaning ─────────────────────────────────────────────────────────────

# Multi-word UI phrases that are safe to remove wherever they appear.
NOISE_PHRASES = [
    "Direkt zur Hauptnavigation",
    "Direkt zum Inhalt",
    "Direkt zur Suche",
    "Print this page",
    "Download this page as PDF",
    "Wir verwenden Cookies",
    "Nur essentielle Cookies akzeptieren",
    "Speichern und schließen",
    "Mehr über die genutzten Cookies erfahren",
    "Cookie optin by Olli machts",
]

# Single words are only dropped when they form a whole line on their own.
# (Removing them as substrings destroyed real content, e.g. every "Analyse"
# in the module handbook.)
SKIP_LINES = {
    "Home", "Intern", "Suchen", "Navigation", "DE", "EN", "de", "en", "sw",
    "A", "Funktional", "Analyse", "Alle akzeptieren", "Alle öffnen", "Alle schließen",
}

PDF_STAMP = re.compile(r"\b\d{2}-\d{2}-\d{2}\s+\d+/\d+\s+PDF\b")


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r", "\n").replace("\t", " ")
    text = text.replace("­", "")  # soft hyphens from PDFs

    for phrase in NOISE_PHRASES:
        text = text.replace(phrase, "")

    text = PDF_STAMP.sub("", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ ]{2,}", " ", line.strip())
        if not line or line in SKIP_LINES:
            continue
        if re.fullmatch(r"\d+", line):  # stand-alone page numbers
            continue
        if len(line) <= 1 and line not in "•-–":
            continue
        lines.append(line)

    return "\n".join(lines).strip()


# ── Web fetching with raw-HTML cache ──────────────────────────────────────────

def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, f"{hashlib.md5(url.encode()).hexdigest()}.html")


def fetch_html(url: str, use_cache: bool = True) -> str:
    path = _cache_path(url)
    if use_cache and os.path.exists(path):
        logger.info("  [cache hit] %s", url)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            html = response.text
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            return html
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("  Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise last_error  # type: ignore[misc]


INLINE_TAGS = ["a", "strong", "b", "em", "i", "span", "u", "abbr", "small", "sup", "sub", "code"]


def html_to_text(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup

    # Keep useful link targets ("Formulare sind hier (https://...) zu finden").
    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        label = a.get_text(" ", strip=True)
        if href.startswith(("http", "/")) and label:
            a.string = f"{label} ({urljoin(base_url, href)})"

    # Unwrap inline elements so that a sentence containing a link is not split
    # across several lines.
    for tag in main.find_all(INLINE_TAGS):
        tag.unwrap()
    main.smooth()

    for br in main.find_all("br"):
        br.replace_with("\n")

    return clean_text(main.get_text(separator="\n"))


def extract_web_source(source: dict, use_cache: bool = True) -> str:
    html = fetch_html(source["url"], use_cache=use_cache)
    return html_to_text(html, source["url"])


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_source(source: dict) -> list[dict]:
    file_path = source["file"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(file_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_number, "text": text})

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {file_path}")
    return pages


# ── FAQ structuring ───────────────────────────────────────────────────────────

QUESTION_STARTERS = (
    "Wie", "Wo", "Was", "Wann", "Warum", "Welche", "Welcher", "Welches",
    "Woran", "Wer", "Kann", "Muss", "Gibt", "Ist", "Bekomme", "Brauche", "Wird",
)


def _is_question(line: str) -> bool:
    if len(line) > 250:
        return False
    if line.endswith("?"):
        return True
    return line.startswith(tuple(w + " " for w in QUESTION_STARTERS)) and len(line) < 160


def structure_faq_text(text: str) -> list[dict]:
    """Parse FAQ text into [{question, answer}], skipping the table of contents."""
    qa_pairs: list[dict] = []
    question: str | None = None
    answer: list[str] = []

    def flush():
        if question and answer:
            qa_pairs.append({"question": question, "answer": "\n".join(answer).strip()})

    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if _is_question(line):
            flush()
            question, answer = line, []
        elif question:
            answer.append(line)
    flush()
    return qa_pairs


# ── Module handbook structuring ───────────────────────────────────────────────

MODULE_HEADER = re.compile(r"Modulbezeichnung:\s*(.+)")


def group_module_pages(pages: list[dict]) -> list[dict]:
    """
    Group handbook pages into one entry per module. A module starts on the page
    containing "Modulbezeichnung:" and runs until the next module starts. Pages
    before the first module (cover, table of contents, study plans) are kept as
    individual entries.
    """
    entries: list[dict] = []
    current: dict | None = None

    for page in pages:
        match = MODULE_HEADER.search(page["text"])
        if match:
            if current:
                entries.append(current)
            current = {
                "module": match.group(1).strip(),
                "page": page["page"],
                "pages": [page["page"]],
                "text": page["text"],
            }
        elif current:
            current["pages"].append(page["page"])
            current["text"] += "\n" + page["text"]
        else:
            entries.append({"module": "", "page": page["page"], "pages": [page["page"]], "text": page["text"]})

    if current:
        entries.append(current)
    return entries


# ── Output ────────────────────────────────────────────────────────────────────

def save_clean_document(source: dict, content, structured_content=None) -> None:
    os.makedirs(CLEAN_DIR, exist_ok=True)
    output_path = os.path.join(CLEAN_DIR, f"{source['id']}.json")

    clean_doc = {
        "id": source["id"],
        "type": source["type"],
        "title": source["title"],
        "category": source.get("category", "unknown"),
        "language": source.get("language", "unknown"),
        "priority": source.get("priority", "normal"),
        "source": source.get("url", source.get("file", "")),
        "content": content,
    }
    if structured_content:
        clean_doc["structured_content"] = structured_content

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_doc, f, ensure_ascii=False, indent=2)
    logger.info("  Saved: %s", output_path)


def process_source(source: dict, use_cache: bool) -> None:
    structured = None

    if source["type"] == "web":
        content = extract_web_source(source, use_cache=use_cache)
        if "faq" in source.get("url", "").lower():
            qa_pairs = structure_faq_text(content)
            structured = {"format": "faq", "qa_pairs": qa_pairs}
            logger.info("  Extracted %d FAQ pairs.", len(qa_pairs))

    elif source["type"] == "pdf":
        content = extract_pdf_source(source)
        if source.get("category") == "modules":
            modules = group_module_pages(content)
            structured = {"format": "modules", "modules": modules}
            logger.info("  Grouped %d pages into %d module entries.",
                        len(content), sum(1 for m in modules if m["module"]))
    else:
        raise ValueError(f"Unknown source type: {source['type']}")

    save_clean_document(source, content, structured)


def main(no_cache: bool = False) -> None:
    with open(SOURCES_FILE, "r", encoding="utf-8") as f:
        sources = json.load(f)
    logger.info("Loaded %d sources.", len(sources))

    errors: list[tuple[str, str]] = []
    for source in sources:
        logger.info("Processing: [%s] %s", source["type"].upper(), source["title"])
        try:
            process_source(source, use_cache=not no_cache)
        except Exception as exc:  # keep going with the remaining sources
            logger.error("  ERROR: %s", exc)
            errors.append((source["title"], str(exc)))

    logger.info("Done. %d/%d sources processed successfully.", len(sources) - len(errors), len(sources))
    for title, err in errors:
        logger.warning("  Failed: %s: %s", title, err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and clean sources")
    parser.add_argument("--no-cache", action="store_true", help="Re-download all web pages")
    main(no_cache=parser.parse_args().no_cache)
