"""Fetch the electropi.ai pages and save them as structured sections.

Fixed page list, not a crawler: the site is ~12 known pages, so a BFS
queue would be ceremony. The part that matters for everything downstream:
extraction keeps the heading structure instead of flattening the page,
because headings become chunk metadata and citation anchors.
"""
import hashlib
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

import config
from utils.files import save_json
from utils.text import clean_text, is_junk

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X 15_0) electropi-rag-assessment",
    "Accept-Language": "en-US,en;q=0.9",
}
STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "form", "svg", "button"]
HEADING_TAGS = ["h1", "h2", "h3", "h4"]
TEXT_TAGS = ["p", "li", "td", "th", "blockquote", "figcaption"]
# Card sub-headings that must stay INSIDE their card's section — otherwise
# every case study splits into anonymous "Challenge"/"Solution" fragments
# that no query can match to an industry.
GLUE_HEADINGS = {"challenge", "solution", "results", "key results", "outcome"}


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """Walk the page in document order; each heading starts a new section."""
    # Not `main`: on this site <main> is a nearly-empty wrapper and the real
    # content sections are its *siblings* (checked, not assumed). Nav/header/
    # footer are already stripped, so body is the safe root.
    root = soup.body
    sections, current = [], {"heading": "Intro", "level": 1, "text": []}
    h2_crumb = ""     # deep sections carry their parent h2 as a breadcrumb —
                      # "Business Results" alone is meaningless to the reranker;
                      # "Case Study: Regional E-Commerce Retailer — Business
                      # Results" is retrievable and citable.

    def flush():
        text = clean_text("\n".join(current["text"]))
        if not is_junk(text, config.MIN_SECTION_CHARS):
            sections.append({"heading": current["heading"], "level": current["level"], "text": text})

    def start_section(heading: str, level: int):
        nonlocal current, h2_crumb
        flush()
        if level >= 3 and h2_crumb:
            heading = f"{h2_crumb} — {heading}"
        current = {"heading": heading, "level": level, "text": []}
        if level <= 2:
            h2_crumb = heading

    for el in root.find_all(HEADING_TAGS + TEXT_TAGS + ["span"]):
        if el.find_parent(TEXT_TAGS + HEADING_TAGS):   # parent already yields it
            continue
        content = el.get_text(" ", strip=True)
        if not content:
            continue

        # This site (Tailwind/Next.js) puts card titles ("E-Commerce",
        # "Financial Services") and result metrics ("70% reduction...") in
        # styled <span>s, not <p>/<h*> — checked in the live HTML. Uppercase-
        # styled spans are card titles; the rest are text.
        classes = " ".join(el.get("class", []))
        if el.name == "span":
            if el.find_parent("a"):
                continue                               # link labels, not content
            if "uppercase" in classes:
                start_section(content, 3)
            else:
                current["text"].append(content)
        elif el.name in HEADING_TAGS:
            if content.lower() in GLUE_HEADINGS:       # keep card structure inline
                current["text"].append(f"{content}:")
            else:
                start_section(content, int(el.name[1]))
        else:
            current["text"].append(content)
    flush()
    return sections


def scrape() -> list[dict]:
    """Fetch every configured page; save a reproducible snapshot to data/."""
    session = requests.Session()
    session.headers.update(HEADERS)
    pages, seen_hashes = [], set()

    for entry in config.PAGES:
        url = entry["url"]
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"  SKIP {url} (HTTP {resp.status_code})")
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        for tag in soup(STRIP_TAGS):
            tag.decompose()
        sections = extract_sections(soup)

        # Content-hash dedup: drop mirror/duplicate pages.
        page_hash = hashlib.md5(" ".join(s["text"] for s in sections).encode()).hexdigest()
        if page_hash in seen_hashes:
            print(f"  SKIP {url} (duplicate content)")
            continue
        seen_hashes.add(page_hash)

        pages.append({
            "url": url,
            "title": title,
            "category": entry["category"],
            "sections": sections,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"  OK   {url} — {len(sections)} sections")
        time.sleep(config.REQUEST_DELAY)

    save_json(pages, config.SCRAPE_FILE)
    total = sum(len(p["sections"]) for p in pages)
    print(f"Saved {len(pages)} pages / {total} sections -> {config.SCRAPE_FILE.name}")
    return pages
