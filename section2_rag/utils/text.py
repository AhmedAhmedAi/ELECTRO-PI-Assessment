"""Text cleaning and the regexes that drive chunking.

Patterns are written for this corpus — a marketing site plus long-form
blog articles. See NOTES.md: chunking strategy follows document type.
"""
import re

# Boilerplate lines a marketing site repeats on every page. If a chunk is
# mostly these, it carries no information worth indexing.
JUNK_PATTERNS = [
    r"^(Get Started|Learn More|Read More|Contact Us|Book a|Schedule a)\b",
    r"^(Home|About|Services|Solutions|Case Studies|Blog|Contact)$",
    r"^© ?\d{4}",                      # copyright footer
    r"^(Privacy Policy|Terms of Service)$",
    r"^\d+ min read$",
]
JUNK_REGEX = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

# Sentence-aware separators for RecursiveCharacterTextSplitter
# (is_separator_regex=True). Splitting AFTER end punctuation via lookbehind
# — a bare "." separator would cut inside "3.2", "et al.", "99.9%".
CHUNK_SEPARATORS = [r"\n\n", r"(?<=[.!?])\s+", r"\n", r" ", r""]


def clean_text(text: str) -> str:
    """Collapse whitespace and drop obvious UI junk lines."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not JUNK_REGEX.match(line):
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_junk(text: str, min_chars: int) -> bool:
    """True when a block is too short or mostly call-to-action noise."""
    if len(text) < min_chars:
        return True
    words = text.split()
    cta_words = {"get", "started", "learn", "more", "contact", "book", "demo"}
    cta_count = sum(1 for w in words[:20] if w.lower() in cta_words)
    return len(words) > 0 and cta_count / min(len(words), 20) > 0.4


def tokenize(text: str) -> list[str]:
    """Word tokenizer for BM25 (keeps Arabic character ranges too)."""
    return re.findall(r"\b\w+\b|[؀-ۿ]+", text.lower())
