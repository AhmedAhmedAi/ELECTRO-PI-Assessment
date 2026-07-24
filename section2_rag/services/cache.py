"""Instant-answer layer: greetings + semantic cache.

Greetings ("hi", "thanks", …) are the most frequent inputs a public
assistant sees — they get a canned reply from a pattern table, before any
API call is spent. Real questions go to the semantic cache: paraphrases of
an already-answered question return the stored answer (with its citations)
for the cost of one embedding call — and that embedding is the same one
retrieval needs anyway, so a lookup adds zero extra API calls.
"""
import re
from datetime import datetime, timezone

import numpy as np

import config
from utils.files import load_json, save_json

# The site (and its audience) is English + Egypt-based — cover both.
GREETING_RESPONSES = [
    ({"hi", "hello", "hey", "good morning", "good evening", "salam",
      "السلام عليكم", "مرحبا", "اهلا", "ازيك"},
     "Hello! I'm the ElectroPi knowledge assistant. Ask me anything about "
     "ElectroPi — services, case studies, the team, or its AI engineering "
     "blog (OCR, chatbots, Arabic voice AI, call-center AI)."),
    ({"thanks", "thank you", "great, thanks", "شكرا", "تسلم"},
     "You're welcome! Anything else you'd like to know about ElectroPi?"),
    ({"bye", "goodbye", "see you", "باي", "مع السلامة"},
     "Goodbye! Come back any time you have questions about ElectroPi."),
]


def greeting_answer(text: str) -> str | None:
    """Canned reply for pure greetings — zero API calls. Exact phrase match
    only (after trimming punctuation), so real questions never match."""
    normalized = re.sub(r"[!.،؟?\s]+$", "", text.strip().lower())
    for phrases, reply in GREETING_RESPONSES:
        if normalized in phrases:
            return reply
    return None


class SemanticCache:
    def __init__(self):
        self.entries: list[dict] = []
        self.vectors: np.ndarray | None = None
        if config.SEMANTIC_CACHE_FILE.exists():
            self.entries = load_json(config.SEMANTIC_CACHE_FILE)
            if self.entries:
                self.vectors = np.array([e["embedding"] for e in self.entries])

    def lookup(self, query_vector: list[float]) -> dict | None:
        """Best stored answer with cosine >= threshold, else None."""
        if self.vectors is None:
            return None
        q = np.array(query_vector)
        sims = self.vectors @ q / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(q))
        best = int(np.argmax(sims))
        if sims[best] >= config.SEMANTIC_CACHE_THRESHOLD:
            hit = dict(self.entries[best])
            hit["similarity"] = round(float(sims[best]), 3)
            hit.pop("embedding", None)
            return hit
        return None

    def store(self, question: str, query_vector: list[float], answer: str, citations: list[dict]):
        self.entries.append({
            "question": question, "embedding": list(query_vector),
            "answer": answer, "citations": citations,
            "stored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self.vectors = np.array([e["embedding"] for e in self.entries])
        save_json(self.entries, config.SEMANTIC_CACHE_FILE)
