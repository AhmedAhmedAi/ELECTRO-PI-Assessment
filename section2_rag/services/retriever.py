"""Retrieval pipeline: multi-query -> hybrid (dense + BM25) -> RRF ->
Gemini listwise rerank -> guardrail -> parent-section expansion.

Two-stage guardrail (both thresholds measured, see the tuning step in analysis.ipynb):
  1. dense gate  — best cosine of the original question; cheap early exit
  2. rerank gate — the reranker must give at least one chunk a real
     relevance score; catches keyword-lookalike false positives
"""
import json

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi

import config
from utils.files import load_json
from utils.text import tokenize

EXPAND_PROMPT = """Rewrite the user's question as {n} alternative search queries
that use different wording (synonyms, related terms) but keep the exact meaning.
Return JSON: {{"queries": ["...", "..."]}}

Question: {question}"""

RERANK_PROMPT = """Score how relevant each passage is for answering the question.
10 = directly answers it, 5 = related background, 0 = unrelated.
Return JSON: {{"scores": [{{"id": <passage id>, "relevance": <0-10>}}, ...]}} — one entry per passage.

Question: {question}

Passages:
{passages}"""


def response_text(message) -> str:
    """Gemini 3.5 returns content as a list of blocks — flatten to text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def query_embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=config.EMBEDDING_DIM,
    )


class Retriever:
    """Holds the loaded indexes (Chroma, BM25, parent map)."""

    def __init__(self):
        data = load_json(config.CHUNKS_FILE)
        self.chunks = {c["id"]: c for c in data["chunks"]}
        self.parents = data["parents"]
        self.embedder = query_embedder()
        self.store = Chroma(
            collection_name="electropi",
            embedding_function=self.embedder,
            persist_directory=str(config.CHROMA_DIR),
        )
        self.bm25_ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self.bm25_ids])
        self.llm = ChatGoogleGenerativeAI(
            model=config.RERANK_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            response_mime_type="application/json",
        )

    # --- stages -----------------------------------------------------------

    def expand(self, question: str) -> list[str]:
        """The original question plus N reworded variants."""
        try:
            raw = response_text(self.llm.invoke(
                EXPAND_PROMPT.format(n=config.N_QUERY_VARIANTS, question=question)
            ))
            variants = json.loads(raw)["queries"][: config.N_QUERY_VARIANTS]
        except (json.JSONDecodeError, KeyError, TypeError):
            variants = []
        return [question] + [v for v in variants if isinstance(v, str) and v.strip()]

    def dense_search(self, query_vector: list[float]) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity), best first."""
        # Despite its name this method returns cosine *distance* (checked the
        # source: "Lower score represents more similarity") — convert here.
        hits = self.store.similarity_search_by_vector_with_relevance_scores(
            query_vector, k=config.DENSE_TOP_K
        )
        return [(doc.id, 1.0 - dist) for doc, dist in hits]

    def bm25_search(self, query: str) -> list[str]:
        """chunk_ids ranked by BM25, zero-score tail cut."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda p: p[1], reverse=True)
        return [cid for cid, s in ranked[: config.BM25_TOP_K] if s > 0]

    @staticmethod
    def rrf(rankings: list[list[str]]) -> list[str]:
        """Reciprocal-rank fusion across all ranked lists."""
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def rerank(self, question: str, candidate_ids: list[str]) -> list[tuple[str, int]]:
        """One listwise Gemini call -> [(chunk_id, relevance 0-10)], best first."""
        passages = "\n\n".join(
            f"[{i}] {self.chunks[cid]['text'][:800]}"
            for i, cid in enumerate(candidate_ids)
        )
        try:
            raw = response_text(self.llm.invoke(
                RERANK_PROMPT.format(question=question, passages=passages)
            ))
            scores = {int(s["id"]): int(s["relevance"]) for s in json.loads(raw)["scores"]}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Rerank failure must not kill the answer: keep RRF order, neutral score.
            scores = {i: 5 for i in range(len(candidate_ids))}
        ranked = sorted(scores.items(), key=lambda p: p[1], reverse=True)
        return [(candidate_ids[i], rel) for i, rel in ranked if i < len(candidate_ids)]

    # --- the pipeline -----------------------------------------------------

    def retrieve(self, question: str, query_vector: list[float]) -> dict:
        """Full pipeline. Returns status, parent contexts, and diagnostics."""
        diag = {}

        dense_main = self.dense_search(query_vector)
        best_cosine = dense_main[0][1] if dense_main else 0.0
        diag["best_cosine"] = round(best_cosine, 3)
        if best_cosine < config.DENSE_GATE:
            diag["gate"] = f"dense gate ({best_cosine:.3f} < {config.DENSE_GATE})"
            return {"status": "no_context", "contexts": [], "diagnostics": diag}

        variants = self.expand(question)
        diag["query_variants"] = variants
        rankings = [[cid for cid, _ in dense_main], self.bm25_search(question)]
        for v in variants[1:]:
            rankings.append([d.id for d in self.store.similarity_search(v, k=config.DENSE_TOP_K)])
            rankings.append(self.bm25_search(v))

        candidates = self.rrf(rankings)[: config.RERANK_CANDIDATES]
        reranked = self.rerank(question, candidates)
        diag["rerank"] = [
            {"id": cid, "relevance": rel,
             "heading": self.chunks[cid]["metadata"]["heading"],
             "preview": self.chunks[cid]["text"][:200]}
            for cid, rel in reranked
        ]

        kept = [(cid, rel) for cid, rel in reranked if rel >= config.RERANK_GATE]
        if not kept:
            diag["gate"] = f"rerank gate (best {reranked[0][1] if reranked else 0} < {config.RERANK_GATE})"
            return {"status": "no_context", "contexts": [], "diagnostics": diag}

        # Small-to-big: hand the LLM whole parent sections, deduped, best first.
        contexts, seen_parents = [], set()
        for cid, rel in kept[: config.RERANK_KEEP]:
            pid = self.chunks[cid]["metadata"]["parent_id"]
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            parent = self.parents[pid]
            contexts.append({
                "n": len(contexts) + 1,
                "heading": parent["heading"], "title": parent["title"],
                "url": parent["url"], "text": parent["text"], "relevance": rel,
            })
        return {"status": "ok", "contexts": contexts, "diagnostics": diag}
