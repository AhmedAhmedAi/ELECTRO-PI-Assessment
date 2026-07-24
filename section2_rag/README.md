# Section 2 — RAG over the grader's own website (LangChain)

The spec offers sample docs or "your own domain docs — state your choice."
I chose **electropi.ai itself** — the company's website and blog, scraped
into a versioned snapshot. Real documents, citations you can verify
because you wrote them, and an honest mix of difficulty: dense long-form
blog articles next to sparse marketing pages. Every answer carries
citations back to the exact page and section; questions the corpus can't
answer are refused by a measured two-stage guardrail, never hallucinated.

- **Framework**: LangChain (embeddings, vector store, chat)
- **Vector store**: Chroma, persisted, cosine space
- **Models**: `gemini-embedding-2` (768-dim MRL, asymmetric task types),
  `gemini-3.5-flash-lite` (answering + reranking), `gemini-3.5-flash` (judge)
- **Spec write-up** (longer documents): [NOTES.md](NOTES.md)

## Pipeline

```
question
  → semantic cache (cosine ≥ 0.92 on query embeddings) ──hit──→ stored answer + citations (~0.5s)
  → dense gate (best cosine < 0.68 → refuse)                      [measured, tuning cell]
  → multi-query expansion (2 Gemini rewrites)
  → hybrid retrieval: Chroma dense + BM25 sparse, per variant
  → Reciprocal Rank Fusion (k=60)
  → Gemini listwise rerank top-10 (all scores < 4/10 → refuse)
  → parent-section expansion (small-to-big)
  → Gemini answer w/ server-cached system prompt → inline [n] citations
```

## Entry point

**[analysis.ipynb](analysis.ipynb)** — the whole pipeline as a stage-by-stage
walkthrough, **pre-executed with real outputs**: every stage's internals
(query variants, BM25-vs-dense-vs-RRF rankings, rerank scores, the guardrail
refusing live, cache timings, judge tables), the 5 example Q&As, and a
"try your own question" cell. The logic lives in `services/` modules; the
notebook drives and inspects them.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'GOOGLE_API_KEY=...' > .env
.venv/bin/jupyter execute analysis.ipynb   # everything end-to-end (~4 min)
# or open it in Jupyter Lab and run cells one by one
```

Readable in 0 minutes — the notebook ships pre-executed. `data/scraped.json`
and `data/chunks.json` are committed, so grading never depends on the live
site (the scrape cell reuses the snapshot unless it's deleted).

## Techniques implemented (not just discussed)

| Technique | Where | Detail |
|---|---|---|
| Structure-preserving scrape | `services/scraper.py` | keeps h1–h4 + styled-span card titles; content-hash dedup |
| Regex chunking, doc-type aware | `utils/text.py` + `ingest.py` | heading-bounded sections, abbreviation-safe sentence regex, junk filters |
| Contextual chunk headers | `ingest.py` | `page title \| section` prefixed into every chunk (helps embedding, BM25, LLM) |
| Asymmetric Gemini embeddings | `ingest.py` / `retriever.py` | `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` task types, 768-dim MRL |
| Hybrid retrieval + RRF | `retriever.py` | dense + BM25 fused with reciprocal-rank fusion |
| Multi-query expansion | `retriever.py` | 2 Gemini rewrites, all variants retrieved and fused |
| LLM listwise reranking | `retriever.py` | RankGPT-style 0–10 scoring in one call |
| Two-stage guardrail (measured) | `retriever.py` + tuning step | dense gate 0.68 + rerank gate 4/10 — both from measured separation |
| Small-to-big retrieval | `retriever.py` | match on ~300-token chunks, answer from full parent sections |
| Greeting instant answers | `services/cache.py` | exact-phrase table (en + ar) answers "hi"/"thanks"/… with zero API calls |
| Semantic cache | `services/cache.py` | paraphrase hits at cosine ≥ 0.92, answers + citations, zero extra API calls |
| Explicit prompt caching | `services/answerer.py` | system prompt (rules + live site map) cached server-side, reused across runs |
| LLM-judge eval (RAGAS-style) | `services/judge.py` | faithfulness / answer relevance / context precision per example |

## How the chunking works — and three bugs this corpus taught me

The chunking is **regex-driven and follows document structure**, because
one splitter never fits all document types: the scraper emits
heading-bounded `(heading, text)` sections → each section is a **parent**
→ `RecursiveCharacterTextSplitter` (regex separators, ~1200 chars, 150
overlap) produces **child** chunks → every chunk gets a contextual header
(`page title | section heading`) plus metadata, then exact-text dedup
(marketing sites repeat blurbs everywhere). Matching happens on ~300-token
chunks; the LLM receives the full parent section — retrieval granularity
and generation context are different needs, so they get different units.

Three real failures shaped it:

1. **`<main>` was a lie.** The site's `<main>` element is a nearly-empty
   wrapper; the content sections are its siblings. Extracting from
   `<main>` silently lost ~85% of the marketing pages. Verify the
   extraction root against the actual DOM — don't trust semantic HTML.
2. **The most valuable facts weren't in `<p>` tags.** Case-study metrics
   ("95% prediction accuracy", "3.2× ROI") and card titles live in styled
   `<span>`s, and card sub-headings ("Challenge"/"Solution") must be glued
   into their card's section — or every case study shatters into anonymous
   fragments no query can match. My first scrape produced exactly that
   failure; the span-aware fix is why the e-commerce example answers
   correctly today.
3. **Deep headings need breadcrumbs.** A section titled "Business Results"
   scored 0/10 at the reranker even though it held the exact answer —
   nothing tied it to its parent "Case Study: Regional E-Commerce
   Retailer". h3+ sections now carry their parent h2 as a breadcrumb,
   which flows into the header, the embedding, BM25, the reranker, and
   the citation. I caught it because the same question flipped between
   answered and refused across runs — rerank-boundary flakiness is
   usually a metadata problem, not a reranker problem.

## Results (from `results/`, reproducible)

**Guardrail separation, measured** (tuning step, 6 on-topic + 6 adversarial
off-topic probes): on-topic best-cosine 0.725–0.857, off-topic 0.533–0.651
— the 0.68 dense gate sits in the empty band between them. Near-topic
questions that slip past it (e.g. "Who is the CEO?" scored 0.75) are caught
by the rerank gate when no passage actually contains the answer. A third
layer sits in the answer prompt itself: a fixed refusal string when the
sources don't contain the answer.

**The 5 example Q&As** (full text: `results/qa_examples.md`):

| Question | Outcome | Judge (faith / rel / prec) |
|---|---|---|
| Founded when + client count | answered, cited → about | 5 / 5 / 5 |
| Why Arabic voice AI is hard | answered, 2 sources | 5 / 5 / 4 |
| E-commerce case study results | answered, cited → blog | 5 / 5 / 5 |
| Chatbot project pricing | answered, cited → blog FAQ | 5 / 5 / 4 |
| **Who is the CEO?** | **refused** (rerank gate, all 0/10) | refusal_correct = true |

The pricing question was *designed* to be the unanswerable trap — then
retrieval found real price ranges in a blog FAQ. It stays as an answered
example; the system knew the corpus better than its author. The CEO
question is verified absent from the whole corpus and is refused at the
rerank stage despite passing the dense gate — exactly the layered behavior
the guardrail is for.

**Latency** (measured on the example run): ~3–5 s full pipeline,
~0.5 s on a semantic-cache hit; 1,138 system-prompt tokens read from
Gemini's server-side cache on every non-cached answer.

## What I'd build next

- **Incremental ingest** — re-embed only changed pages (hash per section);
  today ingest rebuilds all 145 chunks because at this size simplicity wins.
- **GraphRAG / agentic retrieval** — considered, skipped: entity-relation
  graphs pay off on corpora with cross-document structure (contracts,
  wikis); a 12-page site would be pure ceremony.
- **Production path** — these services drop behind a FastAPI
  `routes/rag.py`, Chroma → managed vector DB, semantic cache → Redis with
  TTL + invalidation on re-ingest.

## Honest caveats

- 5 examples show the behaviors, they don't prove statistics; the tune set
  is 12 probes. Both are reproducible by re-running analysis.ipynb, and the
  raw artifacts are in `results/`.
- Judge (`gemini-3.5-flash`) and answerer (`gemini-3.5-flash-lite`) are the
  same model family — same-family judges skew generous. Mitigations: the
  judge sees only sources/question/answer (not the pipeline), scores an
  explicit rubric, and must list unsupported claims.
- Judging a *refusal* needs evidence, not intuition: an early judge version
  ruled the CEO refusal wrong by arguing "a company site surely lists its
  CEO" — speculation, not verification. The refusal judge now receives the
  closest passages retrieval actually found and must answer "do these
  contain the answer?" — with evidence in hand it confirms the refusal.
- The semantic cache returns the stored answer for paraphrases; if the
  index is re-ingested with changed content, the cache must be cleared —
  in production the re-ingest step would invalidate it.
- Marketing copy is buzzword-dense; answers grounded in it inherit that
  tone. The blog articles carry most of the factual weight, and the judged
  examples deliberately lean on them.

## Layout

```
analysis.ipynb       THE entry point — every stage as an executed cell
config.py            every model/threshold/path knob
services/            scraper, ingest, retriever, answerer, cache, judge, pipeline
utils/               text regexes + JSON helpers
data/                scraped snapshot + chunks (committed)
index/               Chroma + caches (rebuilt by the ingest cell, gitignored)
results/             example Q&As, judge scores, threshold measurements
NOTES.md             the spec write-up (chunking/retrieval on longer documents)
```
