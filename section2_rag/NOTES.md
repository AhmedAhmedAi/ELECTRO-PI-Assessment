# Write-up: what I would change about chunking or retrieval if answer quality on longer documents were poor

Actually, I already applied most of what I would normally reach for when
answer quality drops — built into this pipeline, not left as ideas: regex
chunking that follows document structure (heading-bounded sections,
sentence-safe lookbehind splitting so a plain "." never cuts inside "3.2"
or "99.9%", junk filtering), breadcrumbs on deep sections (a section
called "Business Results" means nothing alone — carrying its parent title
took it from a 0/10 rerank score to a correct cited answer, and on long
documents this is what breaks first), hybrid search (BM25 + dense with
RRF), multi-query expansion, LLM re-ranking, and small-to-big retrieval
(match on small chunks, hand the model the full parent section).

So if quality on longer documents were still poor, my first move is not
another technique — it is measuring where the failure is: a small labeled
question-to-passage set to check retrieval recall@k. If the right passage
never reaches the top-k it is a chunking/retrieval problem; if it is
retrieved but the answer is still wrong it is a context problem. Changing
chunk sizes blindly is the classic mistake.

Beyond that, the techniques I always love to use as an option in RAG in general depend on the task:

- Structure-aware chunking with breadcrumbs — applied here.
- Contextual chunk headers — applied here (page title + section prefix).
- Multi-query expansion — applied here.
- Multi-level small-to-big — two levels here; on long documents I extend
  to chunk → section → chapter rather than enlarging chunks, because
  bigger chunks dilute the embedding.
- Metadata pre-filtering — shrink the search space by chapter or document
  type before similarity search runs.
- GraphRAG — for corpora with real cross-document structure (contracts,
  wikis); pure ceremony on a small single-site corpus, so I skipped it.

For this app specifically I would add: LLM-generated contextual chunk
summaries (a couple of lines on where each chunk sits, prepended before
embedding — my current prefix is the cheap version), a labeled recall@k
eval set so retrieval regressions are caught by numbers not by eye, and a
cross-encoder reranker once traffic grows — cheaper per query than an LLM
reranker and trainable on my domain.
