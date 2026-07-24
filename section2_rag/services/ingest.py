"""Turn the scraped snapshot into a searchable index.

scraped.json -> regex chunking -> Gemini embeddings -> Chroma (dense).
Sections are the *parents* (what the LLM reads), chunks are the *children*
(what retrieval matches) — small-to-big retrieval. BM25 is built from
chunks.json at query time by the retriever; the corpus is tiny.
"""
import shutil

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from utils.files import load_json, save_json
from utils.text import CHUNK_SEPARATORS


def slugify(url: str) -> str:
    return url.replace("https://electropi.ai", "").strip("/").replace("/", "_") or "home"


def build_chunks(pages: list[dict]) -> tuple[list[dict], dict]:
    """Regex-chunk every section; return (child chunks, parent map)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
    )
    chunks, parents, seen = [], {}, set()
    dropped = 0

    for page in pages:
        for si, sec in enumerate(page["sections"]):
            parent_id = f"{slugify(page['url'])}#{si}"
            parents[parent_id] = {
                "heading": sec["heading"], "text": sec["text"],
                "url": page["url"], "title": page["title"],
            }
            for ci, piece in enumerate(splitter.split_text(sec["text"])):
                piece = piece.strip()
                if len(piece) < config.MIN_CHUNK_CHARS:
                    continue
                if piece in seen:              # dedup: marketing sites repeat blurbs
                    dropped += 1
                    continue
                seen.add(piece)
                # Contextual header: page + section prefixed into the chunk so
                # the embedding, BM25, and the LLM all see where the text lives.
                text = f"{page['title']} | {sec['heading']}\n{piece}"
                chunks.append({
                    "id": f"{parent_id}.{ci}",
                    "text": text,
                    "metadata": {
                        "url": page["url"], "title": page["title"],
                        "category": page["category"], "heading": sec["heading"],
                        "parent_id": parent_id,
                    },
                })

    print(f"{len(chunks)} chunks from {len(parents)} sections ({dropped} duplicates dropped)")
    return chunks, parents


def doc_embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=config.EMBEDDING_DIM,
    )


def ingest():
    config.require_key()
    pages = load_json(config.SCRAPE_FILE)
    chunks, parents = build_chunks(pages)
    save_json({"chunks": chunks, "parents": parents}, config.CHUNKS_FILE)

    if config.CHROMA_DIR.exists():                # rebuild from scratch, no stale docs
        shutil.rmtree(config.CHROMA_DIR)
    # Answers cached against the old index may no longer be grounded.
    config.SEMANTIC_CACHE_FILE.unlink(missing_ok=True)
    store = Chroma(
        collection_name="electropi",
        embedding_function=doc_embedder(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    print(f"Chroma index built: {store._collection.count()} vectors "
          f"({config.EMBEDDING_DIM}-dim, cosine) -> {config.CHROMA_DIR}")
