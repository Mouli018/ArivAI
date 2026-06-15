"""
document_utils.py – Document loading, chunking, and retrieval.

Vector store: Qdrant Cloud (persistent, cloud-hosted)
  - Collections are created once per document and persist across restarts.
  - Hybrid search: Qdrant dense vectors + BM25 sparse (in-memory).
  - Cross-encoder re-ranking for final precision boost.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import (
    CHUNK_SIZES,
    CHUNK_SIZE,
    CHUNK_OVERLAP_RATIO,
    CHUNK_OVERLAP,
    DOCS_PATH,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    SUPPORTED_EXTENSIONS,
    QDRANT_URL,
    QDRANT_API_KEY,
)

# ─── Embedding model (singleton) ──────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the all-MiniLM-L6-v2 model (cached, runs fully locally)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


# ─── Cross-encoder re-ranker (singleton) ──────────────────────────────────────

_reranker = None

def _get_reranker():
    """Lazy-load the cross-encoder re-ranker (~80 MB, downloaded once)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            print(f"Warning: Could not load cross-encoder reranker: {exc}")
            _reranker = False
    return _reranker


# ─── Qdrant client (singleton) ────────────────────────────────────────────────

_qdrant_client: Optional[QdrantClient] = None

def _get_qdrant_client() -> QdrantClient:
    """Return a shared QdrantClient connected to Qdrant Cloud."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in .env to use Qdrant Cloud."
            )
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
    return _qdrant_client


# ─── VectorstoreBundle ────────────────────────────────────────────────────────

@dataclass
class VectorstoreBundle:
    """Holds the Qdrant vector store, BM25 index, and raw chunks for a document."""
    qdrant: QdrantVectorStore
    bm25: BM25Okapi
    chunks: List[Document]
    filename: str = ""
    collection_name: str = ""


# ─── Collection name helpers ──────────────────────────────────────────────────

def _collection_name(filename: str) -> str:
    """
    Convert a filename to a valid Qdrant collection name.
    Qdrant allows: letters, digits, underscores, hyphens. Max 255 chars.
    """
    name = os.path.splitext(filename)[0]            # strip extension
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)    # replace invalid chars
    name = re.sub(r"_+", "_", name).strip("_-")     # collapse underscores
    return (name or "doc")[:63]                      # max 63 chars


# ─── Document loading ─────────────────────────────────────────────────────────

def _get_loader(path: str, filename: str):
    """Return the appropriate LangChain loader for the given file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path)
    elif ext == ".txt":
        return TextLoader(path, encoding="utf-8")
    elif ext == ".docx":
        return Docx2txtLoader(path)
    return None


def load_documents_grouped(docs_path: str = DOCS_PATH) -> Dict[str, List[Document]]:
    """
    Load all supported documents from *docs_path*, grouped by filename.
    Each page Document gets enriched metadata: source, filetype, page.
    """
    if not os.path.exists(docs_path):
        os.makedirs(docs_path, exist_ok=True)
        return {}

    grouped: Dict[str, List[Document]] = defaultdict(list)

    for filename in sorted(os.listdir(docs_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        full_path = os.path.join(docs_path, filename)
        if not os.path.isfile(full_path):
            continue
        loader = _get_loader(full_path, filename)
        if loader is None:
            continue
        try:
            pages = loader.load()
            for i, page in enumerate(pages):
                page.metadata.setdefault("source", filename)
                page.metadata["filename"] = filename
                page.metadata["filetype"] = ext
                page.metadata.setdefault("page", i)
            grouped[filename].extend(pages)
        except Exception as exc:
            print(f"Warning: Could not load '{filename}': {exc}")

    return dict(grouped)


def load_documents(docs_path: str = DOCS_PATH) -> List[Document]:
    """Return all documents as a flat list (legacy helper)."""
    grouped = load_documents_grouped(docs_path)
    return [doc for docs in grouped.values() for doc in docs]


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _make_splitter(filetype: str) -> RecursiveCharacterTextSplitter:
    size = CHUNK_SIZES.get(filetype, CHUNK_SIZE)
    overlap = max(50, int(size * CHUNK_OVERLAP_RATIO))
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, length_function=len
    )


def _chunk_documents(documents: List[Document]) -> List[Document]:
    """Adaptive chunking per filetype, adds chunk_index to metadata."""
    by_type: Dict[str, List[Document]] = defaultdict(list)
    for doc in documents:
        ft = doc.metadata.get("filetype", ".txt")
        by_type[ft].append(doc)

    all_chunks: List[Document] = []
    for ft, docs in by_type.items():
        chunks = _make_splitter(ft).split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        all_chunks.extend(chunks)
    return all_chunks


# ─── BM25 helpers ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return text.lower().split()

def _build_bm25(chunks: List[Document]) -> BM25Okapi:
    return BM25Okapi([_tokenize(c.page_content) for c in chunks])


# ─── Qdrant chunk fetching (for BM25 rebuild on load) ─────────────────────────

def _fetch_chunks_from_qdrant(client: QdrantClient, col_name: str) -> List[Document]:
    """Scroll all points from a Qdrant collection to rebuild BM25 index."""
    chunks: List[Document] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=col_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            if content:
                chunks.append(Document(page_content=content, metadata=metadata))
        if next_offset is None:
            break
        offset = next_offset
    return chunks


# ─── Vectorstore creation ─────────────────────────────────────────────────────

def create_vectorstore(documents: List[Document], filename: str = "") -> VectorstoreBundle:
    """
    Chunk documents, embed them, and upsert into Qdrant Cloud.

    If the collection already exists (from a previous run), it is loaded
    directly — no re-embedding needed (persistent benefit of Qdrant Cloud).
    """
    col_name = _collection_name(filename or "docs")
    client = _get_qdrant_client()
    embeddings = get_embeddings()

    if client.collection_exists(col_name):
        # ── Load existing collection (fast path) ──
        print(f"Loading existing Qdrant collection: '{col_name}'")
        qdrant_vs = QdrantVectorStore(
            client=client,
            collection_name=col_name,
            embedding=embeddings,
        )
        chunks = _fetch_chunks_from_qdrant(client, col_name)
    else:
        # ── Create new collection ──
        chunks = _chunk_documents(documents)
        if not chunks:
            raise ValueError("No extractable text found in the provided documents.")
        print(f"Creating Qdrant collection '{col_name}' with {len(chunks)} chunks...")
        qdrant_vs = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=col_name,
        )

    bm25_index = _build_bm25(chunks)
    return VectorstoreBundle(
        qdrant=qdrant_vs,
        bm25=bm25_index,
        chunks=chunks,
        filename=filename,
        collection_name=col_name,
    )


def build_all_vectorstores(docs_path: str = DOCS_PATH) -> Dict[str, VectorstoreBundle]:
    """Build/load one VectorstoreBundle per document in *docs_path*."""
    grouped = load_documents_grouped(docs_path)
    bundles: Dict[str, VectorstoreBundle] = {}
    for filename, docs in grouped.items():
        try:
            bundles[filename] = create_vectorstore(docs, filename=filename)
        except Exception as exc:
            print(f"Warning: Could not build vectorstore for '{filename}': {exc}")
    return bundles


# ─── Hybrid retrieval ─────────────────────────────────────────────────────────

def hybrid_search(
    bundle: VectorstoreBundle,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Document]:
    """
    Combine Qdrant dense (cosine) search with BM25 sparse search.
    Scores are normalised to [0,1] and averaged for merged ranking.
    """
    # ── Dense: Qdrant similarity search ──
    # Qdrant cosine scores: higher = more similar (already in [0,1] range)
    dense_results: List[Tuple[Document, float]] = \
        bundle.qdrant.similarity_search_with_score(query, k=top_k)

    max_dense = max((s for _, s in dense_results), default=1.0) or 1.0
    dense_map: Dict[str, Tuple[Document, float]] = {
        doc.page_content[:120]: (doc, score / max_dense)
        for doc, score in dense_results
    }

    # ── Sparse: BM25 keyword search ──
    bm25_scores = bundle.bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    max_bm25 = bm25_scores[top_idx[0]] if top_idx else 1.0

    sparse_map: Dict[str, Tuple[Document, float]] = {}
    for idx in top_idx:
        if idx < len(bundle.chunks):
            doc = bundle.chunks[idx]
            score = bm25_scores[idx] / (max_bm25 or 1.0)
            sparse_map[doc.page_content[:120]] = (doc, score)

    # ── Merge: average scores for docs found by both ──
    merged: Dict[str, Tuple[Document, float]] = {}
    for key, (doc, score) in dense_map.items():
        merged[key] = (doc, score)
    for key, (doc, score) in sparse_map.items():
        if key in merged:
            existing_doc, existing_score = merged[key]
            merged[key] = (existing_doc, (existing_score + score) / 2)
        else:
            merged[key] = (doc, score)

    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ─── Cross-encoder re-ranking ─────────────────────────────────────────────────

def rerank_chunks(
    query: str,
    chunks: List[Document],
    top_n: int = RERANK_TOP_N,
) -> List[Document]:
    """Re-rank candidates with a cross-encoder. Falls back to top-N if unavailable."""
    if not chunks:
        return []
    reranker = _get_reranker()
    if not reranker:
        return chunks[:top_n]
    try:
        scores = reranker.predict([(query, c.page_content) for c in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_n]:
            chunk.metadata["rerank_score"] = round(float(score), 4)
            result.append(chunk)
        return result
    except Exception as exc:
        print(f"Warning: Re-ranking failed: {exc}")
        return chunks[:top_n]


# ─── Citation builder ─────────────────────────────────────────────────────────

def build_citation_text(chunks: List[Document]) -> List[dict]:
    """Build a list of citation dicts from chunk metadata."""
    seen = set()
    citations = []
    for chunk in chunks:
        src = chunk.metadata.get("filename") or chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        score = chunk.metadata.get("rerank_score", None)
        key = f"{src}|{page}"
        if key not in seen:
            seen.add(key)
            citations.append({"source": src, "page": page, "rerank_score": score})
    return citations
