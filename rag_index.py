from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import List, Dict, Any

import numpy as np
import requests


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vectors / norms


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    return _normalize_vectors(vectors)


@dataclass
class RAGSettings:
    base_dir: Path
    doc_dir: Path
    data_dir: Path
    embed_backend: str
    embed_model: str
    embed_device: str
    embed_batch_size: int
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_faiss: bool
    max_context_chars: int
    timeout: int
    rerank_enabled: bool
    rerank_model: str
    rerank_device: str
    rerank_batch_size: int
    rerank_top_k: int

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "RAGSettings":
        base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent

        doc_dir_env = os.getenv("RAG_DOC_DIR")
        data_dir_env = os.getenv("RAG_DATA_DIR")
        doc_dir = Path(doc_dir_env) if doc_dir_env else base_dir / "document"
        data_dir = Path(data_dir_env) if data_dir_env else base_dir / "rag_data"
        if not doc_dir.is_absolute():
            doc_dir = (base_dir / doc_dir).resolve()
        if not data_dir.is_absolute():
            data_dir = (base_dir / data_dir).resolve()

        embed_backend = os.getenv("EMBED_BACKEND", "hf").lower()
        default_model = "huyydangg/DEk21_hcmute_embedding" if embed_backend == "hf" else "mxbai-embed-large"

        use_faiss = _to_bool(os.getenv("RAG_USE_FAISS", "1"))

        return cls(
            base_dir=base_dir,
            doc_dir=doc_dir,
            data_dir=data_dir,
            embed_backend=embed_backend,
            embed_model=os.getenv("EMBED_MODEL", default_model),
            embed_device=os.getenv("EMBED_DEVICE", "cpu"),
            embed_batch_size=int(os.getenv("RAG_EMBED_BATCH_SIZE", "16")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            use_faiss=use_faiss,
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
            timeout=int(os.getenv("RAG_HTTP_TIMEOUT", "60")),
            rerank_enabled=_to_bool(os.getenv("RAG_RERANK_ENABLED", "1")),
            rerank_model=os.getenv("RAG_RERANK_MODEL", "namdp-ptit/ViRanker"),
            rerank_device=os.getenv("RAG_RERANK_DEVICE", "cpu"),
            rerank_batch_size=int(os.getenv("RAG_RERANK_BATCH_SIZE", "8")),
            rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "8")),
        )

_HF_STATE: dict[str, object] = {}
_RERANK_STATE: dict[str, object] = {}


def _get_hf_model(settings: RAGSettings):
    cached = _HF_STATE.get("model")
    if cached and _HF_STATE.get("name") == settings.embed_model and _HF_STATE.get("device") == settings.embed_device:
        return cached

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - only triggers on broken env
        hint = (
            "Failed to import sentence-transformers. If you see a torchvision::nms error, "
            "set TRANSFORMERS_NO_TORCHVISION=1 or uninstall/match torchvision with torch."
        )
        raise RuntimeError(hint) from exc

    model = SentenceTransformer(settings.embed_model, device=settings.embed_device)
    _HF_STATE["model"] = model
    _HF_STATE["name"] = settings.embed_model
    _HF_STATE["device"] = settings.embed_device
    return model


def _embed_texts_hf(texts: List[str], settings: RAGSettings) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_hf_model(settings)
    embeddings = model.encode(
        texts,
        batch_size=settings.embed_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _embed_texts_ollama(texts: List[str], settings: RAGSettings) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = []
    for text in texts:
        payload = {"model": settings.embed_model, "prompt": text}
        response = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json=payload,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama did not return an embedding vector.")
        vectors.append(np.array(embedding, dtype=np.float32))
    return _normalize_vectors(np.vstack(vectors)).astype(np.float32)


def embed_texts(texts: List[str], settings: RAGSettings) -> np.ndarray:
    backend = settings.embed_backend.lower()
    if backend == "hf":
        return _embed_texts_hf(texts, settings)
    if backend == "ollama":
        return _embed_texts_ollama(texts, settings)
    raise ValueError(f"Unknown EMBED_BACKEND: {settings.embed_backend}")


def embed_text(text: str, settings: RAGSettings) -> np.ndarray:
    vectors = embed_texts([text], settings)
    if vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return vectors[0]


def _get_reranker(settings: RAGSettings):
    cached = _RERANK_STATE.get("model")
    if (
        cached
        and _RERANK_STATE.get("name") == settings.rerank_model
        and _RERANK_STATE.get("device") == settings.rerank_device
    ):
        return cached

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover
        hint = (
            "Failed to import sentence-transformers CrossEncoder. "
            "Set TRANSFORMERS_NO_TORCHVISION=1 or fix torch/torchvision."
        )
        raise RuntimeError(hint) from exc

    model = CrossEncoder(settings.rerank_model, device=settings.rerank_device)
    _RERANK_STATE["model"] = model
    _RERANK_STATE["name"] = settings.rerank_model
    _RERANK_STATE["device"] = settings.rerank_device
    return model


def rerank_results(query: str, results: List[Dict[str, Any]], settings: RAGSettings) -> List[Dict[str, Any]]:
    if not results or not settings.rerank_enabled:
        return results
    model = _get_reranker(settings)
    pairs = [[query, item.get("text", "")] for item in results]
    scores = model.predict(pairs, batch_size=settings.rerank_batch_size)
    for item, score in zip(results, scores):
        item["rerank_score"] = float(score)
    return sorted(results, key=lambda item: item.get("rerank_score", 0.0), reverse=True)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix is None or matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = matrix_norm * query_norm
    denom[denom == 0] = 1e-8
    return (matrix @ query_vec) / denom


class RAGIndex:
    def __init__(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray | None,
        settings: RAGSettings,
        faiss_index: object | None = None,
    ):
        self.items = items
        self.embeddings = embeddings
        self.settings = settings
        self.faiss_index = faiss_index

    @property
    def available(self) -> bool:
        has_embeddings = self.embeddings is not None and self.embeddings.size > 0
        return bool(self.items) and (has_embeddings or self.faiss_index is not None)

    @classmethod
    def load(cls, settings: RAGSettings) -> "RAGIndex":
        index_path = settings.data_dir / "index.json"
        embeddings_path = settings.data_dir / "embeddings.npy"
        faiss_index = None
        if not index_path.exists() or not embeddings_path.exists():
            return cls([], None, settings, faiss_index)
        items = json.loads(index_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        if settings.use_faiss:
            faiss_path = settings.data_dir / "index.faiss"
            if faiss_path.exists():
                try:
                    import faiss  # type: ignore

                    faiss_index = faiss.read_index(str(faiss_path))
                except Exception:
                    faiss_index = None
        return cls(items, embeddings, settings, faiss_index)

    def query(self, text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        k = min(top_k or self.settings.top_k, len(self.items))
        if k <= 0:
            return []

        candidate_k = k
        if self.settings.rerank_enabled:
            candidate_k = max(candidate_k, self.settings.rerank_top_k)

        results: List[Dict[str, Any]] = []
        if self.faiss_index is not None:
            query_vec = embed_text(text, self.settings).astype(np.float32)
            query_vec = _normalize_vectors(query_vec)
            query_vec = query_vec.reshape(1, -1)
            distances, indices = self.faiss_index.search(query_vec, min(candidate_k, len(self.items)))
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                item = dict(self.items[int(idx)])
                item["score"] = float(score)
                results.append(item)
        else:
            query_vec = embed_text(text, self.settings)
            scores = cosine_similarity(query_vec, self.embeddings)
            if scores.size == 0:
                return []
            top_indices = np.argsort(-scores)[:candidate_k]
            for idx in top_indices:
                item = dict(self.items[int(idx)])
                item["score"] = float(scores[int(idx)])
                results.append(item)

        results = rerank_results(text, results, self.settings)
        return results[:k]


def build_context(results: List[Dict[str, Any]], settings: RAGSettings) -> str:
    if not results:
        return ""
    chunks = []
    total = 0
    for rank, item in enumerate(results, start=1):
        source = item.get("source_path", "unknown")
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        if page_start and page_end and page_start != page_end:
            page_label = f"pages {page_start}-{page_end}"
        elif page_start:
            page_label = f"page {page_start}"
        else:
            page_label = "page ?"

        snippet = (item.get("text") or "").strip()
        block = f"[{rank}] {snippet}\nSOURCE: {source} ({page_label})"
        if total + len(block) > settings.max_context_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks)
