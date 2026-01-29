from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import List, Dict, Any

import numpy as np
import requests


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
    max_context_chars: int
    timeout: int

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
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000")),
            timeout=int(os.getenv("RAG_HTTP_TIMEOUT", "60")),
        )

_HF_STATE: dict[str, object] = {}


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
    return np.vstack(vectors)


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
    def __init__(self, items: List[Dict[str, Any]], embeddings: np.ndarray | None, settings: RAGSettings):
        self.items = items
        self.embeddings = embeddings
        self.settings = settings

    @property
    def available(self) -> bool:
        return bool(self.items) and self.embeddings is not None and self.embeddings.size > 0

    @classmethod
    def load(cls, settings: RAGSettings) -> "RAGIndex":
        index_path = settings.data_dir / "index.json"
        embeddings_path = settings.data_dir / "embeddings.npy"
        if not index_path.exists() or not embeddings_path.exists():
            return cls([], None, settings)
        items = json.loads(index_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        return cls(items, embeddings, settings)

    def query(self, text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        query_vec = embed_text(text, self.settings)
        scores = cosine_similarity(query_vec, self.embeddings)
        if scores.size == 0:
            return []
        k = min(top_k or self.settings.top_k, len(self.items))
        if k <= 0:
            return []
        top_indices = np.argsort(-scores)[:k]
        results = []
        for idx in top_indices:
            item = dict(self.items[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results


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
