from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from rag_index import RAGSettings, embed_texts, normalize_embeddings


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return " ".join(lines)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return []
    overlap = max(0, min(overlap, max(chunk_size - 1, 0)))
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def file_fingerprint(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    stat = path.stat()
    return {"sha256": hasher.hexdigest(), "size": stat.st_size, "mtime": int(stat.st_mtime)}


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"_meta": {}, "files": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "files" not in data:
        return {"_meta": {}, "files": data}
    return data


def save_state(path: Path, meta: dict, files: dict) -> None:
    payload = {"_meta": meta, "files": files}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_pdf_chunks(path: Path, settings: RAGSettings) -> list[dict]:
    reader = PdfReader(str(path))
    rel_path = str(path.relative_to(settings.base_dir)).replace("\\", "/")
    items = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            items.append(
                {
                    "source_path": rel_path,
                    "page_start": page_number,
                    "page_end": page_number,
                    "text": chunk,
                }
            )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental PDF ingest for local RAG.")
    parser.add_argument(
        "--reindex-all",
        action="store_true",
        help="Ignore cache and rebuild the full index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = RAGSettings.from_env()

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    state_path = settings.data_dir / "state.json"
    index_path = settings.data_dir / "index.json"
    embeddings_path = settings.data_dir / "embeddings.npy"

    state = load_state(state_path)
    state_meta = state.get("_meta", {})
    state_files = state.get("files", {})

    meta_changed = (
        state_meta.get("embed_backend") != settings.embed_backend
        or state_meta.get("embed_model") != settings.embed_model
        or state_meta.get("chunk_size") != settings.chunk_size
        or state_meta.get("chunk_overlap") != settings.chunk_overlap
    )
    force_reindex = args.reindex_all or meta_changed

    pdf_paths = sorted(settings.doc_dir.rglob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in {settings.doc_dir}")
        return

    current_files = {}
    changed_paths: list[Path] = []
    unchanged_paths = set()

    for path in pdf_paths:
        rel_path = str(path.relative_to(settings.base_dir)).replace("\\", "/")
        fingerprint = file_fingerprint(path)
        current_files[rel_path] = fingerprint
        if force_reindex:
            changed_paths.append(path)
        else:
            if state_files.get(rel_path) == fingerprint:
                unchanged_paths.add(rel_path)
            else:
                changed_paths.append(path)

    if force_reindex:
        unchanged_paths.clear()

    existing_items: list[dict] = []
    existing_embeddings: np.ndarray | None = None
    if index_path.exists() and embeddings_path.exists() and not force_reindex:
        existing_items = json.loads(index_path.read_text(encoding="utf-8"))
        existing_embeddings = np.load(embeddings_path)
        if len(existing_items) != existing_embeddings.shape[0]:
            print("Index size mismatch; rebuilding all documents.")
            existing_items = []
            existing_embeddings = None
            unchanged_paths.clear()
            changed_paths = list(pdf_paths)

    kept_items: list[dict] = []
    kept_embeddings: np.ndarray | None = None
    if existing_items and existing_embeddings is not None and unchanged_paths:
        keep_indices = [
            idx
            for idx, item in enumerate(existing_items)
            if item.get("source_path") in unchanged_paths
        ]
        kept_items = [existing_items[idx] for idx in keep_indices]
        if keep_indices:
            kept_embeddings = existing_embeddings[keep_indices]
        else:
            kept_embeddings = np.zeros((0, existing_embeddings.shape[1]), dtype=np.float32)

    new_items: list[dict] = []
    for path in changed_paths:
        print(f"Indexing {path} ...")
        new_items.extend(extract_pdf_chunks(path, settings))

    new_embeddings = embed_texts([item["text"] for item in new_items], settings)

    all_items = kept_items + new_items
    if kept_embeddings is None or kept_embeddings.size == 0:
        all_embeddings = new_embeddings
    elif new_embeddings.size == 0:
        all_embeddings = kept_embeddings
    else:
        if kept_embeddings.shape[1] != new_embeddings.shape[1]:
            raise RuntimeError("Embedding dimensions changed; re-run with --reindex-all.")
        all_embeddings = np.vstack([kept_embeddings, new_embeddings])

    all_embeddings = normalize_embeddings(all_embeddings).astype(np.float32)

    for idx, item in enumerate(all_items):
        item["id"] = idx

    index_path.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embeddings_path, all_embeddings)

    if settings.use_faiss and all_embeddings.size > 0:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(all_embeddings.shape[1])
            index.add(all_embeddings)
            faiss.write_index(index, str(settings.data_dir / "index.faiss"))
        except Exception as exc:
            print(f"FAISS not available, skipped building index.faiss: {exc}")

    meta = {
        "embed_backend": settings.embed_backend,
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "doc_dir": str(settings.doc_dir),
        "use_faiss": settings.use_faiss,
    }
    save_state(state_path, meta, current_files)

    print(f"Indexed {len(all_items)} chunks from {len(pdf_paths)} PDFs.")
    if meta_changed:
        print("Note: settings changed; full reindex was performed.")


if __name__ == "__main__":
    main()
