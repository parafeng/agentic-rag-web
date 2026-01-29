from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_index import RAGIndex, RAGSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG retrieval eval.")
    parser.add_argument("--file", default="eval/qa.json", help="Path to QA json.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval.")
    parser.add_argument("--show", action="store_true", help="Print top sources for each query.")
    return parser.parse_args()


def _match_expected(sources: list[str], expected: list[str]) -> bool:
    for exp in expected:
        if any(exp in src for src in sources):
            return True
    return False


def main() -> int:
    args = parse_args()
    settings = RAGSettings.from_env()
    index = RAGIndex.load(settings)
    if not index.available:
        print("Index not found. Run: python ingest.py")
        return 1

    qa_path = Path(args.file)
    cases = json.loads(qa_path.read_text(encoding="utf-8"))
    total = len(cases)
    passed = 0

    for case in cases:
        question = case["question"]
        expected = case.get("expected_sources", [])
        results = index.query(question, top_k=args.top_k)
        sources = [item.get("source_path", "") for item in results]
        ok = _match_expected(sources, expected)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.get('id', 'case')} -> {question}")
        if args.show:
            for rank, item in enumerate(results, start=1):
                print(f"  {rank}. {item.get('source_path')} (page {item.get('page_start')})")
        if ok:
            passed += 1

    print(f"Summary: {passed}/{total} passed.")
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
