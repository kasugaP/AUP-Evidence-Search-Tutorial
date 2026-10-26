"""Check prepared pages, chunks and reference questions before evaluation.

This tutorial helper checks extracted text, not independent scientific validity.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evidence_search import INSTRUCTION, digest, read_jsonl, tokens


def inspect(work: Path, questions_path: Path, max_tokens: int, tokenizer_path: Path | None = None):
    pages = read_jsonl(work / "pages.jsonl")
    chunks = read_jsonl(work / "chunks.jsonl")
    questions = read_jsonl(questions_path)
    if not pages or not chunks or not questions:
        raise ValueError("Pages, chunks and questions must all be nonempty")
    def page_key(row):
        return f"{row['doc_id']}:p{row['page']}"
    def normalize(text):
        return " ".join(text.split())
    page_texts = {page_key(row): normalize(row["text"]) for row in pages}
    if len(page_texts) != len(pages):
        raise ValueError("Repeated document/page identifiers")
    if len({row["chunk_id"] for row in chunks}) != len(chunks):
        raise ValueError("Repeated chunk IDs")
    by_page = {}
    for chunk in chunks:
        key = page_key(chunk)
        if key not in page_texts:
            raise ValueError("Chunk refers to an unknown page")
        by_page.setdefault(key, []).append(normalize(chunk["text"]))
    ids = [row.get("id") for row in questions]
    if not all(isinstance(value, str) and value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be nonempty unique strings")
    counts = Counter(row.get("split") for row in questions)
    if set(counts) != {"dev", "test"}:
        raise ValueError("Provide both dev and test questions, and no other split")
    for row in questions:
        name = row["id"]
        if not isinstance(row.get("query"), str) or not tokens(row["query"]):
            raise ValueError(f"{name}: provide an English query for this BM25 implementation")
        gold = row.get("gold_pages")
        if not isinstance(gold, list) or not gold or not all(key in page_texts for key in gold):
            raise ValueError(f"{name}: unknown or missing gold page")
        quote = normalize(row.get("key_quote", ""))
        if not quote or not any(quote in page_texts[key] for key in gold):
            raise ValueError(f"{name}: reference quote does not match an indicated source page")
        if not any(quote in text for key in gold for text in by_page.get(key, [])):
            raise ValueError(f"{name}: quote crosses chunks or the source page was not indexed")
    token_record = {"executed": False}
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        max_chunk = max(len(tokenizer.encode(row["text"]).ids) for row in chunks)
        max_query = max(len(tokenizer.encode(INSTRUCTION + row["query"]).ids) for row in questions)
        if max(max_chunk, max_query) > max_tokens:
            raise ValueError(f"Input exceeds {max_tokens} tokens: chunk={max_chunk}, query={max_query}")
        token_record = {"executed": True, "max_chunk_tokens": max_chunk,
                        "max_instructed_query_tokens": max_query, "limit": max_tokens}
    return {
        "documents": len({row["doc_id"] for row in pages}), "physical_pages": len(pages),
        "chunks": len(chunks), "questions": len(questions),
        "development_questions": counts["dev"], "test_questions": counts["test"],
        "source_quote_matches": len(questions), "quote_in_one_chunk": len(questions),
        "corpus_sha256": digest(work / "chunks.jsonl"),
        "questions_sha256": digest(questions_path), "token_check": token_record,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Checks against extracted source text; independent human review is still required.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("private_work"))
    parser.add_argument("--questions", type=Path, default=Path("questions.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=576)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = inspect(args.work, args.questions, args.max_tokens, args.tokenizer)
    if not args.check_only:
        (args.work / "input_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
