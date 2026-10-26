"""Page-grounded literature retrieval. Dense inference requires an AMD ROCm GPU.

No generated answers, no model training, no automatic scientific conclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = (
    "Instruct: Given a research question about aerobic granular sludge, "
    "retrieve original scientific passages containing the experimental "
    "conditions or evidence needed to answer it.\nQuery: "
)
NOTICE = "候选原文，尚未人工核验。相似度不是答案置信度；未命中不代表原文未报告。"
STOP = set("a an and are as at be been by for from how in is it of on or that the their this to was were what when which with".split())


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def chunks(text, size=1000, overlap=150):
    """Return overlapping, word-boundary chunks without crossing PDF pages."""
    if size <= overlap or overlap < 0:
        raise ValueError("chunk size must exceed non-negative overlap")
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def prepare(args):
    from pypdf import PdfReader
    paths = []
    if args.pdf_dir:
        paths.extend(sorted(Path(args.pdf_dir).rglob("*.pdf")))
    if args.manifest:
        paths.extend(Path(line.strip()) for line in Path(args.manifest).read_text(encoding="utf-8-sig").splitlines()
                     if line.strip() and not line.lstrip().startswith("#"))
    if not paths:
        raise ValueError("No PDFs. Supply --pdf-dir or --manifest.")
    rows, documents, pages, seen = [], [], [], set()
    for path in paths:
        sha = digest(path)
        if sha in seen:
            continue
        seen.add(sha)
        doc_id = "p_" + sha[:16]
        reader = PdfReader(path)
        empty, count = [], 0
        for page_no, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"doc_id": doc_id, "sha256": sha, "file": path.name,
                          "page": page_no, "text": text})
            if len(text.strip()) < 80:
                empty.append(page_no)
                continue
            for part, text_part in enumerate(chunks(text, args.chunk_size, args.overlap), 1):
                rows.append({"doc_id": doc_id, "sha256": sha, "file": path.name,
                             "page": page_no, "chunk_id": f"{doc_id}:p{page_no}:c{part}",
                             "text": text_part})
                count += 1
        documents.append({"doc_id": doc_id, "sha256": sha, "file": path.name,
                          "pdf_pages": len(reader.pages), "chunks": count,
                          "unindexed_short_or_empty_pages": empty})
    if not rows:
        raise ValueError("No usable text was extracted. Inspect PDFs; OCR is outside this starter.")
    root = Path(args.work)
    root.mkdir(parents=True, exist_ok=True)
    (root / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (root / "pages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pages), encoding="utf-8")
    dump(root / "catalog.json", {"notice": NOTICE, "page_numbering": "1-based PDF physical page, not printed journal page",
                                "chunk_size_characters": args.chunk_size, "overlap_characters": args.overlap,
                                "documents": documents})
    print(json.dumps({"documents": len(documents), "chunks": len(rows),
                      "warning": "Manually inspect page extraction, tables, numbers and units."}, ensure_ascii=False))


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, rows):
        self.counts = [Counter(tokens(r["text"])) for r in rows]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / len(rows)
        self.df = Counter(t for c in self.counts for t in c)

    def score(self, query):
        scores = [0.0] * len(self.counts)
        for term in set(tokens(query)):
            df = self.df[term]
            if not df:
                continue
            idf = math.log(1 + (len(scores) - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self.counts):
                tf = counts[term]
                if tf:
                    normalizer = tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avg)
                    scores[i] += idf * tf * 2.5 / normalizer
        return scores


def hardware():
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "hip": torch.version.hip, "device_available": torch.cuda.is_available()}
    if not info["hip"] or not info["device_available"]:
        raise RuntimeError("AMD ROCm GPU unavailable. Select the AUP GPU environment; do not count a CPU run as AMD cloud validation. " + str(info))
    info["device"] = torch.cuda.get_device_name(0)
    info["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    x = torch.ones((16, 16), device="cuda")
    value = (x @ x).sum().item()
    torch.cuda.synchronize()
    if value != 4096:
        raise RuntimeError("GPU tensor smoke check failed")
    info["tensor_check_passed"] = True
    for package in ["sentence-transformers", "transformers", "numpy", "pypdf", "huggingface-hub"]:
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = None
    return info


def resolve_model(revision=None):
    requested = revision or MODEL_REVISION
    candidate = Path(os.environ.get("AUP_EMBEDDING_MODEL_DIR", Path(__file__).parent / "models/Qwen3-Embedding-0.6B"))
    if candidate.exists():
        record = json.loads((candidate / "model_manifest.json").read_text(encoding="utf-8"))
        if record["model_id"] != MODEL or record["revision"] != requested:
            raise ValueError("Offline model identity/revision differs from the frozen model")
        for item in record["files"]:
            path = candidate / item["file"]
            if not path.exists() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Offline model file missing or incomplete: {item['file']}")
        return str(candidate.resolve()), requested, True
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL, revision=requested)
    return model_path, Path(model_path).name, False


def encoder(revision, max_tokens):
    from sentence_transformers import SentenceTransformer
    import torch
    start = time.perf_counter()
    model_path, actual_revision, offline = resolve_model(revision)
    model = SentenceTransformer(model_path, device="cuda", trust_remote_code=False,
                                local_files_only=offline, model_kwargs={"attn_implementation": "eager"})
    model.tokenizer.padding_side = "left"
    model.max_seq_length = max_tokens
    model.eval()
    torch.cuda.synchronize()
    return model, actual_revision, time.perf_counter() - start


def index(args):
    import numpy as np
    import torch
    info = hardware()
    root = Path(args.work)
    rows = read_jsonl(root / "chunks.jsonl")
    model, revision, load_seconds = encoder(args.revision, args.max_tokens)
    texts = [r["text"] for r in rows]
    too_long = sum(len(model.tokenizer.encode(t, add_special_tokens=True, truncation=False)) > args.max_tokens for t in texts)
    if too_long:
        raise ValueError(f"{too_long} chunks exceed {args.max_tokens} tokens. Reduce --chunk-size during prepare, or raise --max-tokens; silent truncation is not allowed.")
    torch.cuda.synchronize()
    start = time.perf_counter()
    vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.002):
        raise ValueError("Invalid embedding values or normalization")
    np.save(root / "vectors.npy", vectors.astype("float32"), allow_pickle=False)
    dump(root / "index_metadata.json", {"hardware": info, "model": MODEL, "revision": revision,
                                       "max_tokens": args.max_tokens, "batch_size": args.batch_size,
                                       "query_prompt": INSTRUCTION, "chunks_sha256": digest(root / "chunks.jsonl"),
                                       "chunks": len(rows), "embedding_dimensions": int(vectors.shape[1]),
                                       "overlong_chunks": too_long, "model_load_seconds": load_seconds,
                                       "embedding_seconds": elapsed, "scope": "inference only; no training"})
    print(f"AMD index saved: {len(rows)} chunks, revision {revision}")


def smoke(args):
    import torch
    info = hardware()
    model, revision, elapsed = encoder(args.revision, 512)
    vectors = model.encode(["Aerobic granular sludge", "Sodium chloride concentration"],
                           batch_size=2, normalize_embeddings=True, convert_to_numpy=True)
    torch.cuda.synchronize()
    dump(Path(args.work) / "model_smoke.json", {"hardware": info, "model": MODEL,
          "revision": revision, "load_seconds": elapsed, "embedding_shape": list(vectors.shape),
          "scope": "two self-written phrases only, not a scientific evaluation"})
    print(f"AMD model smoke check passed: {vectors.shape}")


class Retriever:
    def __init__(self, work, method):
        root = Path(work)
        self.rows = read_jsonl(root / "chunks.jsonl")
        self.method = method
        if method == "bm25":
            self.lexical = BM25(self.rows)
            return
        import numpy as np
        self.device = hardware()
        self.meta = json.loads((root / "index_metadata.json").read_text(encoding="utf-8"))
        if digest(root / "chunks.jsonl") != self.meta["chunks_sha256"]:
            raise ValueError("Corpus changed after indexing. Rebuild the dense index.")
        self.vectors = np.load(root / "vectors.npy", allow_pickle=False)
        if len(self.rows) != len(self.vectors):
            raise ValueError("Index length differs from corpus length")
        self.model, revision, self.load_seconds = encoder(self.meta["revision"], self.meta["max_tokens"])
        if revision != self.meta["revision"]:
            raise ValueError("Model revision changed")

    def search(self, query, top_k=5):
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if self.method == "bm25":
            scores = self.lexical.score(query)
        else:
            import torch
            full_query = INSTRUCTION + query
            if len(self.model.tokenizer.encode(full_query, truncation=False)) > self.meta["max_tokens"]:
                raise ValueError("Query exceeds the saved token limit")
            vector = self.model.encode([query], prompt=INSTRUCTION, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)[0]
            torch.cuda.synchronize()
            scores = (self.vectors @ vector).tolist()
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [{"rank": n, "score": float(scores[i]), **self.rows[i]} for n, i in enumerate(order, 1)
                if self.method != "bm25" or scores[i] > 0]


def page_key(row):
    return f"{row['doc_id']}:p{row['page']}"


def evaluate(args):
    engine = Retriever(args.work, args.method)
    questions = [q for q in read_jsonl(args.questions) if q.get("split") == args.split]
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Selected split must contain questions with unique ids")
    available = {page_key(row) for row in engine.rows}
    for q in questions:
        if not q.get("gold_pages") or not set(q["gold_pages"]) <= available or not q.get("key_quote", "").strip():
            raise ValueError(f"{q['id']}: manually fill valid gold_pages and key_quote before evaluation")
    records, manual = [], []
    for q in questions:
        start = time.perf_counter()
        results = engine.search(q["query"], args.top_k)
        elapsed = time.perf_counter() - start
        first = next((r["rank"] for r in results if page_key(r) in q["gold_pages"]), None)
        quote = re.sub(r"\s+", " ", q["key_quote"]).strip()
        exact_quote_hit = any(page_key(r) in q["gold_pages"] and quote in re.sub(r"\s+", " ", r["text"]).strip() for r in results)
        records.append({"id": q["id"], "query": q["query"], "gold_pages": q["gold_pages"],
                        "key_quote": q["key_quote"], "page_hit": first is not None,
                        "exact_quote_hit": exact_quote_hit,
                        "first_relevant_page_rank": first, "reciprocal_rank": 1 / first if first else 0,
                        "query_seconds_excluding_model_load": elapsed, "results": results})
        for r in results:
            manual.append({"query_id": q["id"], "query": q["query"], "method": args.method,
                           "rank": r["rank"], "file": r["file"], "pdf_page": r["page"],
                           "gold_page_match": page_key(r) in q["gold_pages"], "key_quote": q["key_quote"],
                           "candidate_text": r["text"], "supports_key_quote_manual": "",
                           "conditions_checked_manual": "", "notes_manual": ""})
    report = {"notice": NOTICE, "method": args.method, "split": args.split, "questions": len(questions),
              "corpus_sha256": digest(Path(args.work) / "chunks.jsonl"),
              "questions_sha256": digest(args.questions),
              "labels": "source-checked assistant draft; confirm independently before final scientific reporting",
              "top_k": args.top_k, "page_hit_rate": sum(r["page_hit"] for r in records) / len(records),
              "exact_quote_hit_rate": sum(r["exact_quote_hit"] for r in records) / len(records),
              "page_mrr_at_k": sum(r["reciprocal_rank"] for r in records) / len(records),
              "limitation": "Manually inspect support. Page hits and strict literal-quote hits do not establish scientific validity or semantic answer accuracy.",
              "records": records}
    output = Path(args.work) / f"evaluation_{args.method}_{args.split}.json"
    dump(output, report)
    if manual:
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manual[0]))
            writer.writeheader()
            writer.writerows(manual)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--pdf-dir")
    p.add_argument("--manifest")
    p.add_argument("--work", default="private_work")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=150)
    p.set_defaults(function=prepare)
    p = sub.add_parser("check")
    p.add_argument("--work", default="private_work")
    p.set_defaults(function=lambda a: (dump(Path(a.work) / "hardware.json", hardware()), print("AMD GPU check passed")))
    p = sub.add_parser("index")
    p.add_argument("--work", default="private_work")
    p.add_argument("--revision", default=None, help="Optional exact Hugging Face commit; resolved commit is saved")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=576)
    p.set_defaults(function=index)
    p = sub.add_parser("smoke")
    p.add_argument("--work", default="private_work")
    p.add_argument("--revision", default=None)
    p.set_defaults(function=smoke)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--work", default="private_work")
    p.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out")
    def search_command(a):
        report = {"notice": NOTICE, "query": a.query, "method": a.method,
                  "results": Retriever(a.work, a.method).search(a.query, a.top_k)}
        if a.out:
            dump(a.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    p.set_defaults(function=search_command)
    p = sub.add_parser("evaluate")
    p.add_argument("--work", default="private_work")
    p.add_argument("--questions", default="questions.jsonl")
    p.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    if hasattr(args, "top_k") and args.top_k <= 0:
        parser.error("top-k must be positive")
    args.function(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
