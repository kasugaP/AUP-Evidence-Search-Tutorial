"""Recalculate internal consistency of the frozen 2026-10-24 course archive.

This is not a GPU rerun, an independent reproduction, or a general anonymizer.
Only Python's standard library is required. No paper excerpts are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

EXPECTED_ARCHIVE_SHA256 = "55f2d2a1b549a0b6f79e43a1e814bda99bcfc431bde7f85ccdd64328587beebb"
PUBLIC_HARDWARE_FIELDS = (
    "python", "torch", "hip", "device_available", "device", "total_memory_bytes",
    "tensor_check_passed", "sentence-transformers", "transformers", "numpy", "pypdf", "huggingface-hub",
)
PUBLIC_CONFIG_FIELDS = (
    "model_id", "model_revision", "chunk_size_characters", "overlap_characters", "max_input_tokens",
    "batch_size", "top_k", "reason_for_token_limit", "fallback_if_gpu_out_of_memory",
)


def summarize(archive_path: Path, out_dir: Path) -> dict:
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("This exporter only accepts the audited 2026-10-24 course archive; no output was written.")
    with ZipFile(archive_path) as archive:
        def read(name):
            return json.loads(archive.read(name))

        config = read("run_config.json")
        version = read("questions_version.json")
        preparation = read("private_work/preparation_validation.json")
        index = read("private_work/index_metadata.json")
        hardware = read("private_work/hardware.json")
        smoke = read("private_work/model_smoke.json")
        if not (hardware["hip"] and hardware["device_available"] and hardware["tensor_check_passed"]):
            raise ValueError("The archive does not contain a passed AMD GPU check")
        if not (config["model_revision"] == index["revision"] == smoke["revision"]):
            raise ValueError("Model revisions differ")

        metrics, details = {}, []
        for method in ("bm25", "dense"):
            metrics[method] = {}
            for split in ("dev", "test"):
                report = read(f"private_work/evaluation_{method}_{split}.json")
                records = report["records"]
                count = len(records)
                if count != report["questions"] or not count:
                    raise ValueError("Question count mismatch")
                if len({row["id"] for row in records}) != count:
                    raise ValueError("Repeated question IDs")
                if not (report["corpus_sha256"] == index["chunks_sha256"] == preparation["corpus_sha256"]):
                    raise ValueError("Corpus hashes differ")
                if not (report["questions_sha256"] == version["sha256"] == preparation["question_set_sha256"]):
                    raise ValueError("Question hashes differ")
                if report["top_k"] != config["top_k"]:
                    raise ValueError("Top-k differs")

                page_hits, quote_hits, reciprocal_sum = 0, 0, 0.0
                for row in records:
                    relevant = [hit for hit in row["results"]
                                if f"{hit['doc_id']}:p{hit['page']}" in row["gold_pages"]]
                    first = min((hit["rank"] for hit in relevant), default=None)
                    quote = " ".join(row["key_quote"].split())
                    if not quote:
                        raise ValueError(f"Empty reference quote: {row['id']}")
                    quote_hit = any(quote in " ".join(hit["text"].split()) for hit in relevant)
                    if (first is not None) != row["page_hit"] or first != row["first_relevant_page_rank"]:
                        raise ValueError(f"Inconsistent page result: {row['id']}")
                    if quote_hit != row["exact_quote_hit"]:
                        raise ValueError(f"Inconsistent quote result: {row['id']}")
                    reciprocal = 1.0 / first if first else 0.0
                    if not math.isclose(reciprocal, row["reciprocal_rank"], abs_tol=1e-12):
                        raise ValueError("Reciprocal-rank mismatch")
                    page_hits += first is not None
                    quote_hits += quote_hit
                    reciprocal_sum += reciprocal
                    details.append({
                        "split": split, "method": method, "id": row["id"],
                        "page_hit": bool(first), "exact_quote_hit": quote_hit,
                        "first_relevant_page_rank": first,
                        "reciprocal_rank": reciprocal,
                    })

                values = {"page_hit_rate": page_hits / count,
                          "exact_quote_hit_rate": quote_hits / count,
                          "page_mrr_at_k": reciprocal_sum / count}
                for name, value in values.items():
                    if not math.isclose(value, report[name], abs_tol=1e-12):
                        raise ValueError(f"Incorrect aggregate: {method}/{split}/{name}")
                manual_rows = list(csv.DictReader(io.StringIO(archive.read(
                    f"private_work/evaluation_{method}_{split}.csv").decode("utf-8-sig"))))
                touched = sum(any(row.get(key, "").strip() for key in (
                    "supports_key_quote_manual", "conditions_checked_manual", "notes_manual"))
                    for row in manual_rows)
                metrics[method][split] = {
                    "questions": count, "page_hits": page_hits, "exact_quote_hits": quote_hits,
                    **values, "manual_candidate_rows": len(manual_rows),
                    "manual_rows_with_any_entry": touched,
                }

        summary = {
            "run_date": "2026-10-24",
            "source_archive_sha256": archive_sha256,
            "hardware": {key: hardware[key] for key in PUBLIC_HARDWARE_FIELDS if key in hardware},
            "config": {key: config[key] for key in PUBLIC_CONFIG_FIELDS if key in config},
            "model_smoke": {key: smoke[key] for key in ("model", "revision", "embedding_shape", "load_seconds")},
            "index": {key: index[key] for key in (
                "chunks", "embedding_dimensions", "max_tokens", "batch_size", "overlong_chunks",
                "model_load_seconds", "embedding_seconds", "chunks_sha256", "scope")},
            "preparation": {key: preparation[key] for key in (
                "pdfs", "physical_pages", "chunks", "questions", "development_questions", "test_questions",
                "source_quote_matches", "human_confirmed_questions", "scope")},
            "questions_sha256": version["sha256"],
            "metrics": metrics,
            "checks": {"metrics_recomputed_from_ranked_results": True,
                       "recorded_corpus_and_question_hashes_agree": True,
                       "source_text_exported": False},
            "verification_scope": {
                "kind": "internal consistency of existing exported records only",
                "gpu_inference_rerun": False,
                "original_corpus_and_questions_rehashed": False,
                "cloud_source_identity_verified": False,
                "complete_notebook_execution_log_verified": False,
                "original_scores_reproduced_from_public_materials": False,
                "independent_scientific_review_completed": False,
            },
            "limitations": [
                "Small methods-focused convenience set; all gold spans on PDF physical page 2.",
                "Reference labels were source-checked by an assistant; independent human confirmation is pending.",
                "Question-level dev/test split; source papers overlap between splits.",
                "Downloaded notebook does not preserve the complete final execution log.",
                "No CPU embedding benchmark or repeated timing experiment was performed.",
            ],
        }
        versions = archive.read("private_work/requirements-lock.txt").decode("utf-8")
        selected = {"torch", "triton", "sentence-transformers", "transformers", "tokenizers",
                    "huggingface-hub", "numpy", "pypdf", "ipykernel", "ipython", "jupyterlab"}
        version_lines = [line for line in versions.splitlines()
                         if line.split("==")[0].replace("_", "-").lower() in selected and "==" in line]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "per_question_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (out_dir / "observed_versions.txt").write_text(
        "# Observed in this course image. This is a record, not an installation recipe.\n"
        + "\n".join(version_lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = summarize(args.archive, args.out)
    for method in ("bm25", "dense"):
        values = report["metrics"][method]["test"]
        print(method, values["page_hits"], values["exact_quote_hits"], round(values["page_mrr_at_k"], 6))
    print("Saved numeric summaries; no paper excerpts exported.")
