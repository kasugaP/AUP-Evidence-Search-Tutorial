"""Verify a fixed model snapshot, then install this project's custom manifest.

This helper was added when preparing the tutorial; it is not a cloud run log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def verify(model_dir: Path, reference: Path) -> int:
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("Unexpected reference model/revision")
    for record in manifest["files"]:
        path = (model_dir / record["file"]).resolve()
        if not path.is_relative_to(model_dir.resolve()):
            raise ValueError("Reference path escapes the model directory")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or incomplete file: {record['file']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"SHA256 mismatch: {record['file']}")
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-Embedding-0.6B"))
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parent.parent / "resources/model_manifest.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    count = verify(args.model_dir, args.reference)
    target = args.model_dir / "model_manifest.json"
    if not args.check_only and args.reference.resolve() != target.resolve():
        shutil.copyfile(args.reference, target)
    print(f"Verified {count} model files by size and SHA256.")
    print("Check only; no files changed." if args.check_only else f"Manifest ready: {target}")
