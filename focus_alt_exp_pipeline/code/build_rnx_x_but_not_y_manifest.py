"""Prepare the self-contained R&X query-only contrastive scoring batch (no inference)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import pandas as pd

from rnx_contrastive import DEFAULT_MANIFEST, DEFAULT_SOURCE, PIPELINE, build_manifest, sha256
from scoring_manifest import summarize_scoring_manifest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        label = str(args.source_rows.resolve().relative_to(PIPELINE.parent))
    except ValueError:
        label = str(args.source_rows.resolve())
    manifest = build_manifest(pd.read_csv(args.source_rows), source_file=label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    report = {**asdict(summarize_scoring_manifest(manifest)),
              "source_rows": label, "source_rows_sha256": sha256(args.source_rows),
              "manifest_sha256": sha256(args.output), "scoring_status": "prepared_not_scored",
              "prompt_policy": "neutral prefix + weaker surface + ' but not '; only omitted; score stronger only"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    audit = manifest.groupby(["generation_prompt", "trigger", "query"], as_index=False).agg(
        conditions=("condition", lambda values: "|".join(sorted(values))),
        source_items=("source_row_id", lambda values: "|".join(sorted(values))),
    )
    audit.to_csv(args.output.with_name(args.output.stem + "_prompt_audit.csv"), index=False)
    print(json.dumps(report, indent=2))
    print(f"[complete] wrote {args.output}; no model loaded")


if __name__ == "__main__":
    main()
