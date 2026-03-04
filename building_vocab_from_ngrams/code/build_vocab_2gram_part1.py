#!/usr/bin/env python3
import gzip, os, re, json
from collections import defaultdict
from glob import glob
from typing import Dict, List, Set

# ===== EDIT PATHS =====
NGRAM_DIR = "/users/ljohnst7/data/ljohnst7/ngrams/downloaded_files"
NGRAM_GLOB = "googlebooks-eng-all-2gram-*.gz"
TARGET_PATH = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/alternative_task/targets_out/target_2gram.txt"
OUT_JSON = "/users/ljohnst7/data/ljohnst7/ngrams/threshold_2gram.json"
OUT_TARGET_COUNTS = "/users/ljohnst7/data/ljohnst7/ngrams/target_2gram_counts.tsv"
# ======================

WORD_RE = re.compile(r"^[A-Za-z]+$")

def load_targets(path: str) -> List[str]:
    out = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            if WORD_RE.fullmatch(parts[0]) and WORD_RE.fullmatch(parts[1]):
                bg = parts[0] + " " + parts[1]
                if bg not in seen:
                    seen.add(bg)
                    out.append(bg)
    return out

def shard_suffix_for_bigram(bg: str) -> str:
    first_word = bg.split(" ", 1)[0]
    if len(first_word) == 1:
        return first_word + "_"
    return first_word[:2]

def build_shard_index(paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        suffix = os.path.basename(path).rsplit("-", 1)[-1]
        if not suffix.endswith(".gz"):
            continue
        out[suffix[:-3]] = path
    return out

def main():
    gz_files = sorted(glob(os.path.join(NGRAM_DIR, NGRAM_GLOB)))
    if not gz_files:
        raise FileNotFoundError(f"No files matched: {os.path.join(NGRAM_DIR, NGRAM_GLOB)}")

    targets = load_targets(TARGET_PATH)
    if not targets:
        raise ValueError("Target 2-gram list empty after letters-only filtering.")

    counts: Dict[str, int] = {bg: 0 for bg in targets}
    shard_index = build_shard_index(gz_files)
    shard_targets: Dict[str, Set[str]] = defaultdict(set)
    for bg in targets:
        shard_targets[shard_suffix_for_bigram(bg)].add(bg)

    missing_shards = sorted(suffix for suffix in shard_targets if suffix not in shard_index)
    if missing_shards:
        raise FileNotFoundError(
            f"Missing 2-gram shard files for suffixes: {missing_shards}"
        )

    print(
        f"[pass1] targets={len(targets)} required_shards={len(shard_targets)} total_shards={len(gz_files)}",
        flush=True,
    )

    # Only scan the shard(s) that can contain each target bigram.
    for suffix in sorted(shard_targets):
        gz = shard_index[suffix]
        wanted = shard_targets[suffix]
        with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                t1 = line.find("\t")
                if t1 == -1:
                    continue
                token = line[:t1].lower()
                if token not in wanted:
                    continue

                # token is a target; now parse match_count field quickly
                t2 = line.find("\t", t1 + 1)  # after year
                if t2 == -1:
                    continue
                t3 = line.find("\t", t2 + 1)  # after match_count
                if t3 == -1:
                    continue
                try:
                    mc = int(line[t2 + 1 : t3])
                except ValueError:
                    continue
                counts[token] += mc

    missing = [bg for bg in targets if counts.get(bg, 0) == 0]
    if missing:
        raise RuntimeError(
            f"ERROR: {len(missing)} target 2-grams missing (count=0). First 50: {missing[:50]}"
        )

    min_bg = min(targets, key=lambda bg: counts[bg])  # stable via targets order? not needed
    threshold = counts[min_bg]

    out = {
        "min_target_bigram": min_bg,
        "min_target_count": threshold,
        "num_targets": len(targets),
        "num_shards": len(gz_files),
        "targets_path": TARGET_PATH,
        "ngram_dir": NGRAM_DIR,
        "ngram_glob": NGRAM_GLOB,
        "target_counts_path": OUT_TARGET_COUNTS,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TARGET_COUNTS), exist_ok=True)

    with open(OUT_TARGET_COUNTS, "w", encoding="utf-8") as f:
        for bg in targets:
            f.write(f"{bg}\t{counts[bg]}\n")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[threshold] min target='{min_bg}' count={threshold}", flush=True)
    print(f"[wrote] {OUT_TARGET_COUNTS}", flush=True)
    print(f"[wrote] {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
