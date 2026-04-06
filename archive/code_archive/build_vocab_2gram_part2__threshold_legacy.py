#!/usr/bin/env python3
import gzip
import json
import os
import re
from glob import glob
from typing import Dict, Iterable, Tuple

# =========================
# PATHS
# =========================
NGRAM_DIR = "/users/ljohnst7/data/ljohnst7/ngrams/downloaded_files"
NGRAM_GLOB = "googlebooks-eng-all-2gram-*.gz"

THRESHOLD_INFO_PATH = "/users/ljohnst7/data/ljohnst7/ngrams/threshold_2gram.json"

OUT_VOCAB = "/users/ljohnst7/data/ljohnst7/ngrams/vocab_2gram.txt"
OUT_COUNTS = "/users/ljohnst7/data/ljohnst7/ngrams/vocab_2gram_counts.tsv"
OUT_INFO = "/users/ljohnst7/data/ljohnst7/ngrams/vocab_2gram_info.json"

# Memory helper: prune dictionary every N valid lines (0 disables pruning)
PRUNE_EVERY_LINES = 5_000_000
# =========================

WORD_RE = re.compile(r"^[A-Za-z]+$")  # letters-only

def iter_2gram(gz_path: str) -> Iterable[Tuple[str, int]]:
    """
    Yields (bigram_lower, match_count) for letters-only bigrams.
    Expected line: token<TAB>year<TAB>match_count<TAB>volume_count
    token should be "w1 w2" (space separated) for 2-grams.
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            token = parts[0]
            toks = token.split()
            if len(toks) != 2:
                continue

            w1, w2 = toks[0], toks[1]
            if not (WORD_RE.fullmatch(w1) and WORD_RE.fullmatch(w2)):
                continue

            try:
                mc = int(parts[2])
            except ValueError:
                continue

            yield (w1.lower() + " " + w2.lower()), mc

def main():
    gz_files = sorted(glob(os.path.join(NGRAM_DIR, NGRAM_GLOB)))
    if not gz_files:
        raise FileNotFoundError(f"No 2gram gz files found: {os.path.join(NGRAM_DIR, NGRAM_GLOB)}")

    if not os.path.exists(THRESHOLD_INFO_PATH):
        raise FileNotFoundError(f"Threshold info not found: {THRESHOLD_INFO_PATH}")

    with open(THRESHOLD_INFO_PATH, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    threshold = threshold_info.get("min_target_count")
    min_bg = threshold_info.get("min_target_bigram")
    num_targets = threshold_info.get("num_targets")

    if not isinstance(threshold, int):
        raise ValueError(f"Invalid or missing min_target_count in {THRESHOLD_INFO_PATH}")

    print(f"[files]     {len(gz_files)} shards")
    print(f"[threshold] min target bigram='{min_bg}' count={threshold}")

    # Build the full bigram vocab and keep counts at or above the threshold from part 1.
    print("[pass2] building vocab counts (letters-only, lowercased bigrams)...")
    counts: Dict[str, int] = {}
    lines_seen = 0

    for gz in gz_files:
        for bg, mc in iter_2gram(gz):
            counts[bg] = counts.get(bg, 0) + mc
            lines_seen += 1

            if PRUNE_EVERY_LINES and (lines_seen % PRUNE_EVERY_LINES == 0):
                cushion = max(1, threshold // 20)  # 5% cushion (works well when threshold is meaningful)
                cutoff = max(0, threshold - cushion)
                counts = {k: v for k, v in counts.items() if v >= cutoff}
                print(f"[pass2] pruned at {lines_seen:,} lines; dict_size={len(counts):,}")

    kept = [(bg, c) for (bg, c) in counts.items() if c >= threshold]
    kept.sort(key=lambda x: (-x[1], x[0]))
    print(f"[pass2] kept {len(kept):,} bigrams with count >= {threshold}")

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(OUT_COUNTS), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_VOCAB), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_INFO), exist_ok=True)

    # Write counts
    with open(OUT_COUNTS, "w", encoding="utf-8") as out:
        for bg, c in kept:
            out.write(f"{bg}\t{c}\n")

    # Write vocab (bigrams only)
    with open(OUT_VOCAB, "w", encoding="utf-8") as out:
        for bg, _c in kept:
            out.write(bg + "\n")

    info = {
        "ngram_dir": NGRAM_DIR,
        "ngram_glob": NGRAM_GLOB,
        "threshold_info_path": THRESHOLD_INFO_PATH,
        "num_shards": len(gz_files),
        "num_targets": num_targets,
        "min_target_bigram": min_bg,
        "min_target_count": threshold,
        "threshold_inclusive": True,
        "outputs": {
            "vocab": OUT_VOCAB,
            "vocab_counts": OUT_COUNTS,
            "info": OUT_INFO,
        },
        "token_rules": "lowercased; exactly two tokens; each token letters-only (A–Z); no punctuation/numbers/special chars",
        "prune_every_lines": PRUNE_EVERY_LINES,
    }

    with open(OUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("[done]")
    print(f"vocab:   {OUT_VOCAB}")
    print(f"counts:  {OUT_COUNTS}")
    print(f"info:    {OUT_INFO}")

if __name__ == "__main__":
    main()
