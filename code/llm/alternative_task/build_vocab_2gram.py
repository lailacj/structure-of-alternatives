#!/usr/bin/env python3
import gzip
import json
import os
import re
from glob import glob
from typing import Dict, Iterable, List, Tuple

# =========================
# PATHS
# =========================
NGRAM_DIR = "/users/ljohnst7/scratch/ljohnst7/google_ngrams"
NGRAM_GLOB = "googlebooks-eng-all-2gram-*.gz"

TARGET_PATH = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/alternative_task/targets_out/target_2gram.txt"

OUT_VOCAB = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_2gram.txt"
OUT_COUNTS = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_2gram_counts.tsv"
OUT_INFO = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_2gram_info.json"

# Memory helper: prune dictionary every N valid lines (0 disables pruning)
PRUNE_EVERY_LINES = 5_000_000
# =========================

WORD_RE = re.compile(r"^[A-Za-z]+$")  # letters-only

def load_targets(path: str) -> List[str]:
    """
    Load targets as "w1 w2" lowercase, letters-only for both tokens.
    De-dupe preserving order.
    """
    raw: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            if WORD_RE.fullmatch(parts[0]) and WORD_RE.fullmatch(parts[1]):
                raw.append(parts[0] + " " + parts[1])

    seen = set()
    out = []
    for bg in raw:
        if bg not in seen:
            seen.add(bg)
            out.append(bg)
    return out

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

    targets = load_targets(TARGET_PATH)
    if not targets:
        raise ValueError("Target 2-gram list is empty after filtering to letters-only bigrams.")

    target_set = set(targets)
    target_counts: Dict[str, int] = {bg: 0 for bg in target_set}

    print(f"[files]   {len(gz_files)} shards")
    print(f"[targets] {len(targets)} targets (letters-only, lowercased bigrams)")

    # PASS 1: count only target bigrams
    print("[pass1] counting target frequencies...")
    for gz in gz_files:
        for bg, mc in iter_2gram(gz):
            if bg in target_counts:
                target_counts[bg] += mc

    missing = [bg for bg in targets if target_counts.get(bg, 0) == 0]
    if missing:
        preview = missing[:50]
        raise RuntimeError(
            f"ERROR: {len(missing)} target 2-grams were missing (frequency=0) in 2-grams.\n"
            f"First {len(preview)} missing: {preview}\n"
            f"Fix: remove/replace these targets or adjust token rules."
        )

    # Threshold = minimum target bigram frequency (ties broken by target file order)
    min_bg = None
    min_count = None
    for bg in targets:
        c = target_counts[bg]
        if min_count is None or c < min_count:
            min_count = c
            min_bg = bg

    assert min_bg is not None and min_count is not None
    threshold = min_count
    print(f"[threshold] min target bigram='{min_bg}' count={threshold}")

    # PASS 2: build full bigram vocab counts and keep >= threshold
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
        "target_path": TARGET_PATH,
        "num_shards": len(gz_files),
        "num_targets": len(targets),
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
