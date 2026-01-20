import gzip
import json
import os
import re
from glob import glob
from typing import Dict, Iterable, List, Set, Tuple

# ------------------ Paths -------------------
NGRAM_DIR = "/users/ljohnst7/scratch/ljohnst7/google_ngrams"
NGRAM_GLOB_PATTERN = "googlebooks-eng-all-1gram-*.gz"
TARGET_1GRAM_PATH = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/alternative_task/targets_out/target_1gram.txt"

OUT_VOCAB = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_1gram.txt"
OUT_COUNTS = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_1gram_counts.tsv"
OUT_INFO = "/users/ljohnst7/scratch/ljohnst7/vocab/vocab_1gram_info.json"

PRUNE_EVERY_LINES = 5_000_000

WORD_RE = re.compile(r"^[A-Za-z]+$")  # letters only

def load_targets(path: str) -> List[str]:
    """Load targets, lowercase, letters-only, de-dupe preserving order."""
    raw: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if not w:
                continue
            if WORD_RE.fullmatch(w):
                raw.append(w)

    seen = set()
    out = []
    for w in raw:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def iter_1gram(gz_path: str) -> Iterable[Tuple[str, int]]:
    """
    Yields (token_lower, match_count) for letters-only tokens.
    Expected line: token<TAB>year<TAB>match_count<TAB>volume_count
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            raw = parts[0]
            if not WORD_RE.fullmatch(raw):
                continue

            try:
                mc = int(parts[2])
            except ValueError:
                continue

            yield raw.lower(), mc

def main():
    gz_files = sorted(glob(os.path.join(NGRAM_DIR, NGRAM_GLOB_PATTERN)))
    if not gz_files:
        raise FileNotFoundError(f"No 1gram gz files found: {os.path.join(NGRAM_DIR, NGRAM_GLOB)}")

    targets = load_targets(TARGET_1GRAM_PATH)
    if not targets:
        raise ValueError("Target list is empty after letters-only filtering.")

    target_set = set(targets)
    target_counts: Dict[str, int] = {w: 0 for w in target_set}

    print(f"[files]   {len(gz_files)} shards")
    print(f"[targets] {len(targets)} targets (letters-only, lowercased)")

    # PASS 1: count only targets
    print("[pass1] counting target frequencies...")
    for gz in gz_files:
        for tok, mc in iter_1gram(gz):
            if tok in target_counts:
                target_counts[tok] += mc

    missing = [w for w in targets if target_counts.get(w, 0) == 0]
    if missing:
        # Show up to first 50 missing to avoid huge error messages
        preview = missing[:50]
        raise RuntimeError(
            f"ERROR: {len(missing)} target words were missing (frequency=0) in 1-grams.\n"
            f"First {len(preview)} missing: {preview}\n"
            f"Fix: remove/replace these targets or adjust token rules."
        )

    # Threshold is min target frequency (ties broken by target file order)
    min_word = None
    min_count = None
    for w in targets:
        c = target_counts[w]
        if min_count is None or c < min_count:
            min_count = c
            min_word = w

    assert min_word is not None and min_count is not None
    threshold = min_count
    print(f"[threshold] min target word='{min_word}' count={threshold}")

    # PASS 2: build full vocab counts and keep >= threshold
    print("[pass2] building vocab counts (letters-only, lowercased)...")
    counts: Dict[str, int] = {}
    lines_seen = 0

    for gz in gz_files:
        for tok, mc in iter_1gram(gz):
            counts[tok] = counts.get(tok, 0) + mc
            lines_seen += 1

            if PRUNE_EVERY_LINES and (lines_seen % PRUNE_EVERY_LINES == 0):
                # Keep words near threshold to avoid dropping words that accumulate across shards
                cushion = max(1, threshold // 10)
                cutoff = threshold - cushion
                counts = {w: c for w, c in counts.items() if c >= cutoff}
                print(f"[pass2] pruned at {lines_seen:,} lines; dict_size={len(counts):,}")

    kept = [(w, c) for (w, c) in counts.items() if c >= threshold]
    kept.sort(key=lambda x: (-x[1], x[0]))
    print(f"[pass2] kept {len(kept):,} words with count >= {threshold}")

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(OUT_COUNTS), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_VOCAB), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_INFO), exist_ok=True)

    # Write counts
    with open(OUT_COUNTS, "w", encoding="utf-8") as out:
        for w, c in kept:
            out.write(f"{w}\t{c}\n")

    # Write vocab (words only)
    with open(OUT_VOCAB, "w", encoding="utf-8") as out:
        for w, _c in kept:
            out.write(w + "\n")

    info = {
        "ngram_dir": NGRAM_DIR,
        "ngram_glob": NGRAM_GLOB_PATTERN,
        "target_path": TARGET_1GRAM_PATH,
        "num_shards": len(gz_files),
        "num_targets": len(targets),
        "min_target_word": min_word,
        "min_target_count": threshold,
        "threshold_inclusive": True,
        "outputs": {
            "vocab": OUT_VOCAB,
            "vocab_counts": OUT_COUNTS,
            "info": OUT_INFO,
        },
        "token_rules": "lowercased; letters-only (A–Z); no punctuation/numbers/special chars",
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