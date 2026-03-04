# Make a target word list from the triggers and querys in the experimental data. 
# This will be used to build a vocabulary from the ngram corpus. 
# Make a full target word list. 
# Make a target word list for the 1gram words. 
# Make a target word list for the 2gram words.

import pandas as pd
import re
from pathlib import Path

CSV_PATH = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/sca_dataframe.csv"
OUTDIR = Path("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/alternative_task/targets_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"^[A-Za-z]+$")

def is_1gram(s: str) -> bool:
    parts = s.split()
    return len(parts) == 1 and WORD_RE.fullmatch(parts[0]) is not None

def is_2gram(s: str) -> bool:
    parts = s.split()
    return (
        len(parts) == 2
        and WORD_RE.fullmatch(parts[0]) is not None
        and WORD_RE.fullmatch(parts[1]) is not None
    )

df = pd.read_csv(CSV_PATH)

vals = []
for col in ["cleaned_trigger", "cleaned_query"]:
    vals.extend(
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

# keep only 1- or 2-token items, letters-only (per-token)
clean = []
for v in vals:
    if is_1gram(v) or is_2gram(v):
        clean.append(v)

full = sorted(set(clean))
one = sorted({v for v in full if is_1gram(v)})
two = sorted({v for v in full if is_2gram(v)})

(OUTDIR / "target_full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
(OUTDIR / "target_1gram.txt").write_text("\n".join(one) + "\n", encoding="utf-8")
(OUTDIR / "target_2gram.txt").write_text("\n".join(two) + "\n", encoding="utf-8")

print("Wrote:")
print(" target_full.txt  ", len(full))
print(" target_1gram.txt ", len(one))
print(" target_2gram.txt ", len(two))
