
from pathlib import Path
import pandas as pd

base = Path("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/focus_alt_exp")
files = sorted(base.glob("*_bert_static.csv"))

for path in files:
    df = pd.read_csv(path)
    expected = df["set_boundary"].value_counts().mode().iat[0]

    cleaned = (
        df.groupby("set_boundary", group_keys=False, sort=False)
          .tail(expected)
          .reset_index(drop=True)
    )

    out = path.with_name(path.stem)
    cleaned.to_csv(out, index=False)
    print(f"{path.name}: {len(df)} -> {len(cleaned)} rows, kept last {expected} rows per boundary, wrote {out.name}")
