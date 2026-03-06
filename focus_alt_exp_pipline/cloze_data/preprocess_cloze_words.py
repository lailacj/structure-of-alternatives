"""
Preprocess cloze words by context and merge duplicates.

Rules:
1) Ignore frequency.
2) Remove leading "a ", "an ", "my " from words/phrases.
3) Match words after removing spaces.
4) Collapse simple plural forms (e.g., "pens" -> "pen").
5) For each context+word, sum cloze_probability.
6) Keep original-style spacing in output word labels (prefer spaced variants when present).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipline/cloze_data/all_cloze_prob_data.csv"
)
DEFAULT_OUTPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipline/cloze_data/all_cloze_prob_data_preprocessed.csv"
)

REQUIRED_COLUMNS = {"context", "word", "cloze_probability"}
LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|my)\s+", flags=re.IGNORECASE)

# context -> alias key -> canonical key
CANONICAL_KEY_ALIASES = {
    "bag": {
        "makup": "makeup",
        "makeup": "makeup",
        "makeupcase": "makeup",
        "eybrowpencil": "eyebrowpencil",
        "eyebrowpencil": "eyebrowpencil",
        "cellphone": "phone",
        "phone": "phone",
        "medecine": "medicine",
        "medicine": "medicine",
    },
    "bakery": {
        "macarons": "macarons",
        "macaroons": "macarons",
        "macroons": "macarons",
        "biscotti": "biscotti",
        "biscottis": "biscotti",
        "smoothie": "smoothie",
        "smoothy": "smoothie",
        "cinnamonroll": "cinnamonroll",
        "cinamonroll": "cinnamonroll",
        "croissant": "croissant",
        "croisant": "croissant",
        "crossant": "croissant",
    },
    "cut": {
        "scissor": "scissor",
        "sccissor": "scissor",
        "scissior": "scissor",
        "pocketknif": "pocketknife",
        "pocketkif": "pocketknife",
        "pocketknife": "pocketknife",
    },
    "hot": {
        "ceilingfan": "ceilingfan",
        "celingfan": "ceilingfan",
        "calingfan": "ceilingfan",
        "lemonade": "lemonade",
        "lemonaid": "lemonade",
        "lemonad": "lemonade",
        "gatorade": "gatorade",
        "gatoraid": "gatorade",
        "gatorad": "gatorade",
        "popsicle": "popsicle",
        "popsical": "popsicle",
        "popsicl": "popsicle",
    },
    "mall": {
        "burger": "burger",
        "burguer": "burger",
        "hotdog": "hotdog",
        "hotsdog": "hotdog",
    },
    "mask": {
        "handkerchief": "handkerchief",
        "hankerchief": "handkerchief",
        "handsanitizer": "handsanitizer",
        "handsanitizor": "handsanitizer",
        "handsanitzer": "handsanitizer",
        "extramask": "extramask",
        "extramak": "extramask",
        "bandana": "bandana",
        "bandanna": "bandana",
    },
    "meat": {
        "steak": "steak",
        "stak": "steak",
        "broccoli": "broccoli",
        "brocolli": "broccoli",
        "bokchoy": "bokchoy",
        "bokchoi": "bokchoy",
        "dessert": "dessert",
        "desert": "dessert",
    },
    "restaurant": {
        "whiskey": "whiskey",
        "whisky": "whiskey",
        "icedtea": "icedtea",
        "icetea": "icedtea",
        "liquor": "liquor",
        "liquour": "liquor",
        "smoothie": "smoothie",
        "smoothy": "smoothie",
    },
    "salad": {
        "crouton": "crouton",
        "cruton": "crouton",
        "couton": "crouton",
        "cruoton": "crouton",
        "ceasarsalad": "caesarsalad",
        "cesarsalad": "caesarsalad",
        "potatoe": "potato",
        "potato": "potato",
    },
    "science": {
        "psychology": "psychology",
        "pschology": "psychology",
        "meteorology": "meteorology",
        "meterology": "meteorology",
        "meteorolgy": "meteorology",
    },
    "throw": {
        "whiffleball": "whiffleball",
        "wiffleball": "whiffleball",
    },
    "transport": {
        "scooter": "scooter",
        "scoter": "scooter",
    },
}

# context + canonical key -> output display label
FORCED_LABELS = {
    ("bag", "makeup"): "makeup",
    ("bag", "eyebrowpencil"): "eyebrow pencil",
    ("bag", "medicine"): "medicine",
    ("bakery", "macarons"): "macarons",
    ("bakery", "biscotti"): "biscotti",
    ("bakery", "smoothie"): "smoothie",
    ("bakery", "cinnamonroll"): "cinnamon rolls",
    ("bakery", "croissant"): "croissants",
    ("cut", "scissor"): "scissor",
    ("cut", "pocketknife"): "pocket knife",
    ("hot", "ceilingfan"): "ceiling fan",
    ("hot", "lemonade"): "lemonade",
    ("hot", "gatorade"): "gatorade",
    ("hot", "popsicle"): "popsicle",
    ("mall", "burger"): "burgers",
    ("mall", "hotdog"): "hot dogs",
    ("mask", "handkerchief"): "handkerchief",
    ("mask", "handsanitizer"): "hand sanitizer",
    ("mask", "extramask"): "extra mask",
    ("mask", "bandana"): "bandana",
    ("meat", "steak"): "steak",
    ("meat", "broccoli"): "broccoli",
    ("meat", "bokchoy"): "bok choy",
    ("meat", "dessert"): "dessert",
    ("restaurant", "whiskey"): "whiskey",
    ("restaurant", "icedtea"): "iced tea",
    ("restaurant", "liquor"): "liquor",
    ("restaurant", "smoothie"): "smoothie",
    ("salad", "crouton"): "croutons",
    ("salad", "caesarsalad"): "caesar salad",
    ("salad", "potato"): "potatoes",
    ("science", "psychology"): "psychology",
    ("science", "meteorology"): "meteorology",
    ("throw", "whiffleball"): "whiffle ball",
    ("transport", "scooter"): "scooter",
}


def singularize_simple(word: str) -> str:
    """Very small heuristic for plural -> singular collapsing."""
    if len(word) <= 3:
        return word

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]

    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]

    return word


def normalize_display_word(word: object) -> str:
    text = "" if pd.isna(word) else str(word).strip().casefold()
    if not text:
        return ""

    text = LEADING_ARTICLE_RE.sub("", text)
    text = " ".join(text.split())
    return text


def normalize_match_key(display_word: str) -> str:
    key = display_word.replace(" ", "")
    key = re.sub(r"[^a-z0-9]", "", key)
    key = singularize_simple(key)
    return key


def canonicalize_key(context: str, key: str) -> str:
    aliases = CANONICAL_KEY_ALIASES.get(context, {})
    return aliases.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and merge cloze words by context.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["context"] = df["context"].astype(str).str.strip()
    df["word"] = df["word"].map(normalize_display_word)
    df["cloze_probability"] = pd.to_numeric(df["cloze_probability"], errors="coerce")
    df = df.dropna(subset=["cloze_probability"])
    df = df[df["word"] != ""]
    df["word_key"] = df["word"].map(normalize_match_key)
    df["word_key"] = df.apply(lambda r: canonicalize_key(r["context"], r["word_key"]), axis=1)
    df = df[df["word_key"] != ""]

    # Keep one display label per (context, word_key), preferring variants with spaces.
    label_candidates = (
        df.groupby(["context", "word_key", "word"], as_index=False)["cloze_probability"]
        .sum()
        .assign(has_space=lambda d: d["word"].str.contains(" ", regex=False))
        .sort_values(
            ["context", "word_key", "has_space", "cloze_probability", "word"],
            ascending=[True, True, False, False, True],
        )
    )
    labels = (
        label_candidates.drop_duplicates(subset=["context", "word_key"], keep="first")
        [["context", "word_key", "word"]]
        .reset_index(drop=True)
    )

    totals = (
        df.groupby(["context", "word_key"], as_index=False)["cloze_probability"]
        .sum()
        .merge(labels, on=["context", "word_key"], how="left")
    )
    totals["word"] = totals.apply(
        lambda r: FORCED_LABELS.get((r["context"], r["word_key"]), r["word"]),
        axis=1,
    )

    merged = (
        totals[["context", "word", "cloze_probability"]]
        .sort_values(["context", "cloze_probability", "word"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    merged.to_csv(args.output, index=False)
    print(f"[done] Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
