"""
Preprocess cloze words by context and merge duplicates.

Rules:
1) Ignore frequency.
2) Remove leading "a ", "an ", "my " from words/phrases.
3) Match words after removing spaces.
4) Collapse simple plural forms (e.g., "pens" -> "pen").
5) For each context+word, sum cloze_probability.
6) When a human experiment word shares the same normalized key, use the human label.
7) Save human words that are truly absent from the cloze source after normalization.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipeline/cloze_data/all_cloze_prob_data.csv"
)
DEFAULT_OUTPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipeline/cloze_data/all_cloze_prob_data_preprocessed.csv"
)
DEFAULT_HUMAN_INPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipeline/human_exp_data/sca_dataframe.csv"
)
DEFAULT_MISSING_OUTPUT = Path(
    "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/focus_alt_exp_pipeline/cloze_data/human_words_missing_from_cloze.csv"
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


def load_human_words(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "story" in df.columns:
        context_col = "story"
    elif "context" in df.columns:
        context_col = "context"
    else:
        raise ValueError("Human data must contain either 'story' or 'context'.")

    word_cols = []
    if "cleaned_query" in df.columns:
        word_cols.append("cleaned_query")
    elif "query" in df.columns:
        word_cols.append("query")

    if "cleaned_trigger" in df.columns:
        word_cols.append("cleaned_trigger")
    elif "trigger" in df.columns:
        word_cols.append("trigger")

    if not word_cols:
        raise ValueError(
            "Human data must contain cleaned or raw query/trigger columns."
        )

    parts = []
    for word_col in word_cols:
        parts.append(
            df[[context_col, word_col]].rename(
                columns={context_col: "context", word_col: "word"}
            )
        )

    human_words = pd.concat(parts, ignore_index=True)
    human_words["context"] = human_words["context"].astype(str).str.strip()
    human_words["word"] = human_words["word"].map(normalize_display_word)
    human_words = human_words[human_words["word"] != ""].copy()
    human_words["word_key"] = human_words["word"].map(normalize_match_key)
    human_words["word_key"] = human_words.apply(
        lambda r: canonicalize_key(r["context"], r["word_key"]),
        axis=1,
    )
    human_words = human_words[human_words["word_key"] != ""].copy()
    return human_words[["context", "word", "word_key"]].drop_duplicates().reset_index(drop=True)


def build_human_label_lookup(human_words: pd.DataFrame) -> pd.DataFrame:
    label_candidates = (
        human_words.groupby(["context", "word_key", "word"], as_index=False)
        .size()
        .sort_values(
            ["context", "word_key", "size", "word"],
            ascending=[True, True, False, True],
        )
    )
    return (
        label_candidates.drop_duplicates(subset=["context", "word_key"], keep="first")
        [["context", "word_key", "word"]]
        .rename(columns={"word": "human_word"})
        .reset_index(drop=True)
    )


def write_missing_human_words(
    human_words: pd.DataFrame,
    cloze_words: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    cloze_keys = (
        cloze_words[["context", "word_key"]]
        .drop_duplicates()
        .assign(found_in_cloze=1)
    )
    missing = (
        human_words[["context", "word", "word_key"]]
        .drop_duplicates()
        .merge(cloze_keys, on=["context", "word_key"], how="left")
    )
    missing = (
        missing[missing["found_in_cloze"].isna()]
        .drop(columns=["found_in_cloze"])
        .rename(columns={"word": "human_word"})
        .sort_values(["context", "human_word", "word_key"])
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(output_path, index=False)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and merge cloze words by context.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--human-input", type=Path, default=DEFAULT_HUMAN_INPUT)
    parser.add_argument("--missing-output", type=Path, default=DEFAULT_MISSING_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df[["context", "word", "cloze_probability"]].copy()
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
        .rename(columns={"word": "cloze_word"})
        .reset_index(drop=True)
    )

    human_words = load_human_words(args.human_input)
    human_labels = build_human_label_lookup(human_words)
    missing_human_words = write_missing_human_words(
        human_words=human_words,
        cloze_words=df,
        output_path=args.missing_output,
    )

    totals = (
        df.groupby(["context", "word_key"], as_index=False)["cloze_probability"]
        .sum()
        .merge(labels, on=["context", "word_key"], how="left")
        .merge(human_labels, on=["context", "word_key"], how="left")
    )
    totals["word"] = totals["human_word"]

    needs_fallback = totals["word"].isna()
    totals.loc[needs_fallback, "word"] = totals.loc[needs_fallback].apply(
        lambda r: FORCED_LABELS.get((r["context"], r["word_key"]), r["cloze_word"]),
        axis=1,
    )

    merged = (
        totals[["context", "word", "cloze_probability"]]
        .sort_values(["context", "cloze_probability", "word"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    merged.to_csv(args.output, index=False)
    print(f"[done] Wrote {len(merged)} rows to {args.output}")
    print(
        f"[done] Wrote {len(missing_human_words)} missing human words to {args.missing_output}"
    )


if __name__ == "__main__":
    main()
