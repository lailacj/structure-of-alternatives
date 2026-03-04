#!/usr/bin/env python3
import pandas as pd
import re

# —— CONFIGURE THESE PATHS ——
INPUT_PROMPTS_PATH = "../data/prompts_BERT.csv"
INPUT_SCA_PATH     = "../data/sca_dataframe.csv"
OUTPUT_PATH        = "../data/prompts_with_trigger_AND.csv"
# ——————————————————————

def make_trigger_prompt(prompt: str, trigger: str) -> str:
    """
    Replace the last "only have " or "only has " clause in `prompt` (up to the closing apostrophe)
    with "only have/has {trigger} and [MASK].'"
    """
    # look for both variants
    variants = ["only have ", "only has "]
    # find the rightmost occurrence of either
    idx, variant = max(
        ((prompt.rfind(v), v) for v in variants),
        key=lambda x: x[0]
    )
    if idx == -1:
        # no match → return original
        return prompt

    # find the closing apostrophe after variant
    start = idx + len(variant)
    end_quote = prompt.find("'", start)
    suffix = prompt[end_quote+1:] if end_quote != -1 else ""

    # rebuild
    new_mid   = f"{variant}{trigger} and [MASK].'"
    new_prompt = prompt[:idx] + new_mid + suffix
    return new_prompt

def main():
    # 1) Load base prompts and trigger list
    df_prompts = pd.read_csv(INPUT_PROMPTS_PATH)   # expects columns ['story','prompt']
    df_sca     = pd.read_csv(INPUT_SCA_PATH)       # expects columns ['story','trigger', ...]
    
    # 2) Build all unique (story,trigger) pairs
    pairs = df_sca[['story','trigger']].drop_duplicates()
    
    # 3) Merge so each pair carries its story-prompt
    df = pairs.merge(df_prompts, on='story', how='left')
    
    # 4) Check that every story had a base prompt
    missing = df[df['prompt'].isna()]['story'].unique()
    if len(missing) > 0:
        raise RuntimeError(f"No base prompt found for stories: {missing}")
    
    # 5) Create the masked trigger_prompt
    df['prompt_with_mask'] = df.apply(
        lambda row: make_trigger_prompt(row['prompt'], row['trigger']),
        axis=1
    )
    
    # 6) Drop the original prompt column, sort, and write out
    df = df.drop(columns=['prompt'])
    df = df.sort_values(['story','trigger']).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Wrote {len(df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
