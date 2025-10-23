# -*- coding: utf-8 -*-

"""
Batch next-(word|phrase) probabilities for five LLMs, with RAW prob and LOG prob.

Inputs
------
1) prompts_llm.csv
   - columns: story, prompt
2) word_freq_and_cloze_prob.csv
   - columns: context (or story), word, cloze_prob, type
   - we only keep rows where type == "pos"

Output
------
results_next_word_probs.csv with columns:
  llm_name, context, word, cloze_prob, llm_next_word_prob, llm_log_prob

Notes
-----
- Causal LMs (GPT-2, Llama-3.2-3B, Qwen2-7B, DeepSeek-Coder-1.3B):
  teacher-forced across the entire candidate string (single or multi-word).
  Returns RAW P(phrase|prompt) and log P (sum of per-token log probs).

- BERT-Large-uncased-whole-word-masking:
  single [MASK] position; only single-piece words are supported (multi-piece → NaN).
  Returns RAW P(word|prompt) and log P when available.

- No renormalization anywhere.
"""

import math
import re
import gc
import inspect
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import unicodedata
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# ============
# File paths (edit if needed)
# ============
PROMPTS_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/prompts_llm_next_word.csv"
WFC_CSV     = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/word_freq_and_cloze_prob.csv"
OUTPUT_CSV  = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/results_llm_next_word_probs.csv"

# ============
# Models
# ============
MODELS: Dict[str, str] = {
    "GPT-2": "openai-community/gpt2",
    "BERT-Large-WWM": "google-bert/bert-large-uncased-whole-word-masking",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",           # requires Meta HF access
    "Qwen2-7B": "Qwen/Qwen2-7B",
    "DeepSeek-Coder-1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
}

# ============
# Torch helpers
# ============
def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_causal(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", tok.unk_token)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=pick_dtype(), device_map="auto"
    ).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl

def load_bert(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForMaskedLM.from_pretrained(
        model_id, torch_dtype=pick_dtype(), device_map="auto"
    ).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl

# ============
# Probability routines
# ============
def norm_text(s: str) -> str:
    # NFC normalization + strip
    return unicodedata.normalize("NFC", str(s)).strip()

@torch.no_grad()
def causal_phrase_prob(tok, mdl, prompt: str, phrase: str):
    prompt = norm_text(prompt)
    phrase = str(phrase)

    # EXACTLY one trailing space before the continuation
    ptxt = prompt.rstrip() + " "
    enc = tok(ptxt, return_tensors="pt").to(mdl.device)
    input_ids = enc["input_ids"]

    cont_ids = tok(" " + phrase, add_special_tokens=False).input_ids
    if not cont_ids:
        # Nothing to score – return NaNs (so it’s visible upstream)
        return float("nan"), float("nan")

    total_logp = 0.0
    for tid in cont_ids:
        out = mdl(input_ids=input_ids)
        probs = F.softmax(out.logits[:, -1, :], dim=-1)[0]
        p = float(probs[tid].clamp_min(1e-45))
        total_logp += math.log(p)
        input_ids = torch.cat([input_ids, torch.tensor([[tid]], device=input_ids.device)], dim=1)

    return math.exp(total_logp), total_logp

@torch.no_grad()
def bert_word_prob(tok, mdl, prompt: str, word: str) -> Tuple[Optional[float], Optional[float]]:
    """
    RAW P(word|prompt) and logP for BERT at a single [MASK] position.
    Returns (None, None) if the word is multi-piece under BERT tokenizer.
    """
    prompt = norm_text(prompt)
    word = norm_text(word)

    text = prompt.rstrip() + " " + tok.mask_token + "'."
    enc = tok(text, return_tensors="pt").to(mdl.device)
    mask_pos = (enc["input_ids"] == tok.mask_token_id).nonzero(as_tuple=False)
    if mask_pos.numel() == 0:
        return None, None

    i, j = mask_pos[0].tolist()
    logits = mdl(**enc).logits[i, j, :]
    probs = F.softmax(logits, dim=-1)

    ids = tok(" " + word, add_special_tokens=False).input_ids
    if len(ids) == 1:
        p = float(probs[ids[0]].item())
        return p, math.log(max(p, 1e-45))
    return None, None

# ============
# Data loading / matching
# ============
def load_and_match(prompts_csv: str, wfc_csv: str) -> pd.DataFrame:
    df_prompts = pd.read_csv(prompts_csv)
    df_prompts.columns = [c.strip().lower() for c in df_prompts.columns]
    
    if not {"story", "prompt"}.issubset(df_prompts.columns):
        raise ValueError("prompts_llm.csv must have columns: story, prompt")
    
    df_wfc = pd.read_csv(wfc_csv)
    df_wfc.columns = [c.strip().lower() for c in df_wfc.columns]

    df_prompts["story_norm"] = df_prompts["story"].map(norm_text)

    # find context column in wfc (context or story)
    ctx_col = None
    for cand in ["context", "story"]:
        if cand in df_wfc.columns:
            ctx_col = cand
            break
    if ctx_col is None:
        raise ValueError("word_freq_and_cloze_prob.csv must have a 'context' (or 'story') column.")

    df_wfc["context_norm"] = df_wfc[ctx_col].map(norm_text)

    # keep only POS rows
    # if "type" not in df_wfc.columns:
    #     raise ValueError("word_freq_and_cloze_prob.csv must include a 'type' column.")
    # df_wfc = df_wfc[df_wfc["type"].astype(str).str.lower() == "pos"].copy()

    for req in ["word", "cloze_probability"]:
        if req not in df_wfc.columns:
            raise ValueError(f"word_freq_and_cloze_prob.csv is missing required column: {req}")

    merged = pd.merge(
        df_prompts[["story_norm", "story", "prompt"]],
        df_wfc[["context_norm", "word", "cloze_probability"]],
        left_on="story_norm",
        right_on="context_norm",
        how="inner",
    )

    merged = merged.rename(columns={"story": "context"})
    merged = merged[["context", "prompt", "word", "cloze_probability"]].reset_index(drop=True)
    
    if merged.empty:
        print("[warn] No matching contexts after join. Check for accents/quotes/spaces.")
    else:
        print(f"[info] Matched {merged['context'].nunique()} contexts; {len(merged)} rows total.")
    return merged

# ============
# Main batch
# ============
def _from_pretrained_dtype_kw():
    # transformers ≥ 4.44 prefers `dtype`; older uses `torch_dtype`
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    return "dtype" if "dtype" in sig.parameters else "torch_dtype"

def main():
    print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | dtype: {pick_dtype()}")

    df = load_and_match(PROMPTS_CSV, WFC_CSV)
    if df.empty:
        print("No matching contexts after filtering.")
        return

    out_cols = ["llm_name", "context", "word", "cloze_prob", "llm_next_word_prob", "llm_log_prob"]
    pd.DataFrame(columns=out_cols).to_csv(OUTPUT_CSV, index=False)

    def append_rows(rows):
        if rows:
            pd.DataFrame(rows, columns=out_cols).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    # --- Build common kwargs with correct dtype kw name ---
    dtype_kw = _from_pretrained_dtype_kw()
    desired_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # ---------- 1) Causal models on GPU, one at a time ----------
    causal_ids = {k: v for k, v in MODELS.items() if k != "BERT-Large-WWM"}

    for llm_name, model_id in causal_ids.items():
        print(f"\n[info] Loading on GPU: {llm_name} ({model_id})")

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Use accelerate offload with integer GPU index 0; cap VRAM to avoid OOM
        causal_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "max_memory": {0: "20GiB", "cpu": "64GiB"},  # <-- NOTE: int 0, not "cuda:0"
        }
        # add dtype kwarg in a version-compatible way
        causal_kwargs[dtype_kw] = desired_dtype

        mdl = AutoModelForCausalLM.from_pretrained(model_id, **causal_kwargs).eval()
        if hasattr(mdl.config, "use_cache"):
            mdl.config.use_cache = False

        rows = []
        for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"Scoring {llm_name}", ncols=100):
            context = str(r.context)
            prompt  = str(r.prompt)
            word    = str(r.word).strip()
            cloze   = float(r.cloze_probability)
            try:
                p_raw, logp = causal_phrase_prob(tok, mdl, prompt, word)
            except Exception:
                p_raw, logp = float("nan"), float("nan")
            rows.append([llm_name, context, word, cloze, p_raw, logp])
            if len(rows) >= 2000:
                append_rows(rows); rows = []
        append_rows(rows)

        # Free VRAM before next model
        del mdl, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 2) BERT on CPU ----------
    print(f"\n[info] Loading BERT on CPU: {MODELS['BERT-Large-WWM']}")
    bert_tok = AutoTokenizer.from_pretrained(MODELS["BERT-Large-WWM"], use_fast=True)
    # Build kwargs for masked LM with CPU device map
    bert_dtype_kw = _from_pretrained_dtype_kw()
    bert_kwargs = {"device_map": {"": "cpu"}}
    bert_kwargs[bert_dtype_kw] = torch.float32

    bert_mdl = AutoModelForMaskedLM.from_pretrained(MODELS["BERT-Large-WWM"], **bert_kwargs).eval()
    for p in bert_mdl.parameters():
        p.requires_grad_(False)

    rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Scoring BERT", ncols=100):
        context = str(r.context)
        prompt  = str(r.prompt)
        word    = str(r.word).strip()
        cloze   = float(r.cloze_probability)
        try:
            p_raw_b, logp_b = bert_word_prob(bert_tok, bert_mdl, prompt, word)
        except Exception:
            p_raw_b, logp_b = None, None
        rows.append([
            "BERT-Large-WWM", context, word, cloze,
            (float("nan") if p_raw_b is None else p_raw_b),
            (float("nan") if logp_b is None else logp_b),
        ])
        if len(rows) >= 2000:
            append_rows(rows); rows = []
    append_rows(rows)

    print(f"\n[done] Results appended to {OUTPUT_CSV}")

# def main():
#     print(f"torch {torch.__version__} | device: {'CUDA' if torch.cuda.is_available() else 'CPU'} | dtype: {pick_dtype()}")

#     # Load matches
#     df = load_and_match(PROMPTS_CSV, WFC_CSV)
#     if df.empty:
#         print("No matching contexts after filtering type == 'pos'.")
#         return

#     # Preload models
#     causal_ids = {k: v for k, v in MODELS.items() if k != "BERT-Large-WWM"}
#     bert_id = MODELS["BERT-Large-WWM"]

#     causal_models = {}
#     for name, mid in causal_ids.items():
#         print(f"Loading causal model: {name} ({mid})")
#         causal_models[name] = load_causal(mid)

#     print(f"Loading BERT model: BERT-Large-WWM ({bert_id})")
#     bert_tok, bert_mdl = load_bert(bert_id)

#     print("CUDA available:", torch.cuda.is_available())
#     for name, (_, mdl) in causal_models.items():
#         print(f"{name} -> device {mdl.device}, dtype {next(mdl.parameters()).dtype}")
#     print(f"BERT -> device {bert_mdl.device}, dtype {next(bert_mdl.parameters()).dtype}")

#     results = []
#     failure_log = []
#     for row in tqdm(df.itertuples(index=False), total=len(df), desc="Scoring", ncols=100):
#         context = str(row.context)
#         prompt  = str(row.prompt)
#         word    = str(row.word).strip()
#         cloze   = float(row.cloze_probability)

#         # Causal models (support single- and multi-word)
#         for llm_name, (tok, mdl) in causal_models.items():
#             try:
#                 p_raw, logp = causal_phrase_prob(tok, mdl, prompt, word)
#             except Exception as e:
#                 failure_log.append({"llm_name": llm_name, "context": context, "word": word,
#                                     "where": "causal", "error": repr(e)})
#                 p_raw, logp = float("nan"), float("nan")

#             results.append({
#                 "llm_name": llm_name,
#                 "context": context,
#                 "word": word,
#                 "cloze_prob": cloze,
#                 "llm_next_word_prob": p_raw,
#                 "llm_log_prob": logp,
#             })

#         # BERT (single-piece only)
#         try:
#             p_raw_b, logp_b = bert_word_prob(bert_tok, bert_mdl, prompt, word)
#         except Exception as e:
#             failure_log.append({"llm_name": "BERT-Large-WWM", "context": context, "word": word,
#                                 "where": "bert", "error": repr(e)})
#             p_raw_b, logp_b = None, None

#         results.append({
#             "llm_name": "BERT-Large-WWM",
#             "context": context,
#             "word": word,
#             "cloze_prob": cloze,
#             "llm_next_word_prob": (float("nan") if p_raw_b is None else p_raw_b),
#             "llm_log_prob": (float("nan") if logp_b is None else logp_b),
#         })

#     out_df = pd.DataFrame(
#         results,
#         columns=["llm_name", "context", "word", "cloze_prob", "llm_next_word_prob", "llm_log_prob"]
#     )
#     out_df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\nWrote {len(out_df)} rows to {OUTPUT_CSV}")

#     if failure_log:
#         fail_csv = OUTPUT_CSV.replace(".csv", "_failures.csv")
#         pd.DataFrame(failure_log).to_csv(fail_csv, index=False)
#         print(f"[info] Wrote {len(failure_log)} failures -> {fail_csv}")

if __name__ == "__main__":
    main()
