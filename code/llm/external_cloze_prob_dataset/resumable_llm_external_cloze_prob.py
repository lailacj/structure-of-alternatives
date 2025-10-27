#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resumable scorer: continues writing LLM next-(word|phrase) probabilities for cloze data,
skipping any (llm_name, sentence, word) rows already present in OUTPUT_CSV.

Input CSV (CLOZE_CSV) columns:
  - sentence_number, sentence, word, cloze_prob

Output CSV:
  - llm_name, sentence, word, human_cloze_prob, llm_next_word_prob, llm_log_next_word
"""

import os
import math
import unicodedata
import inspect
from typing import Dict, Tuple, Optional, Iterable

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# -------------------
# CONFIG (edit paths)
# -------------------
CLOZE_CSV  = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/external_cloze_prob_dataset/cloze_data.csv"
OUTPUT_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/external_cloze_prob_dataset/llm_next_word_from_external_cloze.csv"

MODELS: Dict[str, str] = {
    "GPT-2": "openai-community/gpt2",
    "BERT-Large-WWM": "google-bert/bert-large-uncased-whole-word-masking",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",   # requires Meta HF access
    "Qwen2-7B": "Qwen/Qwen2-7B",
    "DeepSeek-Coder-1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
}

# Start from this model (inclusive). Set to "Qwen2-7B" to resume there.
START_AT_MODEL = "Qwen2-7B"

# Optional: cap GPU memory to avoid OOM (adjust for your card)
MAX_MEMORY = {0: "20GiB", "cpu": "64GiB"}  # integer GPU index 0

# -------------
# Small helpers
# -------------
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()

def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _from_pretrained_dtype_kw():
    # transformers ≥4.44 uses `dtype`; older uses `torch_dtype`
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    return "dtype" if "dtype" in sig.parameters else "torch_dtype"

def build_done_key(llm_name: str, sentence: str, word: str) -> str:
    # Compact, robust key for set membership
    return f"{llm_name}␞{sentence}␞{word}"

# ------------------------
# Loading / scoring functions
# ------------------------
def load_causal(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", tok.unk_token)
    dtype_kw = _from_pretrained_dtype_kw()
    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "max_memory": MAX_MEMORY if torch.cuda.is_available() else None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs[dtype_kw] = pick_dtype()
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if hasattr(mdl.config, "use_cache"):
        mdl.config.use_cache = False
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl

def load_bert(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dtype_kw = _from_pretrained_dtype_kw()
    kwargs = {"device_map": {"": "cpu"}}
    kwargs[dtype_kw] = torch.float32
    mdl = AutoModelForMaskedLM.from_pretrained(model_id, **kwargs).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl

@torch.no_grad()
def causal_phrase_prob(tok, mdl, sentence: str, word: str) -> Tuple[float, float]:
    """
    Exact next-(word|phrase) probability for causal LMs via teacher forcing.
    Returns (raw_prob, log_prob). Supports multi-word phrases.
    """
    sentence = norm_text(sentence)
    word = str(word)  # do NOT normalize/alter the word

    # ensure exactly one trailing space before generation
    prompt = sentence.rstrip() + " "
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    input_ids = enc["input_ids"]

    # prefix a space so we score a true word start
    cont_ids = tok(" " + word, add_special_tokens=False).input_ids
    if not cont_ids:
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
def bert_word_prob(tok, mdl, sentence: str, word: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Next-word probability for BERT at a single [MASK] position.
    Only words that are a single tokenizer piece are supported; multi-piece -> (None, None).
    """
    sentence = norm_text(sentence)
    word = str(word)

    # Mask as the next token (no trailing punctuation here unless you want it)
    text = sentence.rstrip() + " " + tok.mask_token + "."
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    mask_pos = (input_ids == tok.mask_token_id).nonzero(as_tuple=False)
    if mask_pos.numel() == 0:
        return None, None

    i, j = int(mask_pos[0][0]), int(mask_pos[0][1])
    out = mdl(**{k: v.to(mdl.device) for k, v in enc.items()})
    probs = F.softmax(out.logits[i, j, :], dim=-1)

    # Only single-piece candidates are valid under BERT tokenizer
    ids = tok(" " + word, add_special_tokens=False).input_ids
    if len(ids) != 1:
        return None, None
    p = float(probs[ids[0]].item())
    return p, math.log(max(p, 1e-45))

# -------- Resuming helpers --------
def load_existing_keys(path: str) -> set:
    """
    Return a set of keys (llm_name␞sentence␞word) already present in OUTPUT_CSV.
    If file does not exist or is empty header-only, returns empty set.
    """
    if not os.path.exists(path):
        return set()
    try:
        existing = pd.read_csv(path, usecols=["llm_name", "sentence", "word"])
    except Exception:
        return set()
    if existing.empty:
        return set()
    return {
        build_done_key(str(r.llm_name), str(r.sentence), str(r.word))
        for r in existing.itertuples(index=False)
    }

def filter_pending_rows(df: pd.DataFrame, llm_name: str, done_keys: set) -> pd.DataFrame:
    """
    From the full cloze dataframe, return only rows that are not yet in OUTPUT_CSV for this llm.
    """
    keys = (llm_name + "␞" + df["sentence"].astype(str) + "␞" + df["word"].astype(str))
    mask = ~keys.isin(done_keys)
    return df[mask].copy()

def models_in_run_order(models_dict: Dict[str, str], start_at: Optional[str]) -> Iterable[Tuple[str, str]]:
    names = list(models_dict.keys())
    if start_at is None or start_at not in names:
        start_idx = 0
    else:
        start_idx = names.index(start_at)
    for i in range(start_idx, len(names)):
        yield names[i], models_dict[names[i]]

# -----------
# Main script
# -----------
def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load cloze data
    df = pd.read_csv(CLOZE_CSV)
    needed = {"sentence", "word", "cloze_prob"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {needed}")

    # Hygiene
    df["sentence"] = df["sentence"].map(norm_text)
    df["cloze_prob"] = pd.to_numeric(df["cloze_prob"], errors="coerce")

    # Prepare output (write header only if file does not exist)
    cols = ["llm_name", "sentence", "word", "human_cloze_prob", "llm_next_word_prob", "llm_log_next_word"]
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)

    # Load existing keys to resume
    done_keys = load_existing_keys(OUTPUT_CSV)
    print(f"[resume] Found {len(done_keys):,} existing rows in {OUTPUT_CSV}")

    def append(rows):
        if rows:
            pd.DataFrame(rows, columns=cols).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    # ---- Causal models on GPU, one at a time (starting at START_AT_MODEL) ----
    causal_ids = {k: v for k, v in MODELS.items() if k != "BERT-Large-WWM"}
    for llm_name, model_id in models_in_run_order(causal_ids, START_AT_MODEL):
        # Filter to only rows not yet done for this model
        df_pending = filter_pending_rows(df, llm_name, done_keys)
        if df_pending.empty:
            print(f"\n[skip] {llm_name}: nothing pending (already complete).")
            continue

        print(f"\n[info] Loading causal model: {llm_name} ({model_id}) | pending rows: {len(df_pending):,}")
        tok, mdl = load_causal(model_id)

        batch = []
        for r in tqdm(df_pending.itertuples(index=False), total=len(df_pending), desc=f"Scoring {llm_name}", ncols=100):
            sentence = str(r.sentence)
            word     = str(r.word)
            human_p  = float(r.cloze_prob) if np.isfinite(r.cloze_prob) else np.nan
            try:
                p_raw, logp = causal_phrase_prob(tok, mdl, sentence, word)
            except Exception:
                p_raw, logp = float("nan"), float("nan")
            batch.append([llm_name, sentence, word, human_p, p_raw, logp])

            # Periodically flush and update done_keys to survive crashes mid-model
            if len(batch) >= 1000:
                append(batch)
                for rr in batch:
                    done_keys.add(build_done_key(rr[0], rr[1], rr[2]))
                batch = []
        if batch:
            append(batch)
            for rr in batch:
                done_keys.add(build_done_key(rr[0], rr[1], rr[2]))

        # free VRAM
        del tok, mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- BERT on CPU (also resumable) ----
    llm_name = "BERT-Large-WWM"
    df_pending = filter_pending_rows(df, llm_name, done_keys)
    if df_pending.empty:
        print(f"\n[skip] {llm_name}: nothing pending (already complete).")
    else:
        print(f"\n[info] Loading BERT (CPU): {MODELS[llm_name]} | pending rows: {len(df_pending):,}")
        bert_tok, bert_mdl = load_bert(MODELS[llm_name])

        batch = []
        for r in tqdm(df_pending.itertuples(index=False), total=len(df_pending), desc="Scoring BERT", ncols=100):
            sentence = str(r.sentence)
            word     = str(r.word)
            human_p  = float(r.cloze_prob) if np.isfinite(r.cloze_prob) else np.nan
            try:
                p_raw, logp = bert_word_prob(bert_tok, bert_mdl, sentence, word)
            except Exception:
                p_raw, logp = None, None

            batch.append([
                llm_name, sentence, word, human_p,
                (float("nan") if p_raw is None else p_raw),
                (float("nan") if logp is None else logp),
            ])

            if len(batch) >= 2000:
                append(batch)
                for rr in batch:
                    done_keys.add(build_done_key(rr[0], rr[1], rr[2]))
                batch = []
        if batch:
            append(batch)
            for rr in batch:
                done_keys.add(build_done_key(rr[0], rr[1], rr[2]))

    print(f"\n[done] Resumed/updated results at: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
