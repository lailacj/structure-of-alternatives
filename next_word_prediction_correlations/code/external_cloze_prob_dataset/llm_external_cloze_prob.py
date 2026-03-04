#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score next-(word|phrase) probabilities for multiple LLMs against cloze data.

Input CSV (cloze_data.csv) columns:
  - sentence_number, sentence, word, cloze_prob

Output CSV:
  - llm_name, sentence, word, human_cloze_prob, llm_next_word_prob, llm_log_next_word
"""

import os
import math
import unicodedata
import inspect
from typing import Dict, Tuple, Optional

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
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",           # requires Meta HF access
    "Qwen2-7B": "Qwen/Qwen2-7B",
    "DeepSeek-Coder-1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
}

MAX_MEMORY = {0: "20GiB", "cpu": "64GiB"}  

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

# ------------------------
# Loading / scoring functions
# ------------------------
def load_causal(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # pad token helps some hf stacks; use eos/unk if missing
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
    word = str(word)  # do NOT normalize/alter the word string per user’s preference

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
def bert_word_prob(tok, mdl, sentence: str, phrase: str, end_punct: str = "'.") -> Tuple[Optional[float], Optional[float]]:
    """
    Pseudo-probability for a multi-token next 'word'/phrase with BERT.
    """

    sentence = norm_text(sentence)
    phrase = str(phrase)

    piece_ids = tok(" " + phrase, add_special_tokens=False).input_ids  # leading space → word boundary
    if not piece_ids:
        return None, None

    masks = " ".join([tok.mask_token] * len(piece_ids))
    text = sentence.rstrip() + " " + masks + end_punct
    enc = tok(text, return_tensors="pt") 
    input_ids = enc["input_ids"].to(mdl.device)
    attn_mask = enc["attention_mask"].to(mdl.device)

    # Sanity: confirm we have exactly K masks
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask_token_id (not a masked LM tokenizer).")
    num_masks = int((input_ids == mask_id).sum().item())
    if num_masks != len(piece_ids):
        raise ValueError(f"Expected {len(piece_ids)} masks, found {num_masks} in encoded text.")

    total_logp = 0.0

    # Fill masks left->right using gold pieces (pseudo-likelihood)
    for k, tid in enumerate(piece_ids):
        # find the *current* first mask position
        mask_pos = (input_ids == mask_id).nonzero(as_tuple=False)
        if mask_pos.numel() == 0:
            return None, None  # no mask to fill (shouldn't happen if counts align)
        pos = int(mask_pos[0, 1])

        out = mdl(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[0, pos, :]
        probs = F.softmax(logits, dim=-1)

        p = float(torch.clamp(probs[tid], min=1e-45).item())
        total_logp += math.log(p)

        # replace this mask with the target piece; keep remaining masks for next steps
        input_ids[0, pos] = tid

    return math.exp(total_logp), total_logp

# def bert_word_prob(tok, mdl, sentence: str, word: str) -> Tuple[Optional[float], Optional[float]]:
#     """
#     Next-word probability for BERT at a single [MASK] position.
#     Only words that are a single tokenizer piece are supported; multi-piece -> (None, None).
#     """
#     sentence = norm_text(sentence)
#     word = str(word)

#     text = sentence.rstrip() + " " + tok.mask_token + "."
#     enc = tok(text, return_tensors="pt")
#     input_ids = enc["input_ids"]
#     mask_pos = (input_ids == tok.mask_token_id).nonzero(as_tuple=False)
#     if mask_pos.numel() == 0:
#         return None, None

#     i, j = int(mask_pos[0][0]), int(mask_pos[0][1])
#     out = mdl(**{k: v.to(mdl.device) for k, v in enc.items()})
#     probs = F.softmax(out.logits[i, j, :], dim=-1)

#     # Only single-piece candidates are valid under BERT tokenizer
#     ids = tok(" " + word, add_special_tokens=False).input_ids
#     if len(ids) != 1:
#         return None, None
#     p = float(probs[ids[0]].item())
#     return p, math.log(max(p, 1e-45))

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

    # Basic hygiene (don’t alter word strings per your request)
    df["sentence"] = df["sentence"].map(norm_text)
    df["cloze_prob"] = pd.to_numeric(df["cloze_prob"], errors="coerce")

    # Prepare output
    cols = ["llm_name", "sentence", "word", "human_cloze_prob", "llm_next_word_prob", "llm_log_next_word"]
    pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)

    def append(rows):
        if rows:
            pd.DataFrame(rows, columns=cols).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    # ---- Causal models on GPU, one at a time ----
    causal_ids = {k: v for k, v in MODELS.items() if k != "BERT-Large-WWM"}
    for llm_name, model_id in causal_ids.items():
        print(f"\n[info] Loading causal model: {llm_name} ({model_id})")
        tok, mdl = load_causal(model_id)

        batch = []
        for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"Scoring {llm_name}", ncols=100):
            sentence = str(r.sentence)
            word     = str(r.word)
            human_p  = float(r.cloze_prob) if np.isfinite(r.cloze_prob) else np.nan
            try:
                p_raw, logp = causal_phrase_prob(tok, mdl, sentence, word)
            except Exception:
                p_raw, logp = float("nan"), float("nan")
            batch.append([llm_name, sentence, word, human_p, p_raw, logp])

            if len(batch) >= 2000:
                append(batch); batch = []
        append(batch)

        # free VRAM
        del tok, mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- BERT on CPU ----
    print(f"\n[info] Loading BERT (CPU): {MODELS['BERT-Large-WWM']}")
    bert_tok, bert_mdl = load_bert(MODELS["BERT-Large-WWM"])

    batch = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Scoring BERT", ncols=100):
        sentence = str(r.sentence)
        word     = str(r.word)
        human_p  = float(r.cloze_prob) if np.isfinite(r.cloze_prob) else np.nan
        try:
            p_raw, logp = bert_word_prob(bert_tok, bert_mdl, sentence, word)
        except Exception:
            p_raw, logp = None, None

        batch.append([
            "BERT-Large-WWM",
            sentence, word, human_p,
            (float("nan") if p_raw is None else p_raw),
            (float("nan") if logp is None else logp),
        ])
        if len(batch) >= 2000:
            append(batch); batch = []
    append(batch)

    print(f"\n[done] Wrote results to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
