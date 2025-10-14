# -*- coding: utf-8 -*-
"""
Top-k next *words* (RAW probabilities) + candidate word RAW probabilities for:
- openai-community/gpt2
- google-bert/bert-large-uncased-whole-word-masking
- meta-llama/Llama-3.2-3B
- Qwen/Qwen2-7B
- deepseek-ai/deepseek-coder-1.3b-base

Causal LMs: subtoken chaining with strict alphabetic continuation to compute P(word|ctx).
BERT: single-piece whole words at [MASK].

All printed probabilities are RAW (no normalization over any subset).
"""

import math
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# =========================
# CONFIG (edit these)
# =========================
PROMPT = (
    "You and your friend Sam go for a long walk together. "
    "After the walk, you go back to Sam's house. You say to Sam, 'I'm thirsty.' "
    "Sam opens the fridge and responds, 'I only have "
)  # <- open quote + exactly one trailing space

TOPK_WORDS      = 20     # how many words to show
START_BEAM      = 800    # how many next-token starts to scan (causal first piece)
MAX_SUBTOKENS   = 8      # max pieces to chain inside a word (causal)
CONT_TOPK       = 30     # continuation breadth per step (causal)
MIN_WORD_LEN    = 2      # display filter (allow 1-char only for {"a","I"})
MAX_WORD_LEN    = 24     # hard stop to avoid 'aaaaa...' tails
ALLOW_HYPHEN    = True   # allow single '-' inside words (e.g., 'sugar-free')

CANDIDATES      = ["water", "milk", "juice", "soda", "beer", "tea", "coffee"]

MODELS: Dict[str, str] = {
    "GPT-2": "openai-community/gpt2",
    "BERT-Large-WWM": "google-bert/bert-large-uncased-whole-word-masking",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",           # requires Meta access
    "Qwen2-7B": "Qwen/Qwen2-7B",
    "DeepSeek-Coder-1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
}

# =========================
# Utilities
# =========================
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

# Detect SentencePiece (LLaMA/Qwen) vs BPE (GPT-2/DeepSeek-Coder)
def is_sentencepiece(tok) -> bool:
    ids = tok.encode(" hello", add_special_tokens=False)
    toks = tok.convert_ids_to_tokens(ids)
    return bool(toks) and ("▁" in toks[0])

# Word filters
_ALPHA = re.compile(r"[A-Za-z]")

def looks_alpha_start(piece: str, sp_mode: bool, prompt_tail: str) -> bool:
    """Is this token a plausible English word start?"""
    if sp_mode:
        if not piece.startswith("▁"):
            return False
        s = piece.lstrip("▁")
        return bool(s) and bool(_ALPHA.match(s[0]))
    else:
        # BPE: require leading space OR whitespace-ended prompt
        if not (piece.startswith(" ") or (prompt_tail == "" or prompt_tail[-1].isspace())):
            return False
        s = piece.lstrip(" ")
        return bool(s) and bool(_ALPHA.match(s[0]))

def clean_piece_text(piece: str, sp_mode: bool) -> str:
    return piece.lstrip("▁") if sp_mode else piece.lstrip(" ")

def keep_word(word: str) -> bool:
    if len(word) < MIN_WORD_LEN and word not in {"a", "I"}:
        return False
    if len(word) > MAX_WORD_LEN:
        return False
    # Allow only letters (and optionally one or more hyphens between letters)
    if ALLOW_HYPHEN:
        return bool(re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", word))
    return word.isalpha()

def piece_is_continuation(piece2: str, sp_mode: bool) -> bool:
    """Continuation must NOT start a new word and must be alphabetic (and hyphen if allowed)."""
    # must NOT start a new word
    if sp_mode and piece2.startswith("▁"):
        return False
    if not sp_mode and piece2.startswith(" "):
        return False
    # strip any sp marker then ensure allowed chars only
    txt = piece2.replace("▁", "")
    if not txt:  # empty or special
        return False
    if ALLOW_HYPHEN:
        return all(ch.isalpha() or ch == "-" for ch in txt)
    return txt.isalpha()

# =========================
# Causal LMs
# =========================
@torch.no_grad()
def causal_word_prob(tok, mdl, prompt: str, word: str) -> float:
    """Exact RAW P(word|ctx) via subtoken chaining (teacher forcing)."""
    ctx = tok(prompt, return_tensors="pt").to(mdl.device)
    cont_ids = tok(" " + word, add_special_tokens=False).input_ids
    total_logp = 0.0
    input_ids = ctx["input_ids"]
    for tid in cont_ids:
        out = mdl(input_ids=input_ids)
        next_probs = F.softmax(out.logits[:, -1, :], dim=-1)[0]
        p = float(next_probs[tid].clamp_min(1e-45))
        total_logp += math.log(p)
        input_ids = torch.cat([input_ids, torch.tensor([[tid]], device=input_ids.device)], dim=1)
    return math.exp(total_logp)

@torch.no_grad()
def causal_topk_words(tok, mdl, prompt: str, k=20, start_beam=600, max_subtokens=8, cont_topk=30) -> List[Tuple[str, float]]:
    """
    Collect many likely word-start tokens; for each, greedily chain alphabetic-only
    continuations (no new word-start) to form a whole word; multiply conditional
    probabilities to get RAW P(word|ctx); return top-k by raw prob.
    """
    device = mdl.device
    enc = tok(prompt, return_tensors="pt").to(device)
    base_probs = F.softmax(mdl(**enc).logits[:, -1, :], dim=-1)[0]  # [V]
    top_p, top_id = base_probs.topk(start_beam)

    sp_mode = is_sentencepiece(tok)
    words: Dict[str, float] = {}

    for p1, tid in zip(top_p.tolist(), top_id.tolist()):
        piece = tok.decode([tid], skip_special_tokens=True)
        if not looks_alpha_start(piece, sp_mode, prompt):
            continue

        spelled = clean_piece_text(piece, sp_mode)
        if len(spelled) > MAX_WORD_LEN:  # early length guard
            continue

        total_logp = math.log(max(p1, 1e-45))
        input_ids = torch.cat([enc["input_ids"], torch.tensor([[tid]], device=device)], dim=1)

        # Greedy continuation within the same word (alphabetic-only)
        for _ in range(max_subtokens - 1):
            out2 = mdl(input_ids=input_ids)
            nxt_probs = F.softmax(out2.logits[:, -1, :], dim=-1)[0]
            nxt_p, nxt_id = nxt_probs.topk(cont_topk)
            chosen = None
            for p2, tid2 in zip(nxt_p.tolist(), nxt_id.tolist()):
                piece2 = tok.decode([tid2], skip_special_tokens=True)
                if piece_is_continuation(piece2, sp_mode):
                    chosen = (p2, tid2, piece2)
                    break
            if chosen is None:
                break
            p2, tid2, piece2 = chosen
            total_logp += math.log(max(p2, 1e-45))
            part = piece2.replace("▁", "") if sp_mode else piece2
            spelled += part
            if len(spelled) > MAX_WORD_LEN:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[tid2]], device=device)], dim=1)

        if keep_word(spelled):
            prob = math.exp(total_logp)
            # If same surface form appears via different tokenizations, keep the max
            if prob > words.get(spelled, 0.0):
                words[spelled] = prob

    if not words:
        return []
    return sorted(words.items(), key=lambda x: x[1], reverse=True)[:k]

def run_causal(model_name: str, model_id: str):
    print(f"\n=== {model_name} ({model_id}) ===")
    tok, mdl = load_causal(model_id)

    # Top-k words (RAW)
    print(f"\nTop-{TOPK_WORDS} words (raw P(word|ctx)):")
    for w, p in causal_topk_words(tok, mdl, PROMPT,
                                  k=TOPK_WORDS,
                                  start_beam=START_BEAM,
                                  max_subtokens=MAX_SUBTOKENS,
                                  cont_topk=CONT_TOPK):
        print(f"{w:<22}\t{p:.8e}")

    # Candidate words (RAW)
    print("\nCandidate words (raw P(word|ctx)):")
    for w in CANDIDATES:
        p = causal_word_prob(tok, mdl, PROMPT, w)
        print(f"P({w!r} | ctx) = {p:.8e}")

# =========================
# BERT (masked LM)
# =========================
@torch.no_grad()
def bert_topk_words(tok, mdl, prompt: str, k=20) -> List[Tuple[str, float]]:
    """
    Top-k whole words at [MASK] with RAW probabilities (single-piece only, no '##').
    """
    text = prompt + tok.mask_token + "'."
    enc = tok(text, return_tensors="pt").to(mdl.device)
    mask_pos = (enc["input_ids"] == tok.mask_token_id).nonzero(as_tuple=False)[0]
    logits = mdl(**enc).logits[mask_pos[0], mask_pos[1], :]
    probs = F.softmax(logits, dim=-1)

    top_p, top_id = probs.topk(min(2000, probs.shape[-1]))
    kept: List[Tuple[str, float]] = []
    for p, tid in zip(top_p.tolist(), top_id.tolist()):
        piece = tok.convert_ids_to_tokens(tid)
        if piece.startswith("##"):
            continue  # not a whole word
        word = tok.decode([tid]).strip()
        if not keep_word(word):
            continue
        kept.append((word, float(p)))
        if len(kept) >= k:
            break
    return kept

@torch.no_grad()
def bert_word_prob(tok, mdl, prompt: str, word: str):
    """RAW P(word|ctx) if the word is single-piece under BERT tokenizer; else None."""
    text = prompt + tok.mask_token + "'."
    enc = tok(text, return_tensors="pt").to(mdl.device)
    mask_pos = (enc["input_ids"] == tok.mask_token_id).nonzero(as_tuple=False)[0]
    logits = mdl(**enc).logits[mask_pos[0], mask_pos[1], :]
    probs = F.softmax(logits, dim=-1)

    ids = tok(" " + word, add_special_tokens=False).input_ids
    if len(ids) == 1:
        return float(probs[ids[0]].item())
    return None

def run_bert(model_name: str, model_id: str):
    print(f"\n=== {model_name} ({model_id}) ===")
    tok, mdl = load_bert(model_id)

    # Top-k words (RAW)
    print(f"\nTop-{TOPK_WORDS} words (raw P(word|ctx)):")
    for w, p in bert_topk_words(tok, mdl, PROMPT, k=TOPK_WORDS):
        print(f"{w:<22}\t{p:.8e}")

    # Candidate words (RAW)
    print("\nCandidate words (raw P(word|ctx); multi-piece => N/A):")
    for w in CANDIDATES:
        p = bert_word_prob(tok, mdl, PROMPT, w)
        if p is None:
            print(f"P({w!r} | ctx) = N/A (multi-piece)")
        else:
            print(f"P({w!r} | ctx) = {p:.8e}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"torch {torch.__version__} | device: {'CUDA' if torch.cuda.is_available() else 'CPU'} | dtype: {pick_dtype()}\n")

    # Causal LMs
    run_causal("GPT-2", MODELS["GPT-2"])
    run_causal("Llama-3.2-3B", MODELS["Llama-3.2-3B"])
    run_causal("Qwen2-7B", MODELS["Qwen2-7B"])
    run_causal("DeepSeek-Coder-1.3B", MODELS["DeepSeek-Coder-1.3B"])

    # BERT
    run_bert("BERT-Large-WWM", MODELS["BERT-Large-WWM"])
