# Plan:
# Get next word distribution from Qwen for the focus alternative contexts. 
# Sample an ordering. 
# Then create ordering model, set model, disjunction model, and conjunction model.
# Get negation probabilities and log likelihoods for each experimental trial.

# ------------------ Imports ------------------
import os, math, json, unicodedata, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ Config ------------------
MODEL_NAME = "Qwen/Qwen2-7B"
EXPERIMENTAL_DATA = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/sca_dataframe.csv"
PROMPTS = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/prompts_llm_only.csv"
OUTPUT_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/Qwen_Qwen2-7B_alternative_model_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_LEN = 4096
TOPK_FIRST = 4000        # how many first tokens to consider
BEAM_WIDTH = 3           # beams to finish a word (handles multi-token words)
MAX_WORD_TOKENS = 6      # guardrail to avoid runaway
RNG = np.random.default_rng(123)

def main():


if __name__ == "__main__":
    main()