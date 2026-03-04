# next_word_prob_llama32_3b_inline.py
import math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-3B"   # change to local path if you downloaded weights

# === EDIT THESE TWO VARIABLES ===
context = "You and your friend Taylor are sitting in a park. You pull out a piece of paper and tell Taylor, 'I just had an idea that I want to remember, but don't have anything to write with.' Taylor looks through her handbag and responds, 'Sure, I have "
word    = "highlighter"

def load_model(model_name: str = MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    mdl.eval()
    return tok, mdl

@torch.inference_mode()
def next_token_distribution(tokenizer, model, context_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=context_ids)
    logits_last = out.logits[:, -1, :]        # [1, V]
    return torch.softmax(logits_last, dim=-1).squeeze(0)  # [V]

def ensure_batch(ids, device):
    return torch.tensor([ids], device=device, dtype=torch.long)

def next_word_prob(tokenizer, model, context: str, word: str, autospace=True):
    # Auto-prepend space if context ends with alphanumeric (to align tokenization)
    if autospace and context and context[-1].isalnum() and not word.startswith(" "):
        word = " " + word

    ctx_ids  = tokenizer(context, add_special_tokens=False)["input_ids"]
    word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]

    logp = 0.0
    running = ctx_ids[:]
    for tid in word_ids:
        probs = next_token_distribution(tokenizer, model, ensure_batch(running, model.device))
        p = float(probs[tid].item())
        p = max(p, 1e-45)  # avoid log(0)
        logp += math.log(p)
        running.append(tid)
    return math.exp(logp)

if __name__ == "__main__":
    tokenizer, model = load_model(MODEL_NAME)
    prob = next_word_prob(tokenizer, model, context, word)
    print("=== Next word probability ===")
    print(f"Context: {repr(context)}")
    print(f"Word:    {repr(word)}")
    print(f"Probability: {prob:.8g}")
