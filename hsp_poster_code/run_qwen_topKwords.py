from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from qwen_topKwords import top_k_next_words

model_path = "/users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B/"
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
).eval()

context = "Taylor looked in her bag and said, I have a"

top_words = top_k_next_words(mdl, tok, context, k=10)
for rank, row in enumerate(top_words, start=1):
    print(
        f"{rank:2d}. {row['word']:<16} "
        f"p={row['prob']:.6f} logp={row['logprob']:.4f} token={row['token_str']!r}"
    )
