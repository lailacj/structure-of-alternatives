import re

import torch
import torch.nn.functional as F


def _extract_word(token_text: str) -> str:
    text = token_text.replace("\n", " ").replace("\t", " ")
    if not text.strip():
        return ""
    if not text.startswith(" "):
        return ""
    text = text.lstrip()
    word = text.split()[0]
    if not word:
        return ""
    if not re.match(r"^[A-Za-z][A-Za-z'-]*$", word):
        return ""
    return word


@torch.no_grad()
def top_k_next_words(
    model,
    tokenizer,
    context: str,
    *,
    k: int = 10,
    search_multiplier: int = 200,
    add_prefix_space: bool = True,
):
    """
    Return top-k next single-token words from the model distribution at the next position.
    """
    model.eval()
    device = next(model.parameters()).device

    prompt = context.rstrip()
    if add_prefix_space:
        prompt += " "

    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, -1, :][0].float()
    log_probs = F.log_softmax(logits, dim=-1)

    scan_k = min(int(k * search_multiplier), logits.shape[-1])
    top_log_probs, top_ids = torch.topk(log_probs, k=scan_k)

    results = []
    seen = set()
    for tid, lp in zip(top_ids.tolist(), top_log_probs.tolist()):
        token_text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        word = _extract_word(token_text)
        if not word:
            continue
        word_l = word.lower()
        if word_l in seen:
            continue
        seen.add(word_l)
        results.append(
            {
                "word": word,
                "token_id": int(tid),
                "token_str": token_text,
                "logit": float(logits[tid].item()),
                "logprob": float(lp),
                "prob": float(torch.exp(torch.tensor(lp)).item()),
            }
        )
        if len(results) >= k:
            break

    return results
