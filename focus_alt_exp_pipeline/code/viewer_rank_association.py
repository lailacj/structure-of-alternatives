"""Rebuild viewer word ranks and exploratory tests from its verified payload.

Never substitute direct-target scores for missing sampling scores. The browser
receives the observations and tests together, not an unvalidated static figure.
No inference, CV selection, or writes to historical results are performed.
"""
import copy
import math

import numpy as np
import pandas as pd

SEED = 20260917
PERMUTATIONS = 199999
BOOTSTRAPS = 20000
SPECS = [
    ("No linking structure", "direct", 0), ("X but not Y", "direct", 1),
    ("ordering", "top_k", 4), ("set", "top_k", 2),
    ("conjunction", "top_k", 5), ("disjunction", "top_k", 7),
    ("set", "top_p", 3), ("conjunction", "top_p", 6),
    ("disjunction", "top_p", 8),
]


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denominator, -1, 1)) if denominator else None


def rho(x, y):
    return correlation(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def tests(points, *, permutations=PERMUTATIONS, bootstraps=BOOTSTRAPS):
    """Two definitions, nine predictors; preserve average ties in all resamples."""
    output, streams = [], {}
    for definition in ("target", "sampling"):
        for structure, variant, model in SPECS:
            group = sorted((p for p in points if p["model"] == model
                            and p[definition] is not None and p["negation"] is not None),
                           key=lambda p: p["context"])
            x = np.array([p[definition] for p in group])
            y = np.array([p["negation"] for p in group])
            n = len(group)
            coefficient = rho(x, y)
            record = dict(definition=definition, model=model, structure=structure,
                          variant=variant, n=n, R=coefficient, p=None, holm9=None,
                          holm18=None, ciLow=None, ciHigh=None, looMin=None, looMax=None,
                          omitted=[p["context"] for p in points if p["model"] == model
                                   and (p[definition] is None or p["negation"] is None)])
            output.append(record)
            if coefficient is None:
                continue
            if n not in streams:
                rng = np.random.default_rng(np.random.SeedSequence([SEED, n, 1]))
                perm = rng.permuted(np.tile(np.arange(n), (permutations, 1)), axis=1)
                indices = np.random.default_rng(np.random.SeedSequence([SEED, n, 2])).integers(n, size=(bootstraps, n))
                streams[n] = perm, indices
            perm, indices = streams[n]
            xr, yr = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
            xr, yr = xr - xr.mean(), yr - yr.mean()
            null = np.sum(xr[perm] * yr[None, :], axis=1) / (np.linalg.norm(xr) * np.linalg.norm(yr))
            record["p"] = (int(np.sum(np.abs(null) >= abs(coefficient) - 1e-12)) + 1) / (permutations + 1)
            # Resampled duplicates change midranks; rank each bootstrap anew.
            bx = pd.DataFrame(x[indices]).rank(axis=1).to_numpy()
            by = pd.DataFrame(y[indices]).rank(axis=1).to_numpy()
            bx, by = bx - bx.mean(axis=1, keepdims=True), by - by.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.sum(bx*bx, axis=1) * np.sum(by*by, axis=1))
            draws = np.divide(np.sum(bx*by, axis=1), denominator,
                              out=np.full(bootstraps, np.nan), where=denominator > 0)
            finite = draws[np.isfinite(draws)]
            if len(finite):
                record["ciLow"], record["ciHigh"] = map(float, np.quantile(finite, [.025, .975]))
            loo = [rho(np.delete(x, i), np.delete(y, i)) for i in range(n)]
            loo = [r for r in loo if r is not None]
            record.update(bootstrapValid=len(finite), bootstrapUndefined=bootstraps-len(finite),
                          looMin=min(loo) if loo else None, looMax=max(loo) if loo else None)
    # Undefined tests count in the planned families, conservatively as p=1.
    for definition in ("target", "sampling"):
        group = [r for r in output if r["definition"] == definition]
        for row, p in zip(group, holm([r["p"] if r["p"] is not None else 1 for r in group])):
            if row["p"] is not None:
                row["holm9"] = p
    for row, p in zip(output, holm([r["p"] if r["p"] is not None else 1 for r in output])):
        if row["p"] is not None:
            row["holm18"] = p
    return output


def add_rank_analysis(payload, *, permutations=PERMUTATIONS, bootstraps=BOOTSTRAPS):
    """Called after importing/reusing arrays with matching input hashes."""
    S = payload["spearman"]
    # Keep the separate direct-target comparison available, explicitly labeled.
    old = copy.deepcopy(S)
    payload["targetScoreSpearman"] = old
    by_context = {}
    for prompt in payload["prompts"]:
        if prompt["frame"] == "Neutral":
            for context in prompt["contexts"]:
                if context in by_context:
                    raise ValueError("Multiple neutral prompts for context: " + context)
                by_context[context] = prompt
    words, results = [], []
    for context in payload["contexts"]:
        prompt = by_context[context]
        exported = {c["word"].strip().lower(): c for c in prompt["distribution"]}
        if len(exported) != len(prompt["distribution"]):
            raise ValueError("Duplicate exported sampling candidates: " + context)
        human = [r for r in old["word_paired_ranks"] if r["context"] == context]
        missing = [r["word"] for r in human if r["word"] not in exported]
        coefficient, status = None, "missing_sampling_scores"
        if not missing:
            values = [exported[r["word"]]["logp"] for r in human]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v > 0 for v in values):
                raise ValueError("Invalid sampling scores for " + context)
            ranks = pd.Series(values).rank(ascending=False).to_numpy() - 1
            coefficient = rho([r["human_rank"] for r in human], ranks)
            status = "defined" if coefficient is not None else "constant_human_or_model"
            for row, value, rank in zip(human, values, ranks):
                candidate = exported[row["word"]]
                words.append({**row, "model_value": value, "model_rank": float(rank),
                              "sampling_probability": candidate["normalized"],
                              "vocabulary_rank": candidate["rank"], "prompt_id": prompt["id"],
                              "score_source": "sampling_array"})
        results.append(dict(context=context, model=payload["modelName"], n=len(human)-len(missing),
                            spearman_rho=coefficient, status=status, missing_words=missing,
                            score_source="sampling_array"))
    S["word_paired_ranks"], S["word_spearman_by_context"] = words, results
    valid = [r["spearman_rho"] for r in results if r["spearman_rho"] is not None]
    for row in S["mean_within_context_spearman"]:
        if row["measure"] == "word_ranking":
            row.update(mean_within_context_spearman=sum(valid)/len(valid) if valid else None,
                       valid_contexts=len(valid), total_contexts=len(results), score_source="sampling_array")
    def rounded(value):
        return None if value is None else round(value, 12)
    current = {r["context"]: r for r in results}
    target = {r["context"]: r for r in old["word_spearman_by_context"]}
    negation = {(r["context"], r["structure"], r["variant"]): r
                for r in S["negation_spearman_by_context"]}
    points = []
    for structure, variant, model in SPECS:
        for context in payload["contexts"]:
            neg = negation[context, structure, variant]
            points.append(dict(context=context, model=model, structure=structure, variant=variant,
                               sampling=rounded(current[context]["spearman_rho"]),
                               target=rounded(target[context]["spearman_rho"]),
                               negation=rounded(neg["spearman_rho"]),
                               wordStatus=current[context]["status"], negationStatus=neg["status"]))
    payload["rankAssociation"] = dict(points=points, tests=tests(points, permutations=permutations, bootstraps=bootstraps),
        seed=SEED, permutations=permutations, bootstraps=bootstraps, defaultDefinition="sampling",
        wordSource="Full-support candidate scores used to sample the saved linking structures",
        scope="Exploratory, conditional on saved estimates; context pairing permutations and paired context bootstrap. No participant, norming, sampling, or CV-refit uncertainty is propagated.")
    compared, mismatched, missing = 0, 0, 0
    for prompt in payload["prompts"]:
        if prompt["frame"] != "Neutral":
            continue
        exported = {c["word"].strip().lower(): c for c in prompt["distribution"]}
        for candidate in prompt["candidates"]:
            saved = exported.get(candidate["word"].strip().lower())
            if saved is None:
                missing += 1
                continue
            compared += 1
            mismatched += int(abs(saved["logp"] - candidate["logp"]) > 1e-8)
    payload["scoreAlignment"] = dict(compared=compared, mismatched=mismatched, missing=missing,
                                     matched=compared > 0 and mismatched == 0 and missing == 0)
    if payload.get("run", {}).get("corrected") and not payload["scoreAlignment"]["matched"]:
        raise ValueError("Corrected run requires matching direct-target and sampling scores for every neutral candidate")
