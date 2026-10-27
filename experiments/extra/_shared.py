"""Helpers shared by the analyses on the CV-Bench validation split (tiles, video input).

vlmlr.cvbench.in_validation_split: 187 questions, 108 left-right, 79 above-below.
Paired metric: the gold margin m = log p(gold) - log p(other option); Δm = intervened - original, averaged over
items, with a paired bootstrap (2000 resamples, seed 0).
"""
from vlmlr.cvbench import in_validation_split
from vlmlr.stats import bootstrap_mean, percentile_ci


def validation_items(items):
    return [it for it in items if in_validation_split(it["idx"])]


def gold_margin(logp, choices, gold):
    g = choices.index(gold)
    return logp[g] - logp[1 - g]


def paired(recs, cond, base="orig"):
    """[(idx, m_base, m_cond, ok_base, ok_cond, pred_base, pred_cond)] for items that have both conditions."""
    by = {}
    for r in recs:
        by.setdefault(r["idx"], {})[r["cond"]] = r
    out = []
    for i, d in sorted(by.items()):
        if base in d and cond in d:
            b, c = d[base], d[cond]
            out.append((i, gold_margin(b["logp"], b["choices"], b["gold"]), gold_margin(c["logp"], c["choices"], c["gold"]),
                        b["pred"] == b["gold"], c["pred"] == c["gold"], b["pred"], c["pred"]))
    return out


def margin_ci(diffs, B=2000):
    return percentile_ci(bootstrap_mean(diffs, B=B, seed=0))


def load_cvbench_validation(parquet):
    from vlmlr import cvbench as C
    return validation_items(C.load_items(parquet))
