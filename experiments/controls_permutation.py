"""Permutation controls of the ViT position code (paper Sections 4.1, 4.2, 4.5 and 4.6; protocols in Appendix B).

Four parts, one forward pass per question and condition, pixels unchanged unless stated:
  1. Same-axis, non-mirror permutations of the ViT position code: a half-grid cyclic shift and random permutations
     of the merge units with three seeds (vlmlr.permutations), against the ViT mirror of the main runs;
     CV-Bench 2D Relation, What'sUp and the held-out half of the CV-Bench 3D depth questions.
  2. Depth: a real vertical flip (vflip) as reference for the height mirror.
  3. ViT width code removed (Qwen2.5-VL: constant width ids; InternVL3: absolute embedding averaged over the
     width), then the LLM side alone: width ids mirrored (neu_llm_w), token order mirrored with the tokens keeping
     their ids (neu_order_only_w), contents in mirrored order with the ids in place (neu_order_and_id_w).
  4. InternVL3-2B with the ViT code intact: order only / order and id, width and height.
Scoring as in the main runs: bf16 option-letter logits, log-softmax, a two-option tie goes to the first option
(an fp32 score from the lm_head input is stored as lp32); What'sUp: the largest of the four letter logits; depth:
fp32 scores as in experiments.depth_shortcut, an exact fp32 tie is recorded as no prediction. Every run re-runs
the original condition on every item and checks it against the main-run predictions (summary, first section).

GPU runs (one model and data set each; --limit N for the first N items per axis):
  python -m experiments.controls_permutation run --family qwen --data cv2d --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --parquet <dir>/test_2d.parquet --out results/permutation_controls/q7b_cv2d.jsonl
  python -m experiments.controls_permutation run --family qwen --data whatsup --model ... --whatsup <snapshot> \\
      --out results/permutation_controls/q7b_whatsup.jsonl
  python -m experiments.controls_permutation run --family qwen --data depth --model ... \\
      --items items_test.jsonl --out results/permutation_controls/q7b_depth.jsonl
      (items_test.jsonl: python -m experiments.depth_shortcut items-cvbench3d --split test)
  InternVL3-2B (--family internvl, OpenGVLab/InternVL3-2B-hf): cv2d and whatsup -> ivl2b_cv2d.jsonl, ivl2b_whatsup.jsonl
Analyses (CPU; files <model>_<data>.jsonl under --new, main-run predictions as in REF):
  python -m experiments.controls_permutation summary --new results/permutation_controls
  python -m experiments.controls_permutation posthoc-order --new results/permutation_controls
"""
import argparse
import json
import os
import random
import time

from vlmlr import permutations as PM
from vlmlr.records import read_jsonl

PERMS, SHUF = PM.PERMS, PM.SHUF
MODELS = ("q3b", "q7b", "ivl2b")
REPRO_MIN = 1.0       # reproduction check: a group is evaluated if its orig run matches the main run on every item

# main-run predictions each control is compared with (orig, mirrors, real flips)
REF = {
    ("q3b", "cv2d"): "results/main_table/qwen2.5-vl/qwen2.5-vl-3b-instruct.jsonl",
    ("q7b", "cv2d"): "results/main_table/qwen2.5-vl/qwen2.5-vl-7b-instruct.jsonl",
    ("ivl2b", "cv2d"): "results/main_table/internvl3-2b/internvl3-2b-hf.jsonl",
    ("q3b", "whatsup"): "results/whatsup/qwen2.5-vl_internvl3-2b/qwen25vl3b.jsonl",
    ("q7b", "whatsup"): "results/whatsup/qwen2.5-vl_internvl3-2b/qwen25vl7b.jsonl",
    ("ivl2b", "whatsup"): "results/whatsup/qwen2.5-vl_internvl3-2b/internvl32b.jsonl",
    ("q3b", "depth"): "results/depth_shortcut/cvbench3d_test/test_qwen2.5-vl-3b-instruct.jsonl",
    ("q7b", "depth"): "results/depth_shortcut/cvbench3d_test/test_qwen2.5-vl-7b-instruct.jsonl",
}

# ---------------------------------------------------------------- conditions
# tf: pixel transform; vh / vw: kind of the ViT height / width position code (vlmlr.permutations); emb / pos: LLM-side
# permutation of the visual tokens' contents / position ids (slot j holds the content of token sigma(j) and the
# position of token tau(j); pos alone = the LLM-side mirror llm_w of the main table); axes: which items run it.
NEU = "neu"           # width code removed: 'const' for Qwen2.5-VL, 'mean' for InternVL3 (resolve)


def _c(tf=None, vh="", vw="", emb="", pos="", axes=("lr", "ud")):
    return dict(tf=tf, vh=vh, vw=vw, emb=emb, pos=pos, axes=axes)


COND = {"orig": _c()}
for _k in PERMS:
    COND[f"vit_w_{_k}"] = _c(vw=_k)
    COND[f"vit_h_{_k}"] = _c(vh=_k)
COND.update({
    "vflip": _c(tf="vflip"),
    "neu": _c(vw=NEU),                                         # all items: up-down items show how far out of distribution
    "neu_hflip": _c(tf="hflip", vw=NEU, axes=("lr",)),
    "neu_llm_w": _c(vw=NEU, pos="w", axes=("lr",)),
    "neu_order_only_w": _c(vw=NEU, emb="w", pos="w", axes=("lr",)),
    "neu_order_and_id_w": _c(vw=NEU, emb="w", axes=("lr",)),
    "order_only_w": _c(emb="w", pos="w"), "order_only_h": _c(emb="h", pos="h"),
    "order_and_id_w": _c(emb="w"), "order_and_id_h": _c(emb="h"),
    # re-run on the first items only (--repro) and compared with the main runs
    "vit_w_mirror": _c(vw="mirror"), "vit_h_mirror": _c(vh="mirror"), "llm_w": _c(pos="w"),
})
REPRO = {"orig": "orig", "vit_w_mirror": "vit_w", "vit_h_mirror": "vit_h", "llm_w": "llm_w"}   # this run -> main run

PERM_CONDS = [f"vit_{a}_{k}" for a in "wh" for k in PERMS]
NEU_CONDS = ["neu", "neu_hflip", "neu_llm_w", "neu_order_only_w", "neu_order_and_id_w"]
ORDER_CONDS = ["order_only_w", "order_only_h", "order_and_id_w", "order_and_id_h"]
PLAN = {
    ("qwen", "cv2d"): ["orig"] + PERM_CONDS + NEU_CONDS,
    ("qwen", "whatsup"): ["orig"] + PERM_CONDS,
    ("qwen", "depth"): ["orig"] + PERM_CONDS + ["vflip"],
    ("internvl", "cv2d"): ["orig"] + PERM_CONDS + NEU_CONDS + ORDER_CONDS,
    ("internvl", "whatsup"): ["orig"] + PERM_CONDS,
}
REPRO_EXTRA = {
    ("qwen", "cv2d"): ["vit_w_mirror", "vit_h_mirror"], ("qwen", "whatsup"): ["vit_w_mirror"],
    ("qwen", "depth"): [], ("internvl", "cv2d"): ["vit_w_mirror", "llm_w"], ("internvl", "whatsup"): ["vit_w_mirror"],
}


def resolve(name, family):
    c = dict(COND[name])
    for k in ("vh", "vw"):
        if c[k] == NEU:
            c[k] = "const" if family == "qwen" else "mean"
    return c


def conds_for(family, data, axis, repro=False):
    names = PLAN[(family, data)] + (REPRO_EXTRA[(family, data)] if repro else [])
    return [n for n in names if axis in COND[n]["axes"] or data == "depth"]


def must_change(name, family):
    """Whether the self-check requires the condition to change the first item's output. Under the 2D-RoPE of Qwen
    a cyclic shift changes only the relative positions across the seam, and the first item's bf16 logits can stay
    the same; for those only the position ids must change."""
    c = COND[name]
    return not (family == "qwen" and "shift" in (c["vh"], c["vw"]))


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    import math
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, (max(0.0, c - h), min(1.0, c + h))


def boot(stat, ids, B=10000, seed=0):
    """stat(all items) and the 2.5 / 97.5 percentiles over B resamples of the sorted item ids (seed 0)."""
    ids = sorted(ids)
    rng = random.Random(seed)
    n = len(ids)
    vals = sorted(stat([ids[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return stat(ids), (vals[int(0.025 * B)], vals[int(0.975 * B) - 1])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- reading the predictions
INV = {"left": "right", "right": "left", "above": "below", "below": "above", "on": "under", "under": "on"}


def index(recs):
    """{idx: {condition: record}}; the main runs use 'variant', this run and the depth runs 'cond'."""
    by = {}
    for r in recs:
        by.setdefault(r["idx"], {})[r.get("cond") or r.get("variant")] = r
    return by


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"per-item predictions not found: {path} (run the experiment first, see README)")
    return index(read_jsonl(path))


def depth_pred(r):
    """Depth prediction (None for an exact fp32 tie); records of the depth runs have only lp."""
    if "pred" in r:
        return r["pred"]
    lp = r["lp"]
    return None if lp[0] == lp[1] else int(lp[1] > lp[0])


def geo_subset(ref, axis):
    """Geometry-consistent items: correct on orig and on the matching real flip (lr: hflip, ud: vflip) in the main run."""
    pix = "hflip" if axis == "lr" else "vflip"
    return sorted(i for i, d in ref.items() if "orig" in d and pix in d and d["orig"]["axis"] == axis
                  and d["orig"]["pred"] == d["orig"]["expected"] and d[pix]["pred"] == d[pix]["expected"])


def repro(new, ref, pairs=None, depth=False):
    """Conditions re-run here against the main run: items with the same prediction, largest log-prob difference."""
    pairs = pairs or REPRO
    out = {}
    for mine, theirs in pairs.items():
        ks = [i for i in new if mine in new[i] and theirs in ref.get(i, {})]
        if not ks:
            continue
        pn = depth_pred if depth else (lambda r: r["pred"])
        same = sum(pn(new[i][mine]) == pn(ref[i][theirs]) for i in ks)
        lpk = "lp" if depth else ("logp" if "logp" in ref[ks[0]][theirs] else "logits")
        mk = "lp32" if depth else ("lp" if lpk == "logp" else "logits")
        dmax = max(max(abs(x - y) for x, y in zip(new[i][mine][mk], ref[i][theirs][lpk])) for i in ks)
        out[mine] = {"n": len(ks), "same": same, "max_diff": dmax}
    return out


# ---------------------------------------------------------------- part 1: same-axis permutations (CV-Bench, What'sUp)


def perm_compare(new, ref, axis, B=10000):
    """Geometry-consistent items: flip rate (relative to the main-run orig) of the mirror and of the same-axis
    permutations (shift; shuf averaged over the three seeds per item). D = mirror - max(shift, shuf), paired
    bootstrap. Also: split by the gold label, off-axis permutations, share of changes to the mirror relation, ties removed."""
    a = "w" if axis == "lr" else "h"
    off = "h" if a == "w" else "w"
    G = [i for i in geo_subset(ref, axis) if i in new and all(f"vit_{a}_{k}" in new[i] for k in PERMS)]
    o = {i: ref[i]["orig"]["pred"] for i in G}
    mir = {i: float(ref[i][f"vit_{a}"]["pred"] != o[i]) for i in G}
    sh = {i: float(new[i][f"vit_{a}_shift"]["pred"] != o[i]) for i in G}
    su = {i: mean(new[i][f"vit_{a}_{k}"]["pred"] != o[i] for k in SHUF) for i in G}

    def D(ids):
        return mean(mir[i] for i in ids) - max(mean(sh[i] for i in ids), mean(su[i] for i in ids))

    res = {"n": len(G), "mirror": mean(mir.values()), "shift": mean(sh.values()), "shuf": mean(su.values())}
    res["ctrl"] = max(res["shift"], res["shuf"])
    res["D"], res["D_ci"] = boot(D, G, B) if G else (float("nan"), (float("nan"), float("nan")))
    res["by_gold"] = {}
    for gl in sorted({ref[i]["orig"]["expected"] for i in G}):
        ids = [i for i in G if ref[i]["orig"]["expected"] == gl]
        res["by_gold"][gl] = {"n": len(ids), "mirror": mean(mir[i] for i in ids),
                              "ctrl": max(mean(sh[i] for i in ids), mean(su[i] for i in ids))}
    res["offaxis"] = {k: mean(new[i][f"vit_{off}_{k}"]["pred"] != o[i] for i in G if f"vit_{off}_{k}" in new[i])
                      for k in PERMS}

    def mrel(get):             # among changed answers: share changed to the mirror relation (informative on What'sUp)
        ch = [i for i in G if get(i) != o[i]]
        return (mean(get(i) == INV.get(o[i]) for i in ch), len(ch))
    res["mirror_rel"] = {"mirror": mrel(lambda i: ref[i][f"vit_{a}"]["pred"]),
                         "shift": mrel(lambda i: new[i][f"vit_{a}_shift"]["pred"])}
    for k in SHUF:
        res["mirror_rel"][k] = mrel(lambda i, k=k: new[i][f"vit_{a}_{k}"]["pred"])

    def tie(r):
        lp = r.get("logp") or r.get("lp")
        return bool(lp and len(lp) == 2 and lp[0] == lp[1])
    G2 = [i for i in G if not (tie(ref[i]["orig"]) or tie(ref[i][f"vit_{a}"])
                               or any(tie(new[i][f"vit_{a}_{k}"]) for k in PERMS))]
    res["D_notie"] = (D(G2) if G2 else float("nan"), len(G2))
    return res


# ---------------------------------------------------------------- parts 1 and 2: depth (held-out half)


def depth_compare(new, ref, B=10000):
    """All held-out items: change rates relative to the main-run orig (items with a tie on either side are left
    out), share of changes that choose the higher object as closer, agreement with the real vertical flip."""
    ids = sorted(i for i in ref if "orig" in ref[i] and i in new)
    o = {i: depth_pred(ref[i]["orig"]) for i in ids}
    P = {i: {c: depth_pred(r) for c, r in new[i].items()} for i in ids}
    for i in ids:
        for c in ("vit_h", "vit_w"):
            if c in ref[i]:
                P[i][c] = depth_pred(ref[i][c])
    hi = {i: ref[i]["orig"]["higher"] for i in ids}
    nonamb = set(i for i in ids if ref[i]["orig"]["group"] != "ambiguous")
    perm = {a: [f"vit_{a}_{k}" for k in PERMS] for a in "hw"}

    def valid(i, cs):
        return o[i] is not None and all(P[i].get(c) is not None for c in cs)

    def chg(i, c):
        return float(P[i][c] != o[i])

    def share(I, conds):
        ch = [(i, c) for i in I for c in conds if P[i][c] != o[i]]
        return (mean(P[i][c] == hi[i] for i, c in ch) if ch else float("nan")), len(ch)

    out = {"n": len(ids)}
    for a in "hw":
        V = [i for i in ids if valid(i, [f"vit_{a}"] + perm[a])]
        mir = lambda I, a=a: mean(chg(i, f"vit_{a}") for i in I)
        sh = lambda I, a=a: mean(chg(i, f"vit_{a}_shift") for i in I)
        su = lambda I, a=a: mean(mean(chg(i, f"vit_{a}_{k}") for k in SHUF) for i in I)
        D = lambda I, mir=mir, sh=sh, su=su: mir(I) - max(sh(I), su(I))
        d, ci = boot(D, V, B) if V else (float("nan"), (float("nan"), float("nan")))
        Vn = [i for i in V if i in nonamb]

        def dshare(I, a=a):
            s1, n1 = share(I, [f"vit_{a}"])
            s2, n2 = share(I, perm[a])
            return (s1 - s2) if n1 and n2 else float("nan")

        ds = boot(lambda I: (lambda v: v if v == v else 0.0)(dshare(I)), Vn, B) if Vn else (float("nan"), (float("nan"),) * 2)
        out[a] = {"n": len(V), "mirror": mir(V), "shift": sh(V), "shuf": su(V), "ctrl": max(sh(V), su(V)), "D": d,
                  "D_ci": ci, "dir_mirror": share(Vn, [f"vit_{a}"]), "dir_perm": share(Vn, perm[a]),
                  "dir_diff": dshare(Vn), "dir_diff_ci": ds[1]}
    Vv = [i for i in ids if valid(i, ["vflip", "vit_h"] + perm["h"])]

    def agree(I, conds):
        ch = [(i, c) for i in I for c in conds if P[i][c] != o[i]]
        return (mean(P[i]["vflip"] == P[i][c] for i, c in ch) if ch else float("nan")), len(ch)

    def adiff(I):
        x, n1 = agree(I, ["vit_h"])
        y, n2 = agree(I, perm["h"])
        return (x - y) if n1 and n2 else 0.0

    out["vflip"] = {"n": len(Vv), "change": mean(chg(i, "vflip") for i in Vv),
                    "dir": share([i for i in Vv if i in nonamb], ["vflip"]),
                    "agree_vit_h": agree(Vv, ["vit_h"]), "agree_perm_h": agree(Vv, perm["h"])}
    out["vflip"]["agree_diff"], out["vflip"]["agree_diff_ci"] = boot(adiff, Vv, B) if Vv else (float("nan"), (float("nan"),) * 2)
    return out


# ---------------------------------------------------------------- part 3: ViT width code removed
LLM_SIDE = ("neu_llm_w", "neu_order_only_w", "neu_order_and_id_w")


def neu_compare(new, ref):
    def acc(ids, cond, src):
        return mean(src[i][cond]["pred"] == ref[i]["orig"]["expected"] for i in ids)
    lr = sorted(i for i in new if "neu" in new[i] and new[i]["neu"]["axis"] == "lr" and i in ref)
    ud = sorted(i for i in new if "neu" in new[i] and new[i]["neu"]["axis"] == "ud" and i in ref)
    G = [i for i in lr if all(c in new[i] for c in ("neu_hflip",) + LLM_SIDE)
         and new[i]["neu"]["pred"] == ref[i]["orig"]["expected"]
         and new[i]["neu_hflip"]["pred"] == INV[ref[i]["orig"]["expected"]]]
    out = {"n_lr": len(lr), "n_ud": len(ud), "n_G": len(G),
           "acc_orig_lr": acc(lr, "orig", ref), "acc_neu_lr": acc(lr, "neu", new),
           "acc_orig_ud": acc(ud, "orig", ref), "acc_neu_ud": acc(ud, "neu", new),
           "acc_neu_hflip_lr": mean(new[i]["neu_hflip"]["pred"] == INV[ref[i]["orig"]["expected"]]
                                    for i in lr if "neu_hflip" in new[i])}
    k = sum(new[i]["neu"]["pred"] == ref[i]["orig"]["expected"] for i in lr)
    out["acc_neu_lr_ci"] = wilson(k, len(lr))[1]
    out["flip"] = {}
    for c in LLM_SIDE:
        kk = sum(new[i][c]["pred"] != new[i]["neu"]["pred"] for i in G)
        p, ci = wilson(kk, len(G))
        out["flip"][c] = {"k": kk, "n": len(G), "rate": p, "ci": ci}
    return out


# ---------------------------------------------------------------- part 4: InternVL3-2B order only
ORDER_NULL, ORDER_CARRY = 0.05, 0.15


def order_compare(new, ref):
    out = {}
    for c, axis in (("order_only_w", "lr"), ("order_and_id_w", "lr"), ("order_only_h", "ud"), ("order_and_id_h", "ud")):
        G = [i for i in geo_subset(ref, axis) if i in new and c in new[i]]
        k = sum(new[i][c]["pred"] != ref[i]["orig"]["pred"] for i in G)
        p, ci = wilson(k, len(G))
        out[c] = {"k": k, "n": len(G), "rate": p, "ci": ci}
    return out


# ---------------------------------------------------------------- analyses

def _pct(x):
    return "  nan" if x != x else f"{100 * x:5.1f}"


def _ci(c):
    return f"[{_pct(c[0]).strip()}, {_pct(c[1]).strip()}]"


RUNS = {(m, d) for m in MODELS for d in ("cv2d", "whatsup")} | {("q3b", "depth"), ("q7b", "depth")}


def _pair(new_dir, m, d):
    """(this run, main run) for a model and data set; ({}, {}) for a combination that was not run."""
    if (m, d) not in RUNS:
        return {}, {}
    return load(os.path.join(new_dir, f"{m}_{d}.jsonl")), load(REF[(m, d)])


def summary(a):
    L, R = [], {}
    L.append("== Reproduction: conditions re-run here compared row by row with the main-run predictions (orig re-run on "
             "every item; the mirrors on the first 8 items only)")
    bad = set()
    for m in MODELS:
        for d in ("cv2d", "whatsup", "depth"):
            new, ref = _pair(a.new, m, d)
            if not new:
                continue
            rp = repro(new, ref, depth=(d == "depth"))
            R.setdefault("repro", {})[f"{m}_{d}"] = rp
            for k, v in rp.items():
                ok = v["same"] / v["n"] >= REPRO_MIN
                bad |= set() if ok else {(m, d)}
                L.append(f"  {m:6s} {d:8s} {k:13s} same {v['same']}/{v['n']}  max diff {v['max_diff']:.4f}"
                         + ("" if ok else "  <- not reproduced on every item, group skipped"))

    L.append("\n== Same-axis non-mirror permutations on the ViT side (geometry-consistent subset; flip = change relative to "
             "the main-run orig; permutation = max(shift, mean of the three shuf seeds))")
    for d in ("cv2d", "whatsup"):
        for m in MODELS:
            new, ref = _pair(a.new, m, d)
            if not new or (m, d) in bad:
                continue
            for axis in ("lr", "ud"):
                r = perm_compare(new, ref, axis, a.B)
                if not r["n"]:
                    continue
                R.setdefault("perm", {})[f"{m}_{d}_{axis}"] = r
                L.append(f"  {d:7s} {m:6s} {axis}  n={r['n']:3d}  mirror {_pct(r['mirror'])}  shift {_pct(r['shift'])}  "
                         f"shuf {_pct(r['shuf'])}  diff {_pct(r['D'])} {_ci(r['D_ci'])}")
                L.append("      by gold: " + "; ".join(f"{g} n={x['n']} mirror {_pct(x['mirror'])} permutation {_pct(x['ctrl'])}"
                                                   for g, x in r["by_gold"].items())
                         + "; off-axis permutations " + " ".join(f"{k} {_pct(x)}" for k, x in r["offaxis"].items())
                         + f"; diff without ties {_pct(r['D_notie'][0])} (n={r['D_notie'][1]})")
                if d == "whatsup":
                    L.append("      share of changed items that change to the mirror relation: " + "  ".join(
                        f"{k} {_pct(x[0])} (changed {x[1]})" for k, x in r["mirror_rel"].items()))

    L.append("\n== Depth, held-out half (change rate relative to the main-run orig, ties not counted; direction = share of "
             "the changed non-ambiguous items that now call the object higher in the image the closer one)")
    for m in ("q3b", "q7b"):
        new, ref = _pair(a.new, m, "depth")
        if not new or (m, "depth") in bad:
            continue
        r = depth_compare(new, ref, a.B)
        R.setdefault("depth", {})[m] = r
        for ax in "hw":
            x = r[ax]
            L.append(f"  {m:4s} vit_{ax} n={x['n']}  mirror {_pct(x['mirror'])}  shift {_pct(x['shift'])}  shuf {_pct(x['shuf'])}  "
                     f"diff {_pct(x['D'])} {_ci(x['D_ci'])}  direction: mirror {_pct(x['dir_mirror'][0])} (changed {x['dir_mirror'][1]})  "
                     f"permutation {_pct(x['dir_perm'][0])} (changed rows {x['dir_perm'][1]})  diff {_pct(x['dir_diff'])} {_ci(x['dir_diff_ci'])}")
        y = r["vflip"]
        L.append(f"       vflip n={y['n']}  change rate {_pct(y['change'])}  direction {_pct(y['dir'][0])} (changed {y['dir'][1]})  "
                 f"same as the new vflip answer: vit_h {_pct(y['agree_vit_h'][0])} (changed {y['agree_vit_h'][1]})  "
                 f"height permutations {_pct(y['agree_perm_h'][0])} (changed rows {y['agree_perm_h'][1]})  diff {_pct(y['agree_diff'])} {_ci(y['agree_diff_ci'])}")

    L.append("\n== ViT width position code neutralized (Qwen constant ids, InternVL averaged over the width): does the LLM "
             "side alone carry left-right?")
    for m in MODELS:
        new, ref = _pair(a.new, m, "cv2d")
        if not new or (m, "cv2d") in bad or not any("neu" in x for x in new.values()):
            continue
        r = neu_compare(new, ref)
        R.setdefault("neu", {})[m] = r
        L.append(f"  {m:6s} left-right accuracy orig {_pct(r['acc_orig_lr'])} → neutralized {_pct(r['acc_neu_lr'])} {_ci(r['acc_neu_lr_ci'])}"
                 f" (neutralized + hflip {_pct(r['acc_neu_hflip_lr'])}); up-down orig {_pct(r['acc_orig_ud'])} → neutralized {_pct(r['acc_neu_ud'])}; "
                 f"geometry-consistent left-right items after neutralizing {r['n_G']}")
        L.append("      flips (relative to neutralized): " + "  ".join(f"{c} {x['k']}/{x['n']} {_pct(x['rate'])} {_ci(x['ci'])}"
                                          for c, x in r["flip"].items()))

    L.append("\n== InternVL3-2B order-only (ViT position code intact; geometry-consistent subset, relative to the main-run orig)")
    new, ref = _pair(a.new, "ivl2b", "cv2d")
    if new and ("ivl2b", "cv2d") not in bad and any("order_only_w" in x for x in new.values()):
        r = order_compare(new, ref)
        R["order"] = r
        L.append("  " + "  ".join(f"{c} {x['k']}/{x['n']} {_pct(x['rate'])} {_ci(x['ci'])}" for c, x in r.items()))

    print("\n".join(L))
    if a.json:
        json.dump(R, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


def posthoc_order(a):
    """Post hoc, without object boxes: does the left-right answer follow the order given by the position code?
    A. A random unit permutation reverses any pair of objects in different units with probability 1/2, so if the
       answer follows the order only, the shuf flip rate is about half the mirror's: ratio shuf / mirror (bootstrap).
    B. Whether the items changed by the shift are those changed by the mirror, and changed to the same answer."""
    for d in ("cv2d", "whatsup"):
        for m in MODELS:
            new, ref = _pair(a.new, m, d)
            if not new:
                continue
            axis, ax = "lr", "w"                 # left-right items, width permutations
            G = [i for i in geo_subset(ref, axis) if i in new]
            o = {i: ref[i]["orig"]["pred"] for i in G}
            mir = {i: ref[i][f"vit_{ax}"]["pred"] for i in G}
            sh = {i: new[i][f"vit_{ax}_shift"]["pred"] for i in G}
            su = {k: {i: new[i][f"vit_{ax}_{k}"]["pred"] for i in G} for k in SHUF}
            fm = {i: float(mir[i] != o[i]) for i in G}
            fs = {i: mean(su[k][i] != o[i] for k in SHUF) for i in G}
            ratio = lambda I: (mean(fs[i] for i in I) / mean(fm[i] for i in I)) if mean(fm[i] for i in I) else float("nan")
            r, ci = boot(ratio, G, a.B)
            chg = [i for i in G if sh[i] != o[i]]
            same_ans = mean(sh[i] == mir[i] for i in chg) if chg else float("nan")
            in_mir = mean(fm[i] for i in chg) if chg else float("nan")
            print(f"{d:7s} {m:6s} {axis} n={len(G):3d}  A. shuf/mirror = {r:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] (≈ 0.5 if only the order counts)  "
                  f"B. shift changes {len(chg)} items: also changed by the mirror {100 * in_mir:.0f}%, to the mirror's answer {100 * same_ans:.0f}%")


# ---------------------------------------------------------------- GPU run: hooks

class QwenCtrl:
    """Qwen2.5-VL: wraps visual.rot_pos_emb (ViT (h, w) ids, vlmlr.permutations.qwen_vit_pos_ids),
    model.get_image_features (LLM-side contents) and model.get_rope_index (LLM-side M-RoPE positions).
    With an empty setting every wrapper returns the original result."""

    def __init__(self, model):
        self.inner, self.visual = model.model, model.model.visual
        self.tok = model.config.image_token_id
        self.merge = self.visual.spatial_merge_size
        self.cfg = resolve("orig", "qwen")
        self.hits = {"vit": 0, "emb": 0, "pos": 0}
        self._vit0, self.visual.rot_pos_emb = self.visual.rot_pos_emb, self._vit
        self._feat0, self.inner.get_image_features = self.inner.get_image_features, self._feat
        self._rope0, self.inner.get_rope_index = self.inner.get_rope_index, self._rope

    def set(self, cfg):
        self.cfg = cfg
        self.hits = {"vit": 0, "emb": 0, "pos": 0}

    def _vit(self, grid_thw):
        if not (self.cfg["vh"] or self.cfg["vw"]):
            return self._vit0(grid_thw)
        self.hits["vit"] += 1
        pos = PM.qwen_vit_pos_ids(grid_thw.cpu(), self.merge, self.cfg["vh"], self.cfg["vw"]).to(grid_thw.device)
        return self.visual.rotary_pos_emb(grid_thw[:, 1:].max())[pos].flatten(1)

    def _feat(self, pixel_values, image_grid_thw=None):
        from vlmlr import positions as P
        out = self._feat0(pixel_values, image_grid_thw)
        if self.cfg["emb"]:
            self.hits["emb"] += 1
            out = P.permute_embeds(out, P.image_permutations(image_grid_thw, self.merge, self.cfg["emb"]))
        return out

    def _rope(self, input_ids=None, image_grid_thw=None, *a, **k):
        from vlmlr import positions as P
        pos, deltas = self._rope0(input_ids, image_grid_thw, *a, **k)
        if self.cfg["pos"] and input_ids is not None:
            self.hits["pos"] += 1
            pos = P.permute_positions(pos, input_ids, self.tok,
                                      P.image_permutations(image_grid_thw, self.merge, self.cfg["pos"]))
        return pos, deltas


class InternCtrl:
    """InternVL3 (single tile): wraps vision_tower.embeddings.interpolate_pos_encoding (absolute embedding,
    vlmlr.permutations.abs_pe_remap) and model.get_image_features (the 256 tokens of the tile); LLM-side positions
    are passed explicitly (vlmlr.positions.internvl_llm_positions)."""

    def __init__(self, model):
        from vlmlr import positions as P
        self.P = P
        self.emb_mod = model.model.vision_tower.embeddings
        if self.emb_mod.position_embeddings is None:
            raise RuntimeError("InternViT has no absolute position embedding")
        self.tok = model.config.image_token_id
        g = self.emb_mod.patch_embeddings.patch_shape[0]
        self.coords = P.pixel_shuffle_coords(model.model.pixel_shuffle, g, model.config.downsample_ratio)
        self.perms = {ax: P.mirror_permutation(self.coords, ax) for ax in ("w", "h")}
        self.cfg = resolve("orig", "internvl")
        self.hits = {"vit": 0, "emb": 0}
        self._pe0, self.emb_mod.interpolate_pos_encoding = self.emb_mod.interpolate_pos_encoding, self._pe
        self._feat0, model.model.get_image_features = model.model.get_image_features, self._feat

    def set(self, cfg):
        self.cfg = cfg
        self.hits = {"vit": 0, "emb": 0}

    def _pe(self, embeddings, height, width):
        pe = self._pe0(embeddings, height, width)
        self.hits["vit"] += 1
        return PM.abs_pe_remap(pe, self.cfg["vh"], self.cfg["vw"]) if (self.cfg["vh"] or self.cfg["vw"]) else pe

    def _feat(self, *a, **k):
        out = self._feat0(*a, **k)
        if self.cfg["emb"]:
            self.hits["emb"] += 1
            out = out[:, self.perms[self.cfg["emb"]].to(out.device)]
        return out

    def positions(self, input_ids):
        return self.P.internvl_llm_positions(input_ids, self.tok, self.perms[self.cfg["pos"]] if self.cfg["pos"] else None)


# ---------------------------------------------------------------- GPU run: data

def load_items(data, a):
    """Items with idx, axis ('depth' for depth), gold, prompt, choice fields and image source."""
    from vlmlr import cvbench as C
    if data == "cv2d":
        out = []
        for it in C.load_items(a.parquet):
            rel = C.gold_relation(it)
            ax = C.AXIS[rel]
            ch = C.CHOICES[ax]
            out.append({"idx": it["idx"], "axis": ax, "gold": rel, "choices": ch,
                        "prompt": C.prompt(it["question"], ch, False), "image": it["image"]})
        return out
    if data == "whatsup":
        from vlmlr import whatsup as W
        return [{"idx": it["idx"], "axis": it["axis"], "gold": it["gold"], "rels": it["rels"],
                 "prompt": W.prompt(it["options"]), "path": it["path"]} for it in W.load_items(a.whatsup)]
    out = []
    for x in read_jsonl(a.items):
        if x["task"] != "Depth":
            continue
        out.append({"idx": x["idx"], "axis": "depth", "gold": x["gold"], "group": x["group"], "higher": x["higher"],
                    "choices": x["choices"], "prompt": C.prompt(x["question"], x["choices"], False),
                    "path": os.path.join(a.image_root, x["img"]), "W": x["W"], "H": x["H"]})
    return out


def pick(items, limit):
    """--limit: the first limit // (number of axes) items of each axis (by item index); 0 = all."""
    items = sorted(items, key=lambda x: x["idx"])
    if not limit:
        return items
    axes = sorted({x["axis"] for x in items})
    k = max(1, limit // len(axes))
    sel = []
    for ax in axes:
        sel += [x for x in items if x["axis"] == ax][:k]
    return sorted(sel, key=lambda x: x["idx"])


def image_of(it):
    from PIL import Image
    from vlmlr import cvbench as C
    if "image" in it:
        return C.to_image(it["image"])
    return Image.open(it["path"]).convert("RGB")


def run(a):
    import torch
    from vlmlr import models
    from vlmlr import positions as P
    from vlmlr.cvbench import transform
    if (a.family, a.data) not in PLAN:
        raise SystemExit(f"no plan for {a.family} x {a.data}")
    items = pick(load_items(a.data, a), a.limit)
    print(f"[items] {a.data}: {len(items)} items " + str({ax: sum(x['axis'] == ax for x in items)
                                                      for ax in sorted({x['axis'] for x in items})}), flush=True)
    if a.family == "qwen":
        kw = dict(min_pixels=a.min_pixels) if a.data == "depth" else {}
        model, proc = models.load_qwen(a.model, max_pixels=a.max_pixels, **kw)
        H = QwenCtrl(model)
    else:
        model, proc = models.load_internvl(a.model)
        H = InternCtrl(model)
    n_opt = 4 if a.data == "whatsup" else 2
    letters = models.letter_ids(proc.tokenizer, "ABCD"[:n_opt])
    fp32 = models.FP32OptionScorer(model, letters)

    def build(img, it):
        inp = models.build_inputs(proc, img, it["prompt"], a.family)
        if a.data == "depth":
            gh, gw = inp["image_grid_thw"][0, 1:].tolist()
            assert (gw * 14, gh * 14) == img.size, f"the processor resized the cached image ({img.size} vs {gh}x{gw})"
        return inp

    def fwd(inp, cfg):
        H.set(cfg)
        try:
            if a.family == "qwen":
                model.model.rope_deltas = None
                lg = model(**inp).logits[0, -1].float()
            else:
                lg = model(**inp, position_ids=H.positions(inp["input_ids"])).logits[0, -1].float()
            hits = dict(H.hits)
        finally:
            H.set(resolve("orig", a.family))
        return lg[letters], fp32(), hits

    def self_check(inp):
        o1, f1, _ = fwd(inp, resolve("orig", a.family))
        o2, f2, _ = fwd(inp, resolve("orig", a.family))
        assert torch.equal(o1, o2) and torch.equal(f1, f2), "orig is not reproducible"
        if a.family == "qwen":
            g = inp["image_grid_thw"]
            ref = H._vit0(g)
            mine = H.visual.rotary_pos_emb(g[:, 1:].max())[PM.qwen_vit_pos_ids(g.cpu(), H.merge).to(g.device)].flatten(1)
            assert torch.equal(ref, mine), "re-created ViT ids differ from the original"
            for ax in ("w", "h", "hw"):
                kw = {"h_kind": "mirror" if "h" in ax else "", "w_kind": "mirror" if "w" in ax else ""}
                assert torch.equal(PM.qwen_vit_pos_ids(g.cpu(), H.merge, **kw), P.vit_patch_ids(g.cpu(), H.merge, ax)), \
                    f"mirror({ax}) differs from positions.vit_patch_ids"
            o3, _, h3 = fwd(inp, dict(resolve("orig", "qwen"), emb="id", pos="id"))
            assert torch.equal(o3, o1) and h3["emb"] == 1 and h3["pos"] == 1, "identity permutation through the hooks differs"
        else:
            n_img = int((inp["input_ids"] == H.tok).sum())
            assert n_img == len(H.coords) and inp["pixel_values"].shape[0] == 1, f"not a single tile: {n_img}"
            pe = H._pe0(torch.zeros(1, 1 + H.emb_mod.patch_embeddings.num_patches, 1), *H.emb_mod.image_size)
            assert torch.equal(PM.abs_pe_remap(pe, "", "mirror"), P.mirror_abs_pos_embed(pe, "w"))
            assert torch.equal(PM.abs_pe_remap(pe, "mirror", ""), P.mirror_abs_pos_embed(pe, "h"))
        names = sorted({n for x in ("lr", "ud") for n in conds_for(a.family, a.data, x, True)} - {"orig"})
        same = []
        for n in names:
            cfg = resolve(n, a.family)
            if cfg["tf"]:
                continue
            o, _, h = fwd(inp, cfg)
            if cfg["vh"] or cfg["vw"]:
                assert h["vit"] >= 1, f"{n} did not reach the ViT hook"
            if not must_change(n, a.family):
                g = inp["image_grid_thw"]
                assert not torch.equal(PM.qwen_vit_pos_ids(g.cpu(), H.merge, cfg["vh"], cfg["vw"]),
                                       PM.qwen_vit_pos_ids(g.cpu(), H.merge)), f"{n}: position ids unchanged"
                if torch.equal(o, o1):
                    same.append(n)
                continue
            assert not torch.equal(o, o1), f"{n} changed nothing"
        print(f"[self-check] passed; cyclic-shift conditions that left the output of the first item unchanged: {same or 'none'}", flush=True)

    done = {(r["idx"], r["cond"]) for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    only = set(a.conds.split(",")) if a.conds else None
    t0, n, checked = time.time(), 0, False
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            names = [c for c in conds_for(a.family, a.data, it["axis"], a.repro)
                     if (it["idx"], c) not in done and (only is None or c in only)]
            if not names:
                continue
            img0, cache = image_of(it), {}
            for name in names:
                cfg = resolve(name, a.family)
                if cfg["tf"] not in cache:
                    cache[cfg["tf"]] = build(transform(img0, cfg["tf"]), it)
                if not checked:
                    self_check(cache.setdefault(None, build(img0, it)))
                    checked = True
                bl, fl, _ = fwd(cache[cfg["tf"]], cfg)
                lp = torch.log_softmax(bl, -1).tolist()
                lp32 = torch.log_softmax(fl, -1).tolist()
                rec = {"idx": it["idx"], "cond": name, "data": a.data, "axis": it["axis"], "gold": it["gold"],
                       "lp": lp, "lp32": lp32, "tie": lp[0] == lp[1] if n_opt == 2 else None}
                if a.data == "cv2d":
                    rec.update(choices=it["choices"], pred=it["choices"][int(lp[1] > lp[0])])
                elif a.data == "whatsup":
                    l4 = bl.tolist()
                    rec.update(rels=it["rels"], logits=l4, pred=it["rels"][max(range(4), key=lambda i: l4[i])])
                else:
                    rec.update(group=it["group"], higher=it["higher"], tie32=lp32[0] == lp32[1],
                               pred=None if lp32[0] == lp32[1] else int(lp32[1] > lp32[0]))
                f.write(json.dumps(rec) + "\n")
                n += 1
            f.flush()
    print(f"[run] done {n} forwards in {time.time() - t0:.0f}s -> {a.out}", flush=True)


# ---------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.controls_permutation", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="GPU run of one model on one data set")
    p.add_argument("--family", choices=["qwen", "internvl"], required=True)
    p.add_argument("--data", choices=["cv2d", "whatsup", "depth"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--repro", action="store_true", help="also re-run the mirrors of the main runs (REPRO_EXTRA)")
    p.add_argument("--conds", default="", help="only these conditions (comma-separated)")
    p.add_argument("--parquet", default="datasets/nyu-visionx__CV-Bench/test_2d.parquet")
    p.add_argument("--whatsup", default="datasets/amitakamath2__whatsup_vlms")
    p.add_argument("--items", default="items_test.jsonl", help="depth item table (experiments.depth_shortcut items-cvbench3d)")
    p.add_argument("--image-root", default="", help="directory the item table's image paths start from")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--min-pixels", type=int, default=56 * 56)
    for name, h in (("summary", "the four parts"), ("posthoc-order", "post hoc order analysis")):
        q = sub.add_parser(name, help=h)
        q.add_argument("--new", default="results/permutation_controls", help="directory with <model>_<data>.jsonl")
        q.add_argument("--B", type=int, default=10000)
        if name == "summary":
            q.add_argument("--json")
    a = ap.parse_args(argv)
    {"run": run, "summary": summary, "posthoc-order": posthoc_order}[a.cmd](a)


OUTPUTS = [
    ("outputs/permutation_controls/summary.txt", ["summary", "--new", "results/permutation_controls"], True),
    ("outputs/permutation_controls/posthoc_order.txt", ["posthoc-order", "--new", "results/permutation_controls"], True),
]

if __name__ == "__main__":
    main()
