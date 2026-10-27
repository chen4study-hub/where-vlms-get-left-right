"""Block localization and dose curves (paper Section 4.4, Fig. 3): mirrored positions in only some layers.

Qwen2.5-VL on CV-Bench 2D. Dose: the first or last L ViT blocks see mirrored positions (the main
axis of each question: w for left-right, h for up-down), on the original image ('orig' base: how many layers
must be mirrored before the answer follows) or on the flipped image with positions mirrored back ('pix' base).
Blocks: left-right questions only, ViT width positions mirrored in single blocks or block sets.

  python -m experiments.blocks_dose run --model Qwen/Qwen2.5-VL-3B-Instruct --parquet data/cvbench/test_2d.parquet \\
      --out results/blocks_dose/dose_qwen2.5-vl-3b-instruct.jsonl   # add --blocks for the block scan
  python -m experiments.blocks_dose dose <dose.jsonl> [--whole-model <main-table predictions.jsonl>]
  python -m experiments.blocks_dose blocks <blocks.jsonl> ...
"""
import argparse
import collections
import time

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl
from vlmlr.stats import frac_ci, wilson


# ---------------------------------------------------------------- conditions

def dose_levels(n_layers, step):
    """L in {0, step, 2 step, ..., n_layers} (n_layers appended when it is not a multiple of step)."""
    ls = list(range(0, n_layers + 1, step))
    if ls[-1] != n_layers:
        ls.append(n_layers)
    return ls


def layer_set(n_layers, L, sched):
    """'prefix': the first L layers see mirrored positions; 'suffix': the last L layers."""
    if not 0 <= L <= n_layers:
        raise ValueError(f"L={L} outside [0, {n_layers}]")
    return frozenset(range(L)) if sched == "prefix" else frozenset(range(n_layers - L, n_layers))


def dose_conditions(axis, n_vit, step=4, bases=("orig", "pix")):
    """Conditions of one question (one forward pass each). L = 0 equals the base and is not run; suffix L = n
    equals prefix L = n and only the prefix is run."""
    ax = {"lr": "w", "ud": "h"}[axis]
    pix = {"lr": "hflip", "ud": "vflip"}[axis]
    tf = {"orig": None, "pix": pix}
    out = [dict(cond=f"base_{b}", base=b, transform=tf[b], side="", sched="", L=0, layers=frozenset(), axes="")
           for b in bases]
    for b in bases:
        for sched in ("prefix", "suffix"):
            for L in dose_levels(n_vit, step):
                if L == 0 or (sched == "suffix" and L == n_vit):
                    continue
                out.append(dict(cond=f"{b}_vit_{sched}{L}", base=b, transform=tf[b], side="vit", sched=sched,
                                L=L, layers=layer_set(n_vit, L, sched), axes=ax))
    return out


def block_conditions(n_vit, global_blocks=(7, 15, 23, 31), last=8):
    """Block scan: each of the last `last` blocks and each global-attention block alone, the four global blocks
    together, all but the last block, and all blocks (reference); original image, ViT width positions."""
    one = sorted(set(range(n_vit - last, n_vit)) | {g for g in global_blocks if g < n_vit})
    out = [dict(cond="base_orig", base="orig", transform=None, side="", sched="", L=0, layers=frozenset(), axes="")]
    sets = [(f"vit_blk{k}", frozenset([k])) for k in one]
    sets += [("vit_global", frozenset(g for g in global_blocks if g < n_vit)),
             ("vit_all_but_last", frozenset(range(n_vit - 1))), ("vit_all", frozenset(range(n_vit)))]
    for name, ls in sets:
        out.append(dict(cond=f"orig_{name}", base="orig", transform=None, side="vit", sched="set", L=len(ls),
                        layers=ls, axes="w"))
    return out


# ---------------------------------------------------------------- GPU run

def run(args):
    import torch
    from vlmlr import models
    from vlmlr.hooks import QwenLayerwiseMirror

    bases = tuple(b for b in args.bases.split(",") if b)
    model, proc = models.load_qwen(args.model)
    ids = models.letter_ids(proc.tokenizer, "AB")
    dose = QwenLayerwiseMirror(model)
    print(f"ViT {dose.n_vit} blocks, step={args.step}, bases={bases}", flush=True)
    checked, t0 = False, time.time()
    with ResumableWriter(args.out, key=("idx", "cond")) as out, torch.inference_mode():
        for it in C.load_items(args.parquet, args.limit):
            rel = C.gold_relation(it)
            axis = C.AXIS[rel]
            choices = C.CHOICES[axis]
            if args.blocks and axis != "lr":
                continue
            text = C.prompt(it["question"], choices, False)
            img0, cache = C.to_image(it["image"]), {}
            conds = (block_conditions(dose.n_vit) if args.blocks
                     else dose_conditions(axis, dose.n_vit, args.step, bases))
            for c in conds:
                if (it["idx"], c["cond"]) in out:
                    continue
                if c["transform"] not in cache:
                    cache[c["transform"]] = models.build_inputs(proc, C.transform(img0, c["transform"]), text)
                inputs = cache[c["transform"]]
                if not checked:
                    dose.self_check(model, inputs, log=lambda s: print(s, flush=True))
                    checked = True
                lg, hits = dose.forward_logits(model, inputs, axes=c["axes"], layers=c["layers"])
                assert not c["side"] or hits == len(c["layers"]), f"{c['cond']}: {hits} blocks replaced"
                lp = models.option_logprobs(lg, ids)
                out.write({"idx": it["idx"], "cond": c["cond"], "base": c["base"], "side": c["side"],
                           "sched": c["sched"], "L": c["L"], "axis": axis, "gold_orig": rel,
                           "pred": models.pick(choices, lp), "logp": lp, "choices": choices,
                           "n_vit": dose.n_vit, "source": it["source"]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- analyses

def _curve(by, base, side, sched, axis, n):
    """[(L, k changed, m items, carry)]; prefix includes L = 0 (= base) and L = n; suffix borrows L = n from prefix.

    carry = mean(sign(s0) (s0 - s)) / mean(|s0|) with s = logp_A - logp_B: 0 = no effect, 1 = pushed to 50/50 on
    average, 2 = pushed to an equally strong opposite answer."""
    out = []
    for L in sorted({r["L"] for d in by.values() for r in d.values()
                     if r["base"] == base and r["side"] == side} | {0}):
        cond = f"{base}_{side}_{sched}{L}" if not (sched == "suffix" and L == n) else f"{base}_{side}_prefix{L}"
        pairs = []
        for d in by.values():
            b = d.get(f"base_{base}")
            if b is None or b["axis"] != axis:
                continue
            c = b if L == 0 else d.get(cond)
            if c is not None:
                pairs.append((b, c))
        if not pairs:
            continue
        k = sum(b["pred"] != c["pred"] for b, c in pairs)
        s0 = [b["logp"][0] - b["logp"][1] for b, _ in pairs]
        s1 = [c["logp"][0] - c["logp"][1] for _, c in pairs]
        num = sum((1 if x > 0 else -1) * (x - y) for x, y in zip(s0, s1))
        den = sum(abs(x) for x in s0) or float("nan")
        out.append((L, k, len(pairs), num / den))
    return out


def dose(paths, whole_model=None):
    """Change rate and carry as a function of the number of mirrored layers; with --whole-model, the item-level
    agreement of L = all layers with the main-table whole-model mirror (should be about 100 %)."""
    for path in paths:
        rows = read_jsonl(path)
        by = collections.defaultdict(dict)
        for r in rows:
            by[r["idx"]][r["cond"]] = r
        n_vit = rows[0]["n_vit"]
        bases = sorted({r["base"] for r in rows})
        sides = [("vit", n_vit)]
        print(f"\n=== {path}  items {len(by)}  ViT {n_vit} blocks")
        print("   base orig: positions flipped only (lr→w, ud→h), how many layers until the answer flips; base pix: pixels "
              "flipped + positions flipped back, how many layers until the answer flips back")
        for base in bases:
            for side, n in sides:
                for sched in ("prefix", "suffix"):
                    for axis in ("lr", "ud"):
                        cv = _curve(by, base, side, sched, axis, n)
                        if not cv:
                            continue
                        print(f"-- {base} {side} {sched} {axis}")
                        for L, k, m, carry in cv:
                            print(f"   L={L:>2}  flip {frac_ci(k, m)}  carry {carry:+.2f}")
        if whole_model:
            main = collections.defaultdict(dict)
            for r in read_jsonl(whole_model):
                main[r["idx"]][r["variant"]] = r["pred"]
            print("-- agreement with the whole-model replacement (L = all; should be near 100%)")
            for base in bases:
                for side, n in sides:
                    agree = tot = 0
                    for idx, d in by.items():
                        c = d.get(f"{base}_{side}_prefix{n}")
                        if c is None:
                            continue
                        ax = "w" if c["axis"] == "lr" else "h"
                        pix = "hflip" if ax == "w" else "vflip"
                        v = f"{side}_{ax}" if base == "orig" else f"{pix}_{side}_{ax}"
                        if v in main.get(idx, {}):
                            tot += 1
                            agree += main[idx][v] == c["pred"]
                    print(f"   {base} {side}: {frac_ci(agree, tot)}")


def blocks(paths):
    """Left-right change rate of each block condition relative to base_orig."""
    for path in paths:
        by = collections.defaultdict(dict)
        for r in read_jsonl(path):
            by[r["idx"]][r["cond"]] = r
        conds = sorted({c for d in by.values() for c in d if c != "base_orig"},
                       key=lambda c: (not c.startswith("orig_vit_blk"), int(c[12:]) if c.startswith("orig_vit_blk") else 0, c))
        print(f"=== {path}  left-right items {len(by)}")
        for c in conds:
            pairs = [(d["base_orig"], d[c]) for d in by.values() if "base_orig" in d and c in d]
            k, n = sum(a["pred"] != b["pred"] for a, b in pairs), len(pairs)
            lo, hi = wilson(k, n)
            print(f"  {c:22s} {100 * k / n:5.1f}% [{100 * lo:.0f},{100 * hi:.0f}] n={n}")


R = "results/blocks_dose/"
OUTPUTS = [
    ("outputs/blocks_dose/dose_curves.txt", ["dose", R + "dose_qwen2.5-vl-3b-instruct.jsonl", "--whole-model",
                                             "results/main_table/qwen2.5-vl/qwen2.5-vl-3b-instruct.jsonl"], True),
    ("outputs/blocks_dose/block_scan.txt", ["blocks", R + "blocks_qwen2.5-vl-3b-instruct.jsonl",
                                            R + "blocks_qwen2.5-vl-7b-instruct.jsonl"], True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.blocks_dose", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: dose conditions (or --blocks) on CV-Bench, Qwen2.5-VL")
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True, help="CV-Bench test_2d.parquet")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="first N questions only")
    r.add_argument("--step", type=int, default=4)
    r.add_argument("--bases", default="orig,pix")
    r.add_argument("--blocks", action="store_true", help="block scan (left-right questions, ViT width, orig base)")
    d = sub.add_parser("dose", help=dose.__doc__.split("\n")[0])
    d.add_argument("files", nargs="+")
    d.add_argument("--whole-model", help="main-table predictions of the same model")
    sub.add_parser("blocks", help=blocks.__doc__.split("\n")[0]).add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "dose":
        dose(a.files, a.whole_model)
    else:
        blocks(a.files)


if __name__ == "__main__":
    main()
