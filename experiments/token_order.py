"""LLM-side token order versus position ids (paper Section 4.2, Table 3), Qwen2.5-VL on CV-Bench 2D.

Slot j of an image span normally holds token content e_j with position p_j. Two permutations act separately:
sigma chooses the content (e_sigma(j)) and tau the position (p_tau(j)) of each slot.
  order_only     sigma = tau = mirror: tokens change seats with their ids, so only the causal order changes
  id_only        sigma = identity, tau = mirror: the main-table llm_w / llm_h
  order_and_id   sigma = mirror, tau = identity: contents in mirrored order with the original ids
  colmajor       sigma = tau = column-major traversal
Row-order controls (--set row-order, up-down questions only) compare order_only_h with row shifts and row shuffles
of the same size.
Pixels are never changed; the ViT attends bidirectionally, so order only matters inside the LLM.

  python -m experiments.token_order run --model Qwen/Qwen2.5-VL-3B-Instruct --parquet data/cvbench/test_2d.parquet \\
      --out results/token_order/order_qwen2.5-vl-3b-instruct.jsonl [--set row-order]
  python -m experiments.token_order order <order.jsonl> ...
  python -m experiments.token_order row-order <row_order.jsonl> <main-table predictions.jsonl> [<row_order.jsonl> <main.jsonl> ...]
"""
import argparse
import collections
import time

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl
from vlmlr.stats import frac_ci, wilson

VARIANTS = [  # (variant, content permutation sigma, position permutation tau)
    ("orig", "", ""),
    ("order_only_w", "w", "w"), ("order_only_h", "h", "h"),
    ("id_only_w", "", "w"), ("id_only_h", "", "h"),
    ("order_and_id_w", "w", ""), ("order_and_id_h", "h", ""),
    ("colmajor", "col", "col"),
]
VARIANTS_ROW = [
    ("orig", "", ""), ("order_only_h", "h", "h"), ("order_rowshift", "rowshift", "rowshift"),
    ("order_rowshuffle0", "rowshuffle0", "rowshuffle0"), ("order_rowshuffle1", "rowshuffle1", "rowshuffle1"),
    ("order_rowshuffle2", "rowshuffle2", "rowshuffle2"),
]


# ---------------------------------------------------------------- GPU run

def run(args):
    import torch
    from vlmlr import models
    from vlmlr.hooks import QwenTokenOrder
    from vlmlr.positions import merged_grids

    model, proc = models.load_qwen(args.model)
    ids = models.letter_ids(proc.tokenizer, "AB")
    order = QwenTokenOrder(model)
    checked, t0 = False, time.time()
    with ResumableWriter(args.out) as out, torch.inference_mode():
        for it in C.load_items(args.parquet, args.limit):
            rel = C.gold_relation(it)
            axis = C.AXIS[rel]
            choices = C.CHOICES[axis]
            if args.set == "row-order" and axis != "ud":
                continue
            inputs = models.build_inputs(proc, C.to_image(it["image"]), C.prompt(it["question"], choices, False))
            for name, emb, pos in (VARIANTS if args.set == "main" else VARIANTS_ROW):
                if (it["idx"], name) in out:
                    continue
                if not checked:
                    tol = order.self_check(model, inputs)
                    print(f"[self-check] identity permutation unchanged (jitter {tol:.2e}); id_only_w = main-table llm_w; rope_deltas "
                          f"unchanged; merged grid {merged_grids(inputs['image_grid_thw'], order.merge)}", flush=True)
                    checked = True
                lg, _, _ = order.forward_logits(model, inputs, emb, pos)
                lp = models.option_logprobs(lg, ids)
                out.write({"idx": it["idx"], "variant": name, "gold_orig": rel, "axis": axis,
                           "expected": rel if name == "orig" else None, "pred": models.pick(choices, lp),
                           "logp": lp, "choices": choices, "source": it["source"],
                           "grid": merged_grids(inputs["image_grid_thw"], order.merge)[0]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- analyses

def _by_item(path):
    by = collections.defaultdict(dict)
    for r in read_jsonl(path):
        by[r["idx"]][r["variant"]] = r
    return by


def order(paths):
    """Change rate and carry of every variant relative to orig (left-right items look at _w, up-down at _h; the
    other axis is the control)."""
    names = [v for v, _, _ in VARIANTS if v != "orig"]
    for path in paths:
        by = _by_item(path)
        print(f"\n=== {path}  items {len(by)}")
        xs = [d["orig"] for d in by.values() if "orig" in d]
        print(f"  orig acc {frac_ci(sum(x['pred'] == x['expected'] for x in xs), len(xs))}")
        for v in names:
            for axis in ("lr", "ud"):
                pairs = [(d["orig"], d[v]) for d in by.values() if "orig" in d and v in d and d["orig"]["axis"] == axis]
                k = sum(o["pred"] != x["pred"] for o, x in pairs)
                s0 = [o["logp"][0] - o["logp"][1] for o, _ in pairs]
                s1 = [x["logp"][0] - x["logp"][1] for _, x in pairs]
                den = sum(abs(a) for a in s0) or float("nan")
                carry = sum((1 if a > 0 else -1) * (a - b) for a, b in zip(s0, s1)) / den
                print(f"  orig→{v:<15} {axis}: flip {frac_ci(k, len(pairs))}  carry {carry:+.2f}")


def row_order(paths):
    """Row-order controls: change rates of order_only_h, row shift and row shuffles on all up-down items and on the
    geometry-consistent ones (orig and vflip correct in the main table) and direction agreement with vflip.
    Arguments come in pairs (row-order predictions, main-table predictions)."""
    shuf = ["order_rowshuffle0", "order_rowshuffle1", "order_rowshuffle2"]
    for bpath, mpath in zip(paths[::2], paths[1::2]):
        b, p1 = _by_item(bpath), _by_item(mpath)
        ud = [i for i in b if "orig" in b[i] and i in p1]
        geo = [i for i in ud if p1[i]["orig"]["pred"] == p1[i]["orig"]["expected"]
               and p1[i]["vflip"]["pred"] == p1[i]["vflip"]["expected"]]
        print(f"=== {bpath}  up-down items {len(ud)}, geometry-consistent subset {len(geo)}")
        for name, vs in (("order_only_h", ["order_only_h"]), ("order_rowshift", ["order_rowshift"]),
                         ("order_rowshuffle×3", shuf)):
            for sub, idxs in (("all", ud), ("geometry-consistent", geo)):
                k = n = 0
                for i in idxs:
                    for v in vs:
                        if v in b[i]:
                            n += 1
                            k += b[i][v]["pred"] != b[i]["orig"]["pred"]
                lo, hi = wilson(k, n)
                print(f"  {name:18s} {sub:19s}: {100 * k / max(n, 1):5.1f}% [{100 * lo:.0f},{100 * hi:.0f}] n={n}")
        flipped = [i for i in ud if b[i]["order_only_h"]["pred"] != b[i]["orig"]["pred"]]
        same = sum(p1[i]["vflip"]["pred"] != p1[i]["orig"]["pred"] for i in flipped)
        dir_ok = same / len(flipped) if flipped else float("nan")
        print(f"  same direction: of the {len(flipped)} items flipped by order_only_h, also flipped by vflip {same} ({100 * dir_ok:.1f}%)")


R = "results/token_order/"
M = "results/main_table/qwen2.5-vl/"
OUTPUTS = [
    ("outputs/token_order/order.txt", ["order", R + "order_qwen2.5-vl-3b-instruct.jsonl",
                                       R + "order_qwen2.5-vl-7b-instruct.jsonl"], True),
] + [(f"outputs/token_order/row_order_{s}.txt", ["row-order", R + f"row_order_qwen2.5-vl-{s}-instruct.jsonl",
                                                 M + f"qwen2.5-vl-{s}-instruct.jsonl"], True) for s in ("3b", "7b")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.token_order", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: token-order variants on CV-Bench, Qwen2.5-VL")
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True, help="CV-Bench test_2d.parquet")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="first N questions only")
    r.add_argument("--set", choices=["main", "row-order"], default="main",
                   help="main: the variants of Table 3; row-order: row shifts and shuffles (up-down questions only)")
    sub.add_parser("order", help=order.__doc__.split("\n")[0]).add_argument("files", nargs="+")
    sub.add_parser("row-order", help=row_order.__doc__.split("\n")[0]).add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "order":
        order(a.files)
    else:
        row_order(a.files)


if __name__ == "__main__":
    main()
