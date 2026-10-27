"""What'sUp replication: the main-table counterfactuals on a second benchmark (paper Section 4.3, Table 4, Fig. 2b).

412 photographs of Controlled Images A, four captions each (left of / right of / on / under), seven conditions
(vlmlr.whatsup.VARIANTS), scored by the largest of the four option-letter logits.

  python -m experiments.whatsup run --family qwen --model Qwen/Qwen2.5-VL-3B-Instruct \\
      --data data/whatsup --out results/whatsup/qwen2.5-vl_internvl3-2b/qwen25vl3b.jsonl
  python -m experiments.whatsup summary <predictions.jsonl> ...    # accuracy, change rates, agreement
  python -m experiments.whatsup ties <predictions.jsonl> ...       # summary after removing items with tied logits
  python -m experiments.whatsup off-axis <predictions.jsonl> ...   # where left-right answers go under vit_h
  python -m experiments.whatsup lr-swap <predictions.jsonl> ...    # left<->right swaps under vit_h / llm_h
"""
import argparse
import os
import time
from collections import Counter, defaultdict

from vlmlr import whatsup as W
from vlmlr.records import ResumableWriter, read_jsonl
from vlmlr.stats import resamples, wilson

LR_SWAP = {"left": "right", "right": "left"}


# ---------------------------------------------------------------- GPU run

def dataset_root(data):
    """A local snapshot directory, or a Hugging Face dataset id already in the local cache."""
    if os.path.isdir(data):
        return data
    from huggingface_hub import snapshot_download
    return snapshot_download(data, repo_type="dataset", local_files_only=True)


def run(args):
    import torch
    from PIL import Image
    from vlmlr import models
    from vlmlr.cvbench import transform
    from vlmlr.hooks import InternVLPositionMirror, QwenPositionMirror

    items = W.load_items(dataset_root(args.data), args.limit)
    print(f"items: {len(items)} (lr {sum(i['axis'] == 'lr' for i in items)}, ud {sum(i['axis'] == 'ud' for i in items)})",
          flush=True)
    if args.family == "qwen":
        model, proc = models.load_qwen(args.model)
        mirror = QwenPositionMirror(model)
    else:
        model, proc = models.load_internvl(args.model)
        mirror = InternVLPositionMirror(model)

    def self_check(inputs):
        if args.family == "qwen":
            mirror.self_check(inputs)
            o = mirror.forward_logits(model, inputs)
            assert not torch.equal(mirror.forward_logits(model, inputs, vit="w"), o) and \
                not torch.equal(mirror.forward_logits(model, inputs, llm="w"), o), "mirroring did not change the output"
        else:
            mirror.self_check(model, inputs, full=False)

    ids = models.letter_ids(proc.tokenizer, "ABCD")
    checked, t0 = False, time.time()
    with ResumableWriter(args.out) as out, torch.inference_mode():
        for it in items:
            img0 = Image.open(it["path"]).convert("RGB")
            text = W.prompt(it["options"])
            for name, tf, llm, vit in W.VARIANTS:
                if (it["idx"], name) in out:
                    continue
                inputs = models.build_inputs(proc, transform(img0, tf), text, args.family)
                if not checked:
                    self_check(inputs)
                    checked = True
                lg = mirror.forward_logits(model, inputs, llm, vit)
                l4 = [float(lg[t]) for t in ids]
                out.write({"idx": it["idx"], "variant": name, "gold": it["gold"], "axis": it["axis"],
                           "expected": W.expected(it, tf) if not (llm or vit) else None,
                           "pred": W.predicted_relation(it, l4), "logits": l4, "rels": it["rels"]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- analyses

def _fmt(k, n):
    if n == 0:
        return "  —  "
    lo, hi = wilson(k, n, clamp=True)
    return f"{100 * (k / n):5.1f}% [{100 * lo:.0f},{100 * hi:.0f}] n={n}"


def _by_item(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["idx"]][r["variant"]] = r
    return by


def _name(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def report(rows, name):
    """Accuracy, change rates on the geometry-consistent subset, and agreement with real transforms."""
    by = _by_item(rows)
    print(f"\n=== {name}  items {len(by)}")
    print("-- accuracy (against the relation implied by the geometry)")
    for v in ("orig", "hflip", "vflip"):
        for ax in ("lr", "ud"):
            xs = [d[v] for d in by.values() if v in d and d[v]["axis"] == ax]
            print(f"  {v:7s} {ax}: {_fmt(sum(x['pred'] == x['expected'] for x in xs), len(xs))}")
    print("-- flip rates on the geometry-consistent subset (relative to orig; 'mirror': changed to the mirror relation)")
    flip = {}
    for ax in ("lr", "ud"):
        g = W.geometry_consistent(by, ax)
        for v, mir in (("vit_w", "hflip"), ("llm_w", "hflip"), ("vit_h", "vflip"), ("llm_h", "vflip")):
            k, km, n = W.change_rate(by, g, v, mir)
            flip[v, ax] = k / n if n else None
            print(f"  [{ax}] orig→{v:10s} {_fmt(k, n)}  mirror {_fmt(km, n)}")
    print("-- agreement with the predictions for the real pixel transform (all items)")
    for v, ref in (("vit_w", "hflip"), ("llm_w", "hflip"), ("vit_h", "vflip"), ("orig", "hflip")):
        for ax in ("lr", "ud"):
            xs = [(d[v], d[ref]) for d in by.values() if v in d and ref in d and d[v]["axis"] == ax]
            print(f"  {v:10s} vs {ref:6s} {ax}: {_fmt(sum(x['pred'] == y['pred'] for x, y in xs), len(xs))}")
    return flip


def summary(paths, rows_of=None):
    for p in paths:
        report(rows_of(p) if rows_of else W.main_rows(read_jsonl(p)), _name(p))


def _tied(r):
    lg = sorted(r["logits"], reverse=True)
    return len(lg) > 1 and lg[0] == lg[1]


def ties(paths):
    """Remove every item with a tie of the two highest logits in any variant, then re-run `summary`."""
    kept = {}
    for p in paths:
        rows = W.main_rows(read_jsonl(p))
        tie_idx, per_variant = set(), Counter()
        for r in rows:
            if _tied(r):
                tie_idx.add(r["idx"])
                per_variant[r["variant"]] += 1
        n_idx = len({r["idx"] for r in rows})
        print(f"{os.path.basename(p)}: {len(rows)} rows, {sum(per_variant.values())} tied"
              f" ({100 * sum(per_variant.values()) / len(rows):.1f}%), on {len(tie_idx)}/{n_idx} items; "
              f"per variant {dict(per_variant.most_common(6))}")
        kept[p] = [r for r in rows if r["idx"] not in tie_idx]
    print("---- re-computed without the tied items ----")
    summary(paths, rows_of=kept.__getitem__)


def off_axis(paths):
    """Left-right items whose answer changes under vit_h: how many move to on/under, how many swap left<->right."""
    for p in paths:
        by = _by_item(W.main_rows(read_jsonl(p)))
        tot = under = swap = 0
        for i, d in by.items():
            r = d.get("vit_h")
            if r is None or r["axis"] != "lr":
                continue
            o = d["orig"]
            if o["pred"] != r["pred"]:
                tot += 1
                under += r["pred"] in ("under", "on")
                swap += {o["pred"], r["pred"]} == {"left", "right"}
        gc = [i for i, d in by.items() if "orig" in d and d["orig"]["axis"] == "lr"
              and d["orig"]["pred"] == d["orig"]["expected"] and d["hflip"]["pred"] == d["hflip"]["expected"]]
        sw = sum({by[i]["orig"]["pred"], by[i]["vit_h"]["pred"]} == {"left", "right"} for i in gc)
        print(_name(p), "changed", tot, "to on/under", under, "left<->right", swap, "| geometry-consistent subset", len(gc), "left-right swaps", sw)


def _lr_consistent(by):
    """Left-right items answered correctly on orig and on hflip."""
    return [i for i in by if by[i]["orig"]["axis"] == "lr" and by[i]["orig"]["pred"] == by[i]["orig"]["expected"]
            and by[i]["hflip"]["pred"] == by[i]["hflip"]["expected"]]


def lr_swap(paths):
    """Geometry-consistent left-right items whose answer swaps left<->right under vit_h / llm_h; with both
    InternVL3 sizes, the 8B - 2B difference (412 items resampled jointly 10000 times, seed 0)."""
    for p in paths:
        by = _by_item(W.main_rows(read_jsonl(p)))
        I = _lr_consistent(by)
        for v in ("vit_h", "llm_h"):
            k = sum(by[i][v]["pred"] == LR_SWAP.get(by[i]["orig"]["pred"]) for i in I)
            ch = Counter(by[i][v]["pred"] for i in I if by[i][v]["pred"] != by[i]["orig"]["pred"])
            lo, hi = wilson(k, len(I))
            print(f"{p.split('/')[-1]:20s} {v}: left-right swaps {k}/{len(I)} = {100 * k / len(I):.1f}% "
                  f"[{100 * lo:.1f}, {100 * hi:.1f}]; changed answers go to {dict(ch)}")
    joined = " ".join(paths)
    if len(paths) >= 2 and "internvl32b" in joined and "internvl38b" in joined:
        def per(path):
            by = _by_item(W.main_rows(read_jsonl(path)))
            return {i: int(by[i]["vit_h"]["pred"] == LR_SWAP.get(by[i]["orig"]["pred"])) for i in _lr_consistent(by)}
        a = per([p for p in paths if "internvl32b" in p][0])
        b = per([p for p in paths if "internvl38b" in p][0])
        ids = sorted(set(a) | set(b) | set(range(412)))

        def rate(d, s):
            k = n = 0
            for i in s:
                if i in d:
                    k += d[i]
                    n += 1
            return 100 * k / n
        ds = sorted(rate(b, [ids[j] for j in s]) - rate(a, [ids[j] for j in s]) for s in resamples(len(ids), 10000, 0))
        print(f"vit_h left-right swaps 8B − 2B: {rate(b, ids) - rate(a, ids):+.1f} [{ds[249]:+.1f}, {ds[9749]:+.1f}]"
              f" (412 items resampled jointly 10000 times, seed 0)")


ANALYSES = {"summary": summary, "ties": ties, "off-axis": off_axis, "lr-swap": lr_swap}

_D = "results/whatsup/qwen2.5-vl_internvl3-2b/"
Q3, Q7, I2 = _D + "qwen25vl3b.jsonl", _D + "qwen25vl7b.jsonl", _D + "internvl32b.jsonl"
I8 = "results/whatsup/internvl3-8b/internvl38b.jsonl"
OUTPUTS = [
    ("outputs/whatsup/summary.txt", ["summary", Q3, Q7, I2], True),
    ("outputs/whatsup/without_ties.txt", ["ties", Q3, Q7, I2], False),
    ("outputs/whatsup/off_axis.txt", ["off-axis", Q3, Q7, I2, I8], True),
    ("outputs/whatsup/height_mirror_lr_swaps.txt", ["lr-swap", Q3, Q7, I2, I8], True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.whatsup", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: run the seven conditions on What'sUp")
    r.add_argument("--family", choices=["qwen", "internvl"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--data", default="amitakamath2/whatsup_vlms",
                   help="dataset directory (test.json + controlled_images/) or the Hugging Face dataset id")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="first N photographs only")
    for name, fn in ANALYSES.items():
        sub.add_parser(name, help=fn.__doc__.split("\n")[0] if fn.__doc__ else None).add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    else:
        ANALYSES[a.cmd](a.files)


if __name__ == "__main__":
    main()
