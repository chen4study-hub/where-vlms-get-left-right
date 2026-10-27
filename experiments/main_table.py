"""Main table: position-only counterfactuals on CV-Bench 2D Relation (paper Section 4.1, Table 2, Fig. 2a).

Four models: Qwen2.5-VL-3B / 7B (2D-RoPE in the ViT, M-RoPE in the LLM) and InternVL3-2B / 8B (learned absolute
position embedding in the ViT, 1D RoPE in the LLM). Each of the 650 questions is asked under the 19 conditions
of vlmlr.cvbench.variants; one forward pass per condition, scored by the option-letter logits.

  python -m experiments.main_table run --family qwen --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --parquet data/cvbench/test_2d.parquet --out results/main_table/qwen2.5-vl/qwen2.5-vl-7b-instruct.jsonl
  python -m experiments.main_table summary  <predictions.jsonl> ...   # accuracy, consistency, change rates
  python -m experiments.main_table compare  <predictions.jsonl> ...   # agreement with real flips (Table 2)
  python -m experiments.main_table ties     <predictions.jsonl> ...   # Table 2 metrics without bf16 ties
  python -m experiments.main_table leakage  <reference.jsonl> <other.jsonl> ...   # cross-axis leakage differences
"""
import argparse
import math
import time
from collections import defaultdict

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl
from vlmlr.stats import frac_ci, pct_ci, percentile_ci, resamples


# ---------------------------------------------------------------- GPU run

def run(args):
    import torch
    from vlmlr import models
    from vlmlr.hooks import InternVLPositionMirror, QwenPositionMirror

    if args.family == "qwen":
        model, proc = models.load_qwen(args.model)
        mirror = QwenPositionMirror(model)
        image_token_id = model.config.image_token_id
    else:
        model, proc = models.load_internvl(args.model)
        mirror = InternVLPositionMirror(model)
        image_token_id = mirror.image_token_id
        assert proc.image_token_id == image_token_id, "processor and config disagree on the image token"
    ids = models.letter_ids(proc.tokenizer, "AB")
    items = C.load_items(args.parquet, args.limit)
    checked, t0 = False, time.time()
    with ResumableWriter(args.out) as out, torch.inference_mode():
        for it in items:
            img0 = C.to_image(it["image"])
            for v in C.variants(it):
                if (it["idx"], v["variant"]) in out:
                    continue
                text = C.prompt(v["question"], v["choices"], v["frame_hint"])
                inputs = models.build_inputs(proc, C.transform(img0, v["transform"]), text, args.family)
                if not checked:          # first item: the hooks reproduce the unmodified model where they should
                    mirror.self_check(inputs) if args.family == "qwen" else mirror.self_check(model, inputs)
                    checked = True
                lp = models.option_logprobs(mirror.forward_logits(model, inputs, v["llm_flip"], v["vit_flip"]), ids)
                gold = C.gold_relation(it)
                out.write({"idx": it["idx"], "variant": v["variant"], "gold_orig": gold, "expected": v["expected"],
                           "pred": models.pick(v["choices"], lp), "logp": lp, "choices": v["choices"],
                           "axis": C.AXIS[gold], "source": it["source"],
                           "n_img_tokens": int((inputs["input_ids"] == image_token_id).sum())})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- analyses

def _load(path):
    by, axis = defaultdict(dict), {}
    for r in C.main_rows(read_jsonl(path)):
        by[r["idx"]][r["variant"]] = r
        axis[r["idx"]] = r["axis"]
    return by, axis


def summary(paths):
    """Accuracy per variant, self-contradiction rates and change rates of every position-only variant."""
    implied = {"swap": lambda o: C.INVERSE[o],
               "hflip": lambda o: C.INVERSE[o] if o in ("left", "right") else o,
               "vflip": lambda o: C.INVERSE[o] if o in ("above", "below") else o}
    combos = [(f"{s}_{ax}", "orig") for ax in "wh" for s in ("llm", "vit", "both")]
    combos += [(f"{pix}_{s}_{ax}", pix) for ax, pix in (("w", "hflip"), ("h", "vflip")) for s in ("llm", "vit", "both")]
    combos += [("ids_rot180", "rot180"), ("ids_rot180", "orig")]
    for path in paths:
        by, _ = _load(path)
        print(f"\n=== {path}  items {len(by)}")
        print("-- accuracy (against the answer implied by the geometry)")
        for v in ["orig", "swap", "hflip", "vflip", "rot180"]:
            for ax in ("lr", "ud"):
                xs = [d[v] for d in by.values() if v in d and d[v]["axis"] == ax]
                print(f"  {v:<11} {ax}: {frac_ci(sum(x['pred'] == x['expected'] for x in xs), len(xs))}")
        print("-- self-contradiction rate (logically inconsistent with the orig prediction, gold not used)")
        for v, f in implied.items():
            for ax in ("lr", "ud"):
                xs = [(d["orig"], d[v]) for d in by.values() if "orig" in d and v in d and d["orig"]["axis"] == ax]
                print(f"  orig↔{v:<10} {ax}: {frac_ci(sum(b['pred'] != f(o['pred']) for o, b in xs), len(xs))}")
        print("-- position flips: share of answers that change relative to the base variant (lr items: _w, ud items: _h; "
              "the other axis is the control and should be near 0)")
        print("   positions only (base orig): how far left-right / up-down follow the position ids;")
        print("   pixels flipped + positions flipped back (base hflip / vflip): how many answers flip back")
        for v, base in combos:
            for ax in ("lr", "ud"):
                xs = [(d[base], d[v]) for d in by.values() if base in d and v in d and d[base]["axis"] == ax]
                print(f"  {base}→{v:<11} {ax}: {frac_ci(sum(o['pred'] != b['pred'] for o, b in xs), len(xs))}")
        print("-- accuracy against the gold label and confidence (orig)")
        xs = [d["orig"] for d in by.values() if "orig" in d]
        conf = sum(math.exp(max(x["logp"])) for x in xs) / max(len(xs), 1)
        print(f"  orig acc {frac_ci(sum(x['pred'] == x['expected'] for x in xs), len(xs))}  mean confidence {conf:.2f}")


def _geometry_consistent(by, axis, ref, a, extra=lambda i: True):
    """Items of axis a answered correctly on orig and on the real transform ref."""
    return [i for i in by if axis[i] == a and extra(i) and by[i]["orig"]["pred"] == by[i]["orig"]["expected"]
            and by[i][ref]["pred"] == by[i][ref]["expected"]]


def compare(paths):
    """Agreement of position-only variants with the real pixel transform, and change rates on the
    geometry-consistent subset (the metrics of Table 2)."""
    for path in paths:
        by, axis = _load(path)
        print("==", path.split("/")[-1])
        for a, lab in (("lr", "left"), ("ud", "above")):
            I = [i for i in by if axis[i] == a]
            print(f"[{a}] orig predicts {lab}: {pct_ci(sum(by[i]['orig']['pred'] == lab for i in I), len(I))}; "
                  f"gold {pct_ci(sum(by[i]['orig']['gold_orig'] == lab for i in I), len(I))}")

        def agree(v, ref, a):
            I = [i for i in by if axis[i] == a]
            return pct_ci(sum(by[i][v]["pred"] == by[i][ref]["pred"] for i in I), len(I))
        for v, ref in (("vit_w", "hflip"), ("llm_w", "hflip"), ("orig", "hflip"), ("vit_h", "vflip"),
                       ("orig", "vflip"), ("ids_rot180", "rot180"), ("orig", "rot180")):
            ud = "" if ref == "rot180" else f" | ud: {agree(v, ref, 'ud')}"
            print(f"  {v:10s} vs {ref:6s} lr: {agree(v, ref, 'lr')}{ud}")

        def gc(ref, v, a):
            I = _geometry_consistent(by, axis, ref, a)
            return pct_ci(sum(by[i][v]["pred"] != by[i]["orig"]["pred"] for i in I), len(I))
        print("  flip rates, geometry-consistent subset: orig→vit_w lr", gc("hflip", "vit_w", "lr"), "| ud", gc("hflip", "vit_w", "ud"))
        print("                                         orig→llm_w lr", gc("hflip", "llm_w", "lr"))
        print("                                         orig→vit_h ud", gc("vflip", "vit_h", "ud"), "| lr", gc("vflip", "vit_h", "lr"))
        print("                                         orig→llm_h ud", gc("vflip", "llm_h", "ud"))


def ties(paths):
    """The metrics of `compare` after removing every item with an exact tie in any compared condition."""
    for path in paths:
        by, axis = _load(path)
        tie = defaultdict(dict)
        for i, d in by.items():
            for v, r in d.items():
                lp = r.get("logp") or r.get("lp")
                tie[i][v] = bool(lp and lp[0] == lp[1])
        print("==", path.split("/")[-1])

        def no_tie(i, *vs):
            return not any(tie[i][v] for v in vs)

        def agree(v, ref, a):
            I = [i for i in by if axis[i] == a and no_tie(i, v, ref)]
            return pct_ci(sum(by[i][v]["pred"] == by[i][ref]["pred"] for i in I), len(I))
        for v, ref in (("vit_w", "hflip"), ("llm_w", "hflip"), ("orig", "hflip"), ("vit_h", "vflip"), ("orig", "vflip")):
            print(f"  {v:6s} vs {ref:6s} lr: {agree(v, ref, 'lr')} | ud: {agree(v, ref, 'ud')}")

        def gc(ref, v, a):
            I = _geometry_consistent(by, axis, ref, a, lambda i: no_tie(i, "orig", ref, v))
            return pct_ci(sum(by[i][v]["pred"] != by[i]["orig"]["pred"] for i in I), len(I))
        print("  flip rates, geometry-consistent subset: orig→vit_w lr", gc("hflip", "vit_w", "lr"), "| ud", gc("hflip", "vit_w", "ud"))
        print("                                         orig→llm_w lr", gc("hflip", "llm_w", "lr"))
        print("                                         orig→vit_h ud", gc("vflip", "vit_h", "ud"), "| lr", gc("vflip", "vit_h", "lr"))
        print("                                         orig→llm_h ud", gc("vflip", "llm_h", "ud"))
        per_variant = defaultdict(int)
        for i in tie:
            for v, t in tie[i].items():
                per_variant[v] += t
        print("  rows with a tie (per variant):", {k: v for k, v in sorted(per_variant.items()) if v})


# (label, item axis, counterfactual, real transform defining the geometry-consistent subset or None = all items)
LEAKAGE_METRICS = [
    ("vit_w→ud all items", "ud", "vit_w", None), ("vit_w→ud geometry-consistent", "ud", "vit_w", "hflip"),
    ("vit_h→lr all items", "lr", "vit_h", None), ("vit_h→lr geometry-consistent", "lr", "vit_h", "vflip"),
    ("vit_w→lr geometry-consistent (same axis)", "lr", "vit_w", "hflip"), ("vit_h→ud geometry-consistent (same axis)", "ud", "vit_h", "vflip")]


def leakage(paths):
    """Change rates of each model minus those of the first (reference) model; the 650 items are resampled
    jointly (10000 times, seed 0) and each model keeps its own geometry-consistent subset."""
    data = [_load(p) for p in paths]
    names = [p.split("/")[-1].replace(".jsonl", "") for p in paths]
    items = sorted(data[0][0])
    B = 10000
    boots = [[items[k] for k in s] for s in resamples(len(items), B, 0)]

    def changed(by, axis, a, v, ref):
        out = {}
        for i in by:
            if axis[i] != a:
                continue
            if ref and not (by[i]["orig"]["pred"] == by[i]["orig"]["expected"]
                            and by[i][ref]["pred"] == by[i][ref]["expected"]):
                continue
            out[i] = int(by[i][v]["pred"] != by[i]["orig"]["pred"])
        return out

    def rate(d, sample):
        k = n = 0
        for i in sample:
            if i in d:
                k += d[i]
                n += 1
        return 100 * k / n if n else float("nan")
    for lab, a, v, ref in LEAKAGE_METRICS:
        ds = [changed(by, axis, a, v, ref) for by, axis in data]
        pts = [rate(d, items) for d in ds]
        print(f"[{lab}] " + "  ".join(f"{nm}={p:.1f} (n={len(d)})" for nm, p, d in zip(names, pts, ds)))
        for j in range(1, len(ds)):
            lo, hi = percentile_ci(sorted(rate(ds[j], s) - rate(ds[0], s) for s in boots))
            print(f"    {names[j]} − {names[0]}: {pts[j] - pts[0]:+.1f} [{lo:+.1f}, {hi:+.1f}]")


ANALYSES = {"summary": summary, "compare": compare, "ties": ties, "leakage": leakage}

_Q = "results/main_table/qwen2.5-vl/qwen2.5-vl-{}-instruct.jsonl"
Q3, Q7 = _Q.format("3b"), _Q.format("7b")
I2 = "results/main_table/internvl3-2b/internvl3-2b-hf.jsonl"
I8 = "results/main_table/internvl3-8b/internvl3-8b-hf.jsonl"
# (output file, arguments of the analysis command that writes it, behind a table, figure or the main text)
OUTPUTS = [
    ("outputs/main_table/summary_qwen2.5-vl.txt", ["summary", Q3, Q7], False),
    ("outputs/main_table/summary_internvl3-2b.txt", ["summary", I2], True),
    ("outputs/main_table/summary_internvl3-8b.txt", ["summary", I8], True),
    ("outputs/main_table/flip_agreement.txt", ["compare", Q3, Q7, I2, I8], True),
    ("outputs/main_table/without_ties.txt", ["ties", Q3, Q7, I2], True),
    ("outputs/main_table/cross_axis_leakage.txt", ["leakage", I2, I8, Q3, Q7], True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.main_table", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: run the 19 conditions on CV-Bench")
    r.add_argument("--family", choices=["qwen", "internvl"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True, help="CV-Bench test_2d.parquet")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="first N questions only")
    for name, fn in ANALYSES.items():
        sub.add_parser(name, help=fn.__doc__.split("\n")[0]).add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    else:
        ANALYSES[a.cmd](a.files)


if __name__ == "__main__":
    main()
