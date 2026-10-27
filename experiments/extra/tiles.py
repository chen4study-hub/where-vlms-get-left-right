"""Image tiles and tile order in InternVL3-2B (paper Section 4.2).

InternViT encodes 448 x 448 tiles (32 x 32 patches, learned absolute position embedding added once); after the
0.5 pixel shuffle each tile gives 256 LLM tokens, and the tiles enter the LLM in row-major order. Fixed layouts
instead of the dynamic tiling:
  S         one tile (the setting of the main table)
  H-T       the image resized to 896 x 448 and cut into a left and a right tile, no thumbnail
  H-T_swap  the same two tiles in reversed order
Options are scored in fp32; left-right questions of the CV-Bench validation split.

  python -m experiments.extra tiles run --model OpenGVLab/InternVL3-2B-hf --parquet data/cvbench/test_2d.parquet \\
      --out results/extra/tiles/internvl3-2b.jsonl
  python -m experiments.extra tiles summary <predictions.jsonl> [--json <summary.json>]
"""
import argparse
import json
import statistics
import time

from experiments.extra._shared import load_cvbench_validation, margin_ci, paired
from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl

TILE = 448
LAYOUTS = {"S": (1, 1), "H": (1, 2)}          # (rows R, columns C)
CONDS = ["S", "H-T", "H-T_swap"]


def tile_boxes(layout):
    """Crop boxes (left, top, right, bottom) of the tiles on the resized whole image, with tile coordinates (r, c)."""
    R, Cn = LAYOUTS[layout]
    return [((c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE), (r, c)) for r in range(R) for c in range(Cn)]


def parse_cond(cond):
    """'S' / 'H-T' / 'H-T_swap' -> dict(layout, swap)."""
    if cond not in CONDS:
        raise ValueError(cond)
    return {"layout": cond[0], "swap": cond.endswith("_swap")}


def tile_plan(cond):
    """The tiles fed to the ViT [(tile (r, c), layout)]."""
    p = parse_cond(cond)
    tiles = [rc for _, rc in tile_boxes(p["layout"])]
    if p["swap"]:
        tiles = tiles[::-1]
    return [(rc, p["layout"]) for rc in tiles]


def acc(recs, cond, axis):
    """Accuracy (%); a tie (pred None after fp32 scoring) counts 0.5."""
    xs = [0.5 if r["pred"] is None else float(r["pred"] == r["gold"]) for r in recs if r["cond"] == cond and r["axis"] == axis]
    return (sum(xs) / len(xs) * 100) if xs else float("nan"), len(xs)


def ties(recs):
    return sum(r.get("pred", 0) is None for r in recs)


def contrast(recs, cond, base, axis):
    """Paired comparison: accuracy difference (points), Δm with interval, net fixes."""
    p = paired([r for r in recs if r["axis"] == axis], cond, base)
    if not p:
        return None
    d = [mc - mb for _, mb, mc, *_ in p]
    lo, hi = margin_ci(d)
    return {"n": len(p), "dacc": (sum(x[4] for x in p) - sum(x[3] for x in p)) / len(p) * 100,
            "dm": statistics.mean(d), "ci": (lo, hi), "net_fix": sum((not x[3]) and x[4] for x in p) - sum(x[3] and not x[4] for x in p)}


# ---------------------------------------------------------------- GPU run

def tile_pixels(proc, img, layout):
    """Normalised pixels (3, 448, 448) of every tile, resized by the processor itself."""
    R, Cn = LAYOUTS[layout]
    ip = proc.image_processor
    big = ip(images=[img], crop_to_patches=False, size={"height": R * TILE, "width": Cn * TILE},
             return_tensors="pt")["pixel_values"][0]
    assert tuple(big.shape[-2:]) == (R * TILE, Cn * TILE), f"processor did not resize as requested: {tuple(big.shape)}"
    return {rc: big[:, t:b, l:r] for (l, t, r, b), rc in tile_boxes(layout)}


def run(a):
    import torch
    from vlmlr import models
    from vlmlr.models import FP32OptionScorer
    model, proc = models.load_internvl(a.model)
    tok = proc.tokenizer
    ids = models.letter_ids(tok, "AB")
    assert proc.image_seq_length == 256, f"{proc.image_seq_length} tokens per tile"
    scorer = FP32OptionScorer(model, ids)

    def build(chat, cache, img, cond):
        seq = tile_plan(cond)
        lay = seq[0][1]
        if lay not in cache:
            cache[lay] = tile_pixels(proc, img, lay)
        pix = torch.stack([cache[lay][rc] for rc, _ in seq]).to("cuda", torch.bfloat16)
        text = chat.replace(proc.image_token, f"{proc.start_image_token}{proc.image_token * 256 * len(seq)}{proc.end_image_token}", 1)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        inp = {"input_ids": enc["input_ids"].to("cuda"), "attention_mask": enc["attention_mask"].to("cuda"), "pixel_values": pix}
        return inp, seq

    def fwd(inp):
        model(**inp)
        return scorer.scores()

    def self_check(chat, img):
        inp, _ = build(chat, {}, img, "S")
        ref = proc(text=[chat], images=[img], crop_to_patches=False, return_tensors="pt")
        assert torch.equal(inp["input_ids"].cpu(), ref["input_ids"]), "single-tile text differs from the processor"
        dpix = float((inp["pixel_values"].float().cpu() - ref["pixel_values"].float()).abs().max())
        assert dpix <= 1e-2, f"single-tile pixels differ from the processor by {dpix:.4f}"
        assert torch.equal(fwd(inp), fwd(inp)), "single-tile scores not reproducible"

    items = [it for it in load_cvbench_validation(a.parquet) if C.AXIS[C.gold_relation(it)] == "lr"]
    items = items[: a.limit] if a.limit else items
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("idx", "cond")) as out, torch.inference_mode():
        for it in items:
            rel = C.gold_relation(it)
            axis = C.AXIS[rel]
            choices = C.CHOICES[axis]
            chat = models.chat_text(proc, C.prompt(it["question"], choices, False))
            img, cache = C.to_image(it["image"]), {}
            if not checked:
                self_check(chat, img)
                checked = True
            for c in CONDS:
                if (it["idx"], c) in out:
                    continue
                inp, seq = build(chat, cache, img, c)
                lp = torch.log_softmax(fwd(inp), -1).tolist()
                tie = lp[0] == lp[1]
                out.write({"idx": it["idx"], "cond": c, "axis": axis, "gold": rel, "choices": choices,
                           "pred": None if tie else choices[int(lp[1] > lp[0])], "tie": tie, "logp": lp, "n_tiles": len(seq),
                           "source": it["source"]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


# ---------------------------------------------------------------- analysis

PAIRS = [("H-T_swap", "H-T", "lr")]


def _show(recs, pairs):
    print(f"ties (still exactly equal in fp32; accuracy counts 0.5, paired comparisons count them as wrong): {ties(recs)}")
    print(f"{'condition':10s} {'lr accuracy':>14s}")
    for c in CONDS:
        al, nl = acc(recs, c, "lr")
        print(f"{c:10s} {al:7.1f}% n={nl}")
    out = {}
    for c, b, ax in pairs:
        s = contrast(recs, c, b, ax)
        if s:
            out[f"{c} vs {b} ({ax})"] = {**s, "ci": list(s["ci"])}
            print(f"  {c} vs {b} ({ax}): Δacc {s['dacc']:+.1f} points, Δm {s['dm']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}], net fixes {s['net_fix']:+d}")
    return out


def summary(a):
    recs = [r for r in read_jsonl(a.predictions) if r["cond"] in CONDS and r["axis"] == "lr"]
    print("== Fixed layouts, left-right questions")
    res = _show(recs, PAIRS)
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.extra tiles", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: fixed tile layouts on the left-right questions of the CV-Bench validation split")
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary", help="accuracy per layout and the paired tile-order contrast")
    s.add_argument("predictions")
    s.add_argument("--json")
    a = ap.parse_args(argv)
    {"run": run, "summary": summary}[a.cmd](a)


R = "results/extra/tiles/"
OUTPUTS = [("outputs/extra/tiles.txt", ["summary", R + "internvl3-2b.jsonl"])]
