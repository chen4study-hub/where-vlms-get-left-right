"""Position / content / neither decomposition (paper Section 4.5, Fig. 4a; Section 5.1) and the blank-image controls.

Each CV-Bench question is also asked with the image replaced by a grey image of the same size (same number of
image tokens). Together with the main-table predictions every answer is assigned one source:
  position  the answer changes when only the ViT positions are mirrored (vit_w for left-right, vit_h for up-down);
  content   it does not, but it changes on the blank image;
  neither   it changes under neither.

  python -m experiments.decomposition run --family qwen --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --parquet data/cvbench/test_2d.parquet --out blank.jsonl [--scoring logits]
  python -m experiments.decomposition sources --pair NAME <main.jsonl> <blank.jsonl> [--pair ...]
  python -m experiments.decomposition error-types LABEL <main.jsonl> <blank.jsonl> [LABEL ...]
"""
import argparse
import collections
import os
import time

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl

MIRROR = {"lr": "vit_w", "ud": "vit_h"}
REAL = {"lr": "hflip", "ud": "vflip"}
SOURCES = ("position", "content", "neither")


# ---------------------------------------------------------------- GPU run

def run(args):
    import torch
    from PIL import Image
    from vlmlr import models

    orders = ["fwd"]                    # options in the order of the main table
    fp32 = args.scoring == "fp32"
    model, proc = models.load(args.model, args.family)
    ids = models.letter_ids(proc.tokenizer, "AB")
    img_tok = model.config.image_token_id
    if fp32:
        # option scores in fp32 (last hidden state x the two lm_head rows): bf16 logits are occasionally exactly tied,
        # and a tie falls on the first option
        last = {}
        model.lm_head.register_forward_pre_hook(lambda m, a: last.__setitem__("h", a[0][0, -1].detach()))
        w_ab = model.lm_head.weight[ids].float()

    def logits(inputs):
        if args.family == "qwen":
            model.model.rope_deltas = None
        if fp32:
            model(**inputs)
            return last["h"].float() @ w_ab.t()
        return model(**inputs).logits[0, -1].float()

    checked, t0 = False, time.time()
    key = ("idx", "order") if fp32 else ("idx",)
    with ResumableWriter(args.out, key=key) as out, torch.inference_mode():
        for it in C.load_items(args.parquet, args.limit):
            if all(((it["idx"], o) if fp32 else (it["idx"],)) in out for o in orders):
                continue
            img0 = C.to_image(it["image"])
            blank = Image.new("RGB", img0.size, (128, 128, 128))
            rel = C.gold_relation(it)
            axis = C.AXIS[rel]
            for order in orders:
                if fp32 and (it["idx"], order) in out:
                    continue
                choices = C.CHOICES[axis] if order == "fwd" else C.CHOICES[axis][::-1]
                text = C.prompt(it["question"], choices, False)
                inputs = models.build_inputs(proc, blank, text, args.family)
                if not checked:
                    ref = models.build_inputs(proc, img0, text, args.family)
                    nb, no = int((inputs["input_ids"] == img_tok).sum()), int((ref["input_ids"] == img_tok).sum())
                    assert nb == no, f"blank image has {nb} image tokens, the original {no}"
                    assert torch.equal(logits(inputs), logits(inputs)), "forward pass not reproducible"
                    if fp32:
                        lg_ref = model(**inputs).logits[0, -1].float()[ids]
                        assert (lg_ref - logits(inputs)).abs().max() < 0.25, "fp32 option scores differ from the logits"
                    print(f"[self-check] passed: blank and original image have the same number of image tokens ({nb}), forward pass reproducible", flush=True)
                    checked = True
                if fp32:
                    lp = torch.log_softmax(logits(inputs), -1).tolist()
                    tie = lp[0] == lp[1]
                    out.write({"idx": it["idx"], "variant": "blank", "order": order, "axis": axis, "gold": rel,
                               "choices": choices, "pred": None if tie else choices[int(lp[1] > lp[0])], "tie": tie,
                               "logp": lp, "source": it["source"]})
                else:
                    lp = models.option_logprobs(logits(inputs), ids)
                    out.write({"idx": it["idx"], "variant": "blank", "axis": axis, "gold": rel, "choices": choices,
                               "pred": models.pick(choices, lp), "logp": lp, "source": it["source"]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- decomposition

def index_sources(main_rows, blank_rows):
    """{idx: {"axis", "gold", "orig", "mirror", "real", "blank"}} for the questions with all four predictions.
    main_rows: main-table predictions; blank_rows: blank-image predictions (a later record of the same question
    replaces an earlier one)."""
    by = collections.defaultdict(dict)
    for r in main_rows:
        by[r["idx"]][r["variant"]] = r
    out = {}
    for b in blank_rows:
        d = by.get(b["idx"], {})
        ax = b["axis"]
        if all(k in d for k in ("orig", MIRROR[ax], REAL[ax])):
            out[b["idx"]] = {"axis": ax, "gold": d["orig"]["gold_orig"], "orig": d["orig"]["pred"],
                             "mirror": d[MIRROR[ax]]["pred"], "real": d[REAL[ax]]["pred"], "blank": b["pred"]}
    return out


def answer_source(q):
    """'position' if the ViT mirror changes the answer, else 'content' if the blank image does, else 'neither'."""
    if q["mirror"] != q["orig"]:
        return "position"
    return "content" if q["blank"] != q["orig"] else "neither"


def decompose(items, axis, consistent_only=False):
    """Share (%) of each source on one axis; consistent_only: questions whose answer the real flip changes."""
    qs = [q for q in items.values() if q["axis"] == axis and (not consistent_only or q["real"] != q["orig"])]
    c = collections.Counter(answer_source(q) for q in qs)
    n = len(qs)
    return {k: (100 * c[k] / n if n else float("nan")) for k in SOURCES}, n


def blank_distribution(items, axis):
    """Option shares (%) on the blank image, and blank-image accuracy."""
    qs = [q for q in items.values() if q["axis"] == axis]
    c = collections.Counter(q["blank"] for q in qs)
    n = len(qs)
    acc = sum(q["blank"] == q["gold"] for q in qs) / n * 100 if n else float("nan")
    return {k: 100 * v / n for k, v in c.items()}, acc, n


def sources(pairs):
    """Share of position / content / neither answers per model and axis, on all questions and on the questions
    whose answer the real flip changes; blank-image accuracy and option shares."""
    for name, main, blank in pairs:
        for p in (main, blank):
            if not os.path.exists(p):
                raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
        items = index_sources(read_jsonl(main), read_jsonl(blank))
        print(f"\n== {name} ({len(items)} items)")
        for ax in ("lr", "ud"):
            for sub in (False, True):
                d, n = decompose(items, ax, sub)
                label = "changed by real flip" if sub else "all"
                print(f"  {ax} {label:20s} (n={n:3d}): position {d['position']:5.1f}%  content {d['content']:5.1f}%  neither {d['neither']:5.1f}%")
            bias, acc, n = blank_distribution(items, ax)
            print(f"  {ax} blank image: accuracy {acc:.1f}%, answers " + ", ".join(f"{k} {v:.1f}%" for k, v in sorted(bias.items())))


def _by_variant(path):
    by = collections.defaultdict(dict)
    for r in read_jsonl(path):
        by[r["idx"]][r["variant"]] = r
    return by


def _blank_orig_order(path):
    return {r["idx"]: r for r in read_jsonl(path) if r.get("order") in (None, "fwd")}


def error_types(triples):
    """Original errors that keep the same label when subject and reference are swapped (pair-prior errors), how
    many of them equal the blank-image answer, and the remaining direction errors."""
    for label, main, blank in triples:
        print(f"## {label}")
        by, bl = _by_variant(main), _blank_orig_order(blank)
        for ax in ("lr", "ud"):
            err = [i for i, v in by.items() if v["orig"]["axis"] == ax and v["orig"]["pred"] != v["orig"]["expected"] and i in bl]
            same = [i for i in err if by[i]["swap"]["pred"] == by[i]["orig"]["pred"]]
            blank_same = [i for i in same if bl[i]["pred"] == by[i]["orig"]["pred"]]
            print(f"  {ax} errors {len(err)}: pair-prior errors (same label after the swap) {len(same)} ({len(same) / max(len(err), 1) * 100:.0f}%), "
                  f"of which same answer as the blank image {len(blank_same)}; direction errors (label inverted after the swap) {len(err) - len(same)}")


def _groups(xs, k):
    if len(xs) % k:
        raise SystemExit(f"expected groups of {k} arguments")
    return [xs[i:i + k] for i in range(0, len(xs), k)]


M3, M7 = (f"results/main_table/qwen2.5-vl/qwen2.5-vl-{s}-instruct.jsonl" for s in ("3b", "7b"))
MI = "results/main_table/internvl3-2b/internvl3-2b-hf.jsonl"
B3, B7 = (f"results/decomposition/blank_image/blank_qwen2.5-vl-{s}-instruct.jsonl" for s in ("3b", "7b"))
BI = "results/decomposition/blank_image_fp32/blank_internvl3-2b-hf.jsonl"
_QWEN = ["--pair", "qwen3b", M3, B3, "--pair", "qwen7b", M7, B7]
_THREE = ["Qwen 3b", M3, B3, "Qwen 7b", M7, B7, "InternVL3-2B", MI, BI]
OUTPUTS = [
    ("outputs/decomposition/answer_sources.txt", ["sources", *_QWEN, "--pair", "internvl3-2b", MI, BI], True),
    ("outputs/decomposition/error_types.txt", ["error-types", *_THREE], True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.decomposition", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: blank-image condition on CV-Bench")
    r.add_argument("--family", choices=["qwen", "internvl"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True, help="CV-Bench test_2d.parquet")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="first N questions only")
    r.add_argument("--scoring", choices=["fp32", "logits"], default="fp32",
                   help="fp32: last hidden state x the two lm_head rows in fp32 (ties recorded); logits: bf16 logits")
    s = sub.add_parser("sources", help=sources.__doc__.split("\n")[0])
    s.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "MAIN", "BLANK"), required=True)
    for name, fn in (("error-types", error_types),):
        sub.add_parser(name, help=fn.__doc__.split("\n")[0]).add_argument("args", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a)
    elif a.cmd == "sources":
        sources(a.pair)
    else:
        error_types(_groups(a.args, 3))


if __name__ == "__main__":
    main()
