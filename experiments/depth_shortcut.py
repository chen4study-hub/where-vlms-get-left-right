"""Depth shortcut: does 'higher in the image means farther' run through the ViT height positions?
(paper Section 4.6, Table 5, Fig. 4b; Appendix D)

Qwen2.5-VL-3B / 7B answer the Depth questions of CV-Bench 3D (validation and held-out test half) and the near-far
questions of EmbSpatial-Bench under orig, vit_h / vit_w (ViT height / width positions mirrored), llm_h (LLM-side
height ids mirrored) and vflip (pixels flipped upside down). Option logits are computed in fp32 from the lm_head
input (vlmlr.models.FP32OptionScorer); the validation half is scored with the model's bf16 logits
(--scoring logits) and ties go to the first option.

Data preparation (CPU; images are resized once to the processor's input size and cached):
  python -m experiments.depth_shortcut items-cvbench3d --parquet test_3d.parquet --split test \\
      --out items_test.jsonl --img-dir img
  python -m experiments.depth_shortcut items-embspatial --json embspatial_bench.json \\
      --out emb_items.jsonl --img-dir emb_img
GPU runs (one model each; --limit N for the first N items):
  python -m experiments.depth_shortcut run-cvbench3d --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --items items_test.jsonl --out test_qwen2.5-vl-7b-instruct.jsonl
  python -m experiments.depth_shortcut run-embspatial --model ... --items emb_items.jsonl --out emb_7b.jsonl
Analyses: summary, embspatial-summary, specificity, embspatial-direction (see --help of each).
"""
import argparse
import base64
import collections
import io
import json
import math
import os
import time

from vlmlr import depth as D
from vlmlr.records import ResumableWriter, read_jsonl
from vlmlr.models import FP32OptionScorer
from vlmlr.stats import bootstrap

GROUPS = ("consistent", "counter", "all")
MIN_PIXELS, MAX_PIXELS = 56 * 56, 1280 * 28 * 28


# ---------------------------------------------------------------- data preparation (CPU)

def _resize_cached(im, path, min_pixels, max_pixels, atomic=False):
    from PIL import Image
    from vlmlr.geometry import smart_resize
    Hr, Wr = smart_resize(im.height, im.width, 28, min_pixels, max_pixels)
    if not os.path.exists(path):
        if atomic:
            tmp = path + ".tmp.png"
            im.resize((Wr, Hr), Image.BICUBIC).save(tmp)
            os.replace(tmp, path)
        else:
            im.resize((Wr, Hr), Image.BICUBIC).save(path)
    return Wr, Hr


def items_cvbench3d(a):
    """Depth questions of one split of CV-Bench 3D, with resized images and scaled boxes."""
    import pyarrow.parquet as pq
    from PIL import Image
    from vlmlr.geometry import scale_box
    rows = [r for r in pq.read_table(a.parquet).to_pylist() if r["task"] == "Depth"]
    os.makedirs(a.img_dir, exist_ok=True)
    out, cnt = [], {}
    for r in rows:
        if D.cvbench3d_split(r["idx"]) != a.split:
            continue
        assert r.get("bbox") and len(r["bbox"]) >= 2, f"idx {r['idx']}: missing boxes {r.get('bbox')}"
        img = r["image"]
        im = Image.open(io.BytesIO(img.get("bytes") if isinstance(img, dict) else img)).convert("RGB")
        W0, H0 = im.size
        grp = D.depth_group(r, H0)
        path = os.path.join(a.img_dir, f"{r['idx']}.png")
        Wr, Hr = _resize_cached(im, path, a.min_pixels, a.max_pixels)
        out.append({"idx": r["idx"], "task": r["task"], "question": r["question"], "choices": r["choices"],
                    "gold": D.gold_index(r), "group": grp, "img": path, "W": Wr, "H": Hr,
                    "bbox": [scale_box(b, Wr / W0, Hr / H0) for b in r["bbox"]],
                    "source_dataset": r.get("source_dataset"),
                    "higher": D.higher_index(r)})
        cnt[(r["task"], grp)] = cnt.get((r["task"], grp), 0) + 1
    with open(a.out, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[items] {a.split}: {len(out)} items -> {a.out}")
    for k in sorted(cnt):
        print(f"  {k[0]:8s} {k[1]:10s} {cnt[k]}")


def items_embspatial(a):
    """Near-far questions of EmbSpatial-Bench (streamed from the 253 MB JSON), grouped by the heuristic."""
    from PIL import Image
    os.makedirs(a.img_dir, exist_ok=True)
    cnt, n = {}, 0
    with open(a.out, "w") as f:
        for x in D.iter_json_array(a.json):
            if x.get("relation") not in ("close", "far"):
                continue
            g = D.embspatial_group(x)
            cnt[(x["relation"], g)] = cnt.get((x["relation"], g), 0) + 1
            if g is None:
                continue
            opts = D.as_list(x["answer_options"])
            b = D.embspatial_bottoms(x)
            im = Image.open(io.BytesIO(base64.b64decode(x["image"])))
            im.load()
            im = im.convert("RGB")
            path = os.path.join(a.img_dir, f"{x['question_id']}.png")
            Wr, Hr = _resize_cached(im, path, a.min_pixels, a.max_pixels, atomic=True)
            f.write(json.dumps({"idx": x["question_id"], "relation": x["relation"], "question": x["question"], "options": opts,
                                "gold": int(x["answer"]), "group": g, "heuristic": D.embspatial_heuristic(x["relation"], b),
                                "anti": D.embspatial_anti(x["relation"], b), "img": path, "W": Wr, "H": Hr},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"[emb items] close / far {sum(cnt.values())} items, {n} written -> {a.out}; groups {dict(sorted(cnt.items(), key=str))}"
          f" (None = option names do not match the boxes, dropped)")


# ---------------------------------------------------------------- GPU runs

class _Runner:
    """Model, processor and position mirror shared by the runs; scores options in fp32 (or bf16 logits)."""

    def __init__(self, a, letters):
        import torch
        from vlmlr import models
        from vlmlr.hooks import QwenPositionMirror
        self.torch = torch
        self.model, self.proc = models.load_qwen(a.model, max_pixels=a.max_pixels, min_pixels=a.min_pixels)
        tok = self.proc.tokenizer
        ids = [tok.encode(x, add_special_tokens=False) for x in letters]
        assert all(len(t) == 1 for t in ids) and len({t[0] for t in ids}) == len(ids), ids
        self.ids = [t[0] for t in ids]
        self.mirror = QwenPositionMirror(self.model)
        self.fp32 = FP32OptionScorer(self.model, self.ids)

    def inputs(self, img, text):
        from vlmlr.models import chat_text
        inp = self.proc(text=[chat_text(self.proc, text)], images=[img], return_tensors="pt").to("cuda")
        gh, gw = inp["image_grid_thw"][0, 1:].tolist()
        assert (gw * 14, gh * 14) == img.size, f"the processor resized the cached image ({img.size} vs grid {gh}x{gw})"
        return inp

    def forward(self, inp, vit, llm, scoring="fp32"):
        """Option logits (fp32 from the lm_head input, or the model's own logits) with the given mirrors."""
        m = self.mirror
        m.vit, m.llm = vit, llm
        self.model.model.rope_deltas = None
        try:
            out = self.model(**inp)
            return self.fp32() if scoring == "fp32" else out.logits[0, -1].float()[self.ids]
        finally:
            m.vit = m.llm = ""

    def self_check(self, inp, conds, scoring="fp32", log_probs=False):
        """Image not resized again; ViT ids re-created exactly; orig reproducible; every counterfactual changes the
        output; fp32 scores agree with the model's logits up to bf16 error (compared as log-probs if log_probs)."""
        torch = self.torch
        f = (lambda x: torch.log_softmax(x, -1)) if log_probs else (lambda x: x)
        self.mirror.self_check(inp)
        b1, b2 = f(self.forward(inp, "", "", scoring)), f(self.forward(inp, "", "", scoring))
        assert torch.equal(b1, b2), "orig is not reproducible"
        if scoring == "fp32":
            self.model.model.rope_deltas = None
            ref = f(self.model(**inp).logits[0, -1].float()[self.ids])
            assert (ref - b1).abs().max() < 0.25, f"fp32 scores far from the model logits: {b1.tolist()} vs {ref.tolist()}"
        for v, l in conds:
            assert not torch.equal(b1, f(self.forward(inp, v, l, scoring))), f"vit={v!r} llm={l!r} changed nothing"


def _image(a, it):
    from PIL import Image
    return Image.open(os.path.join(a.image_root, it["img"])).convert("RGB")


def run_cvbench3d(a):
    from PIL import ImageOps
    from vlmlr.cvbench import prompt
    R = _Runner(a, ("A", "B"))
    torch = R.torch
    items = read_jsonl(a.items)[: a.limit or None]
    conds = a.conds.split(",")
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("idx", "cond")) as out, torch.inference_mode():
        for it in items:
            im = _image(a, it)
            assert im.size == (it["W"], it["H"]), "cached image size differs from the item table"
            text = prompt(it["question"], it["choices"], False)
            if not checked:
                R.self_check(R.inputs(im, text), (("h", ""), ("w", ""), ("", "h")), a.scoring)
                checked = True
            for c in conds:
                if (it["idx"], c) in out:
                    continue
                vit, llm, flip = D.CONDITION_SPEC[c]
                lg = R.forward(R.inputs(ImageOps.flip(im) if flip else im, text), vit, llm, a.scoring)
                lp = torch.log_softmax(lg, -1).tolist()
                rec = {"idx": it["idx"], "task": it["task"], "group": it["group"], "cond": c, "gold": it["gold"]}
                if a.scoring == "fp32":
                    rec.update(higher=it.get("higher"), lp=lp, tie=lp[0] == lp[1])
                else:
                    rec["lp"] = lp
                out.write(rec)
            out.flush()
    print(f"[run] done {out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


def run_embspatial(a):
    from vlmlr.cvbench import prompt
    R = _Runner(a, "ABCD")
    torch = R.torch
    items = read_jsonl(a.items)[: a.limit or None]
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("idx", "cond")) as out, torch.inference_mode():
        for it in items:
            im = _image(a, it)
            inp_f = R.inputs(im, prompt(it["question"], it["options"], False))
            inp_r = R.inputs(im, prompt(it["question"], it["options"][::-1], False))
            if not checked:
                R.self_check(inp_f, (("h", ""), ("w", ""), ("", "h"), ("", "w")), log_probs=True)
                checked = True
            for c in a.conds.split(","):
                vit, llm, _ = D.CONDITION_SPEC[c]
                if (it["idx"], c) in out:
                    continue
                lf = torch.log_softmax(R.forward(inp_f, vit, llm), -1).tolist()
                lr = torch.log_softmax(R.forward(inp_r, vit, llm), -1).tolist()
                sc = D.average_orders(lf, lr)
                out.write({"idx": it["idx"], "relation": it["relation"], "group": it["group"], "cond": c,
                           "gold": it["gold"], "heuristic": it["heuristic"], "anti": it["anti"],
                           "lp_fwd": lf, "lp_rev": lr, "score": sc, "pred": D.argmax_or_none(sc),
                           "pred_fwd": D.argmax_or_none(lf)})
            out.flush()
    print(f"[run] done {2 * out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


# ---------------------------------------------------------------- analyses: summaries

def _analyze(recs, ties):
    bc = {}
    for r in recs:
        if r["task"] == "Depth":
            bc.setdefault(r["cond"], {})[r["idx"]] = r
    if "orig" not in bc:
        return None
    orig = bc["orig"]
    gold = {i: r["gold"] for i, r in orig.items()}
    ids = {g: [i for i, r in orig.items() if g == "all" or r["group"] == g] for g in GROUPS}
    P = {c: {i: D.predict(r["lp"], ties) for i, r in d.items()} for c, d in bc.items()}
    res = {"n": {g: len(ids[g]) for g in GROUPS}}
    res["acc_orig"] = {g: D.accuracy(P["orig"], gold, ids[g])[0] for g in GROUPS}
    res["flip"] = {c: D.change_rate(P["orig"], P[c]) for c in D.CONDITIONS[1:] if c in P}
    if ties == "separate":
        res["ties"] = {c: D.count_ties(p) for c, p in P.items()}
    hi = {i: r.get("higher") for i, r in orig.items()}
    if "vit_h" in P and all(v is not None for v in hi.values()):
        nonamb = [i for i in orig if orig[i]["group"] != "ambiguous"]
        res["direction"] = {c: D.direction_share(P["orig"], P[c], hi, nonamb) for c in ("vit_h", "vit_w", "llm_h") if c in P}
    res["gap"] = res["acc_orig"]["consistent"] - res["acc_orig"]["counter"]
    return res


def summary(a):
    """CV-Bench 3D Depth: change rates, accuracy by group and direction of the changes."""
    allres = {}
    for size, path in a.run:
        recs = read_jsonl(path)
        if a.items:
            hi = {it["idx"]: it["higher"] if it.get("higher") is not None else D.higher_index(it) for it in read_jsonl(a.items)}
            for r in recs:
                r.setdefault("higher", hi.get(r["idx"]))
        allres[size] = _analyze(recs, a.ties)
    for size, r in allres.items():
        if not r:
            print(f"== {size}: no results")
            continue
        print(f"== {size}  items consistent {r['n']['consistent']} / counter {r['n']['counter']} / all {r['n']['all']}")
        print("  orig accuracy: " + ", ".join(f"{g} {r['acc_orig'][g]:.1f}" for g in GROUPS) + f" (consistent − counter = {r['gap']:.1f})")
        if "ties" in r:
            print("  ties (equal log-probabilities of the two options, counted apart; accuracy 0.5, not in the change rates): " +
                  ", ".join(f"{c} {k}" for c, k in r["ties"].items()))
        for c, (fr, n, (lo, hi)) in r["flip"].items():
            print(f"  {c:6s} change rate {fr:5.1f}%  [{lo:.1f}, {hi:.1f}]  n={n}")
        for c, (sh, n, (lo, hi)) in r.get("direction", {}).items():
            print(f"  {c:6s} changed (non-ambiguous) items that now call the object higher in the image closer: {sh:5.1f}%  [{lo:.1f}, {hi:.1f}]  changed {n} items")
    if a.json:
        json.dump(allres, open(a.json, "w"), ensure_ascii=False, indent=1)


def _load_runs(runs):
    out = []
    for size, path in runs:
        by = {}
        for r in read_jsonl(path):
            by.setdefault(r["idx"], {})[r["cond"]] = r
        out.append((size, by))
    return out


def embspatial_summary(a):
    """EmbSpatial near-far: accuracy by group, errors fixed / correct answers harmed by each counterfactual,
    moves to the opposite end, and the change-rate split by heuristic answers (post hoc)."""
    res = {}
    for size, by in _load_runs(a.run):
        print(f"\n== {size}")
        out = {"ties": {c: sum(d[c]["pred"] is None for d in by.values() if c in d) for c in D.CONDITIONS}}
        for g in ("consistent", "counter"):
            o = [d["orig"] for d in by.values() if "orig" in d and d["orig"]["group"] == g]
            acc = 100 * sum(0.25 if r["pred"] is None else float(r["pred"] == r["gold"]) for r in o) / len(o) if o else float("nan")
            accf = 100 * sum(r["pred_fwd"] == r["gold"] for r in o) / len(o) if o else float("nan")
            out[f"acc_{g}"] = acc
            print(f"  {g:10s}: {len(o)} items, orig accuracy {acc:.1f}% (decided on the mean of both orders; original order only {accf:.1f}%)")
        sub = {i: d for i, d in by.items() if d.get("orig", {}).get("group") == "counter"}
        fr = D.fix_rates(sub)
        print(f"  ties (not counted among errors or in the rates): {out['ties']}")
        print(f"  counter group: orig wrong {fr['n_err']}, correct {fr['n_ok']}")
        for c in ("vit_h", "vit_w", "llm_h"):
            f_, n, (lo, hi) = fr[c]["fix"]
            ch = [d for d in sub.values() if c in d and d[c]["pred"] is not None and d["orig"]["pred"] is not None
                  and d[c]["pred"] != d["orig"]["pred"]]
            anti = 100 * sum(d[c]["pred"] == d[c]["anti"] for d in ch) / len(ch) if ch else float("nan")
            print(f"  {c:6s}: errors now correct {f_:5.1f}% [{lo:.1f}, {hi:.1f}] (n={n}); correct answers broken {fr[c]['harm'][0]:.1f}%; "
                  f"of the {len(ch)} changed items, moved to the opposite end of the heuristic {anti:.1f}%")
            fr[c]["anti_share"] = anti
        db = D.embspatial_debias(by)
        for c in ("vit_h", "vit_w", "llm_h"):
            x = db[c]
            print(f"  baseline split (post hoc) {c:6s}: change rate, heuristic-following group {x['follow'][2]:.1f}% [{x['follow'][3][0]:.1f}, {x['follow'][3][1]:.1f}] (n={x['follow'][1]}) | "
                  f"opposing group {x['against'][2]:.1f}% [{x['against'][3][0]:.1f}, {x['against'][3][1]:.1f}] (n={x['against'][1]})")
        out["debias"] = {c: {"follow": db[c]["follow"][:3], "against": db[c]["against"][:3]} for c in db}
        res[size] = {**out, **fr}
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- analyses: post hoc (CV-Bench 3D)

def _wilson_text(k, n, z=1.96, scaled=False):
    """'xx.x% [lo,hi] n=N' with one decimal ('n=0' for n = 0); scaled=True prints 100*k/n instead of 100*p."""
    if n == 0:
        return "n=0"
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"{100 * k / n if scaled else 100 * p:5.1f}% [{100 * (c - h) / d:.1f},{100 * (c + h) / d:.1f}] n={n}"


def _by_task_item(path):
    R = collections.defaultdict(dict)
    for r in read_jsonl(path):
        R[(r["task"], r["idx"])][r["cond"]] = r
    return R


def _pr(r):
    return int(r["lp"][1] > r["lp"][0])


def _tie(r):
    return r["lp"][0] == r["lp"][1]


def _heuristic(v):
    g = v["orig"]["gold"]
    return g if v["orig"]["group"] == "consistent" else 1 - g


def specificity(a):
    """Depth items split by whether the orig answer follows the heuristic: change rates of each condition in
    both halves (ties and ambiguous items removed). A height shortcut changes heuristic answers only."""
    for f in a.files:
        R = _by_task_item(f)
        print("==", f.split("/")[-1])
        for c in ["vit_h", "vit_w", "llm_h"]:
            k = collections.Counter()
            for (t, i), v in R.items():
                if t != "Depth" or v["orig"]["group"] not in ("consistent", "counter") or c not in v or _tie(v["orig"]) or _tie(v[c]):
                    continue
                side = "follow" if _pr(v["orig"]) == _heuristic(v) else "against"
                k[(side, "n")] += 1
                k[(side, "ch")] += _pr(v[c]) != _pr(v["orig"])
            print(f"  {c:6s} change rate, heuristic-following group {_wilson_text(k[('follow', 'ch')], k[('follow', 'n')])} | "
                  f"opposing group {_wilson_text(k[('against', 'ch')], k[('against', 'n')])}")


# ---------------------------------------------------------------- analyses: post hoc (EmbSpatial)

def embspatial_direction(a):
    """Items whose orig answer is the heuristic one: share moving to the opposite end under vit_h, vit_w and llm_h,
    and vit_h - vit_w with a paired bootstrap interval (10000 resamples of these items, seed 0)."""
    for p in a.files:
        by = collections.defaultdict(dict)
        for r in read_jsonl(p):
            by[r["idx"]][r["cond"]] = r
        ids = sorted(i for i, d in by.items() if all(c in d for c in ("orig", "vit_h", "vit_w", "llm_h")))
        print(f'== {p.split("/")[-1]}  n={len(ids)}')
        fol = [i for i in ids if by[i]["orig"]["pred"] == by[i]["orig"]["heuristic"]]
        x = {c: [int(by[i][c]["pred"] == by[i][c]["anti"]) for i in fol] for c in ("vit_h", "vit_w", "llm_h")}
        for c in x:
            print(f"  heuristic-following items (n={len(fol)}) moved to the opposite end under {c} {sum(x[c]) / len(fol) * 100:5.1f}%")
        diffs = bootstrap(lambda s: sum(x["vit_h"][k] - x["vit_w"][k] for k in s) / len(fol), len(fol))
        d = (sum(x["vit_h"]) - sum(x["vit_w"])) / len(fol)
        print(f"  vit_h − vit_w = {d * 100:+.1f} points [{diffs[250] * 100:+.1f}, {diffs[9749] * 100:+.1f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.depth_shortcut", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sizes(p):
        p.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
        p.add_argument("--max-pixels", type=int, default=MAX_PIXELS)

    p = sub.add_parser("items-cvbench3d", help="CPU: CV-Bench 3D items of one split")
    p.add_argument("--parquet", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--img-dir", required=True)
    sizes(p)
    p = sub.add_parser("items-embspatial", help="CPU: EmbSpatial-Bench near-far items")
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img-dir", required=True)
    sizes(p)
    for name, conds in (("run-cvbench3d", ",".join(D.CONDITIONS)), ("run-embspatial", ",".join(D.CONDITIONS))):
        p = sub.add_parser(name, help="GPU: " + name[4:])
        p.add_argument("--model", required=True)
        p.add_argument("--items", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--image-root", default="", help="directory the items' relative image paths start from")
        p.add_argument("--conds", default=conds, help="comma-separated; CV-Bench 3D also has vflip")
        if name == "run-cvbench3d":
            p.add_argument("--scoring", choices=["fp32", "logits"], default="fp32",
                           help="logits: the model's bf16 logits (used for the validation half)")
        sizes(p)
    for name, help_ in (("summary", "CV-Bench 3D Depth summary"), ("embspatial-summary", "EmbSpatial summary")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--run", nargs=2, action="append", metavar=("SIZE", "FILE"), required=True)
        p.add_argument("--json")
        if name == "summary":
            p.add_argument("--items", help="item table, to add the 'higher' field to records that lack it")
            p.add_argument("--test", action="store_true", help="held-out test half (as run, with ties kept separate)")
            p.add_argument("--ties", choices=["separate", "first"], default="separate",
                           help="first: ties go to option A (rule used for the validation half)")
    for name in ("specificity", "embspatial-direction"):
        sub.add_parser(name).add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    {"items-cvbench3d": items_cvbench3d, "items-embspatial": items_embspatial,
     "run-cvbench3d": run_cvbench3d, "run-embspatial": run_embspatial,
     "summary": summary, "embspatial-summary": embspatial_summary,
     "specificity": specificity, "embspatial-direction": embspatial_direction}[a.cmd](a)


_V = "results/depth_shortcut/cvbench3d_val/qwen2.5-vl-{}-instruct.jsonl"
_T = "results/depth_shortcut/cvbench3d_test/test_qwen2.5-vl-{}-instruct.jsonl"
_E = "results/depth_shortcut/embspatial/emb_{}.jsonl"
# (output file, arguments of the analysis command that writes it, behind a table, figure or the main text)
OUTPUTS = [
    ("outputs/depth_shortcut/cvbench3d_val_summary.txt",
     ["summary", "--ties", "first", "--run", "3B", _V.format("3b"), "--run", "7B", _V.format("7b")], False),
    ("outputs/depth_shortcut/cvbench3d_test_summary.txt",
     ["summary", "--test", "--run", "3B", _T.format("3b"), "--run", "7B", _T.format("7b")], True),
    ("outputs/depth_shortcut/specificity.txt",
     ["specificity", _V.format("3b"), _V.format("7b"), _T.format("3b"), _T.format("7b")], True),
    ("outputs/depth_shortcut/embspatial_summary.txt",
     ["embspatial-summary", "--run", "3B", _E.format("3b"), "--run", "7B", _E.format("7b")], False),
    ("outputs/depth_shortcut/embspatial_direction.txt", ["embspatial-direction", _E.format("3b"), _E.format("7b")], True),
]


if __name__ == "__main__":
    main()
