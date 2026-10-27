"""Referring localisation with "left" / "right" under position-only mirrors (Qwen2.5-VL; paper Section 4.7
and Appendix B, "Error analyses").

RefCOCO validation expressions containing a spatial left / right; the model outputs a box (greedy generation).
The evaluation copy only has the target box, so the real flip serves as reference: A = the target (only items
where orig finds it), B = the box predicted on the hflip image mapped back to the original coordinates, i.e. the
object on the other side. Items are usable if A and B are different objects (IoU < 0.5) and B is not the mirror
position of A. The boxes under vit_w / llm_w are classified by `python -m experiments.further_checks
box-reclassification`.

  python -m experiments.extra box_readout items --out data/box_readout/items.jsonl --img-dir <dir>   # RefCOCO, images pre-resized
  python -m experiments.extra box_readout run --model Qwen/Qwen2.5-VL-7B-Instruct --items data/box_readout/items.jsonl \\
      --image-root <dir> --out results/extra/box_readout/qwen2.5-vl-7b.jsonl
"""
import argparse
import io
import json
import os
import random
import re
import time

from vlmlr import geometry
from vlmlr.records import ResumableWriter, read_jsonl

SPATIAL = re.compile(r"\b(left|right|leftmost|rightmost|left-most|right-most)\b", re.I)
NON_SPATIAL = re.compile(r"\b(left\s+(hand|arm|leg|foot|eye|ear|side of (his|her|the) (face|body))|right\s+(hand|arm|leg|foot|eye|ear))\b|\bright\s+(now|away)\b", re.I)
CONDS = ("orig", "vit_w", "hflip", "llm_w")
THR = 0.5
PROMPT = 'Locate "{ref}" in the image, and output its bbox coordinates in JSON format.'


def is_spatial(ref):
    """The expression uses left / right as a direction (not 'left hand', 'right now')."""
    return bool(SPATIAL.search(ref)) and not (NON_SPATIAL.search(ref) and len(SPATIAL.findall(ref)) == 1)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mirror(box, W):
    if box is None:
        return None
    return [W - box[2], box[1], W - box[0], box[3]]


def item_status(d, gt, W):
    """'orig_wrong' / 'no_switch_ref' (hflip found the same object) / 'ambiguous' (B at A's mirror position) / 'ok'."""
    A = gt
    if iou(d.get("orig"), gt) < THR:
        return "orig_wrong", A, None
    B = mirror(d.get("hflip"), W)
    if B is None or iou(A, B) >= THR:
        return "no_switch_ref", A, B
    if iou(B, mirror(A, W)) >= THR:
        return "ambiguous", A, B
    return "ok", A, B


def items(a):
    """Select RefCOCO expressions with a spatial left / right, pre-resize the images by the processor rule
    (256-1024 tokens) and scale the target boxes accordingly."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image
    d = snapshot_download(a.repo, repo_type="dataset", local_files_only=True)
    files = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in sorted(fs) if f.endswith(".parquet") and a.split in os.path.join(r, f)]
    if not files:
        raise SystemExit(f"no parquet containing '{a.split}' in {a.repo}")
    rows = []
    for p in files:
        rows += pq.read_table(p).to_pylist()
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    os.makedirs(a.img_dir, exist_ok=True)
    n, seen = 0, 0
    with open(a.out, "w") as f:
        for r in rows:
            ref = r.get("answer") or r.get("sentences") or r.get("sentence") or r.get("refexp")
            if isinstance(ref, list):
                ref = ref[0] if ref else None
            if isinstance(ref, dict):
                ref = ref.get("raw") or ref.get("sent")
            if not ref or not is_spatial(str(ref)) or not r.get("bbox") or r.get("image") is None:
                continue
            seen += 1
            img = r["image"]
            im = Image.open(io.BytesIO(img.get("bytes") if isinstance(img, dict) else img)).convert("RGB")
            W0, H0 = im.size
            box0 = geometry.xywh_to_xyxy([float(v) for v in r["bbox"]])
            if (box0[2] - box0[0]) * (box0[3] - box0[1]) < a.min_area_frac * W0 * H0:
                continue
            Hr, Wr = geometry.smart_resize(H0, W0, a.factor, a.min_pixels, a.max_pixels)
            path = os.path.join(a.img_dir, f"{n:05d}.png")
            im.resize((Wr, Hr), Image.BICUBIC).save(path)
            f.write(json.dumps({"id": f"refcoco/{r.get('question_id', n)}", "img": path, "ref": str(ref).strip(),
                                "gt": geometry.scale_box(box0, Wr / W0, Hr / H0), "W": Wr, "H": Hr}, ensure_ascii=False) + "\n")
            n += 1
            if n >= a.n:
                break
    print(f"[items] expressions with left / right: {seen} (scanned up to item {n}), {n} items taken -> {a.out}")


def run(a):
    import torch
    from PIL import Image, ImageOps
    from vlmlr import models
    from vlmlr.models import greedy_text
    from vlmlr.hooks import QwenPositionMirror
    model, proc = models.load_qwen(a.model, max_pixels=int(a.max_pixels * 1.2), min_pixels=a.min_pixels)
    fl = QwenPositionMirror(model)
    todo = read_jsonl(a.items)[: a.limit or None]

    def inputs(img, ref):
        inp = models.build_inputs(proc, img, PROMPT.format(ref=ref))
        gh, gw = inp["image_grid_thw"][0, 1:].tolist()
        assert (gw * 14, gh * 14) == img.size, f"the processor resized the image ({img.size} vs grid {gh}x{gw})"
        return inp

    def generate(inp, vit, llm):
        fl.vit, fl.llm = vit, llm
        try:
            return greedy_text(model, proc.tokenizer, inp, a.max_new)
        finally:
            fl.vit = fl.llm = ""

    spec = {"orig": ("", "", False), "vit_w": ("w", "", False), "hflip": ("", "", True), "llm_w": ("", "w", False)}
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("id", "cond")) as out, torch.inference_mode():
        for it in todo:
            path = os.path.join(a.image_root, os.path.basename(it["img"])) if a.image_root else it["img"]
            im = Image.open(path).convert("RGB")
            assert im.size == (it["W"], it["H"]), "cached image size differs from the item list"
            if not checked:
                inp = inputs(im, it["ref"])
                fl.self_check(inp)
                b1, b2 = fl.forward_logits(model, inp), fl.forward_logits(model, inp)
                assert torch.equal(b1, b2), "orig first-step logits not reproducible"
                for v, l in (("w", ""), ("", "w")):
                    assert not torch.equal(b1, fl.forward_logits(model, inp, llm=l, vit=v)), f"vit={v!r} llm={l!r} changed nothing"
                checked = True
            for c, (vit, llm, flip) in spec.items():
                if (it["id"], c) in out:
                    continue
                text = generate(inputs(ImageOps.mirror(im) if flip else im, it["ref"]), vit, llm)
                out.write({"id": it["id"], "cond": c, "ref": it["ref"], "text": text[:300], "box": geometry.parse_box(text),
                           "gt": it["gt"], "W": it["W"], "H": it["H"]})
            out.flush()
    print(f"[run] done {out.n} gens in {time.time() - t0:.0f}s -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.extra box_readout", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items", help="select RefCOCO items and cache pre-resized images")
    p.add_argument("--repo", default="lmms-lab-encoder/RefCOCO")
    p.add_argument("--split", default="val")
    p.add_argument("--out", default="data/box_readout/items.jsonl")
    p.add_argument("--img-dir", default="data/box_readout/img")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--min-area-frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--factor", type=int, default=28)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    p = sub.add_parser("run", help="GPU: greedy box generation under orig, vit_w, hflip, llm_w")
    p.add_argument("--model", required=True)
    p.add_argument("--items", default="data/box_readout/items.jsonl")
    p.add_argument("--image-root", help="directory with the cached images (default: the paths in the item list)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    a = ap.parse_args(argv)
    {"items": items, "run": run}[a.cmd](a)


OUTPUTS = []      # the box classes are computed by experiments.further_checks box-reclassification
