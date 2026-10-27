"""What the model reads from a single printed word when only the ViT width positions are mirrored
(Qwen2.5-VL; paper Section 4.7 and Appendix B).

100 synthetic word images (448 x 224, not resized) under orig / vit_w / hflip; greedy generation of up to 8 tokens,
judged as correct / reversed / other. The post hoc prefix analysis compares the beginning of each reading with the
word spelled backwards and forwards, and with another word spelled backwards as a control.

  python -m experiments.extra word_reading make-data --out data/word_reading
  python -m experiments.extra word_reading run --model Qwen/Qwen2.5-VL-7B-Instruct --data data/word_reading \\
      --out results/extra/word_reading/qwen2.5-vl-7b.jsonl
  python -m experiments.extra word_reading summary --m7b <7B.jsonl> [--m3b <3B.jsonl>]
  python -m experiments.extra word_reading prefix <predictions.jsonl> ...      # post hoc: reversed-prefix analysis
"""
import argparse
import json
import os
import random
import re
import time

from vlmlr.records import ResumableWriter, read_jsonl

# 100 common English words of 5-8 letters; none is a palindrome or reads as a common word backwards
WORDS = """garden window silver planet market winter rocket bridge castle forest
yellow orange purple button candle doctor engine farmer guitar hammer
island jacket kitten ladder magnet napkin office pencil rabbit saddle
tunnel violin wallet zipper basket camera dinner flower helmet insect
jungle kettle lemon monkey needle pepper puzzle ribbon shadow tomato
ticket turkey velvet wizard anchor bottle carpet desert finger goblet
harbor lantern meadow orchard pillow quartz sponge throne umbrella vessel
walnut blanket compass dolphin feather glacier horizon journal kingdom lobster
mansion network pumpkin rainbow sandwich teacher uniform volcano whistle balloon
captain diamond elephant fountain gallery hospital library mountain notebook painter""".split()
assert len(WORDS) == 100 and len(set(WORDS)) == 100 and all(w != w[::-1] for w in WORDS)

W, H = 448, 224                 # multiples of 28: the processor does not resize
FONT_SIZE = 56
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans.ttf"]
READ_PROMPT = "What word is written in the image? Answer with the word only."
CONDS = ("orig", "vit_w", "hflip")

# ---------------------------------------------------------------- items

def make_word_items(seed=0):
    rng = random.Random(seed)
    return [{"idx": f"w{i}", "word": w, "dx": rng.randint(-40, 40), "dy": rng.randint(-20, 20)} for i, w in enumerate(WORDS)]


def render_word(item, font_path):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_path, FONT_SIZE)
    l, t, r, b = dr.textbbox((0, 0), item["word"], font=font)
    x = (W - (r - l)) // 2 - l + item["dx"]
    y = (H - (b - t)) // 2 - t + item["dy"]
    dr.text((x, y), item["word"], fill=(0, 0, 0), font=font)
    return im


def make_data(a):
    assert a.font, "no font found; pass --font with a .ttf file"
    items = make_word_items()
    os.makedirs(os.path.join(a.out, "words"), exist_ok=True)
    with open(os.path.join(a.out, "words.jsonl"), "w") as f:
        for it in items:
            render_word(it, a.font).save(os.path.join(a.out, "words", f"{it['idx']}.png"))
            f.write(json.dumps({**it, "font": os.path.basename(a.font)}) + "\n")
    print(f"[data] {len(items)} word images -> {a.out}/words/ (font {a.font})")


# ---------------------------------------------------------------- metrics

def _words(s):
    return re.sub(r"[^a-z]", " ", s.lower()).split()


def judge_read(output, word):
    toks = _words(output)
    if word in toks:
        return "correct"
    if word[::-1] in toks:
        return "reversed"
    return "other"


def read_acc(recs, cond):
    xs = [r for r in recs if r["task"] == "read" and r["cond"] == cond]
    if not xs:
        return float("nan"), float("nan"), 0
    return (sum(r["judge"] == "correct" for r in xs) / len(xs) * 100,
            sum(r["judge"] == "reversed" for r in xs) / len(xs) * 100, len(xs))


# ---------------------------------------------------------------- GPU run

def run(a):
    import torch
    from PIL import Image, ImageOps
    from vlmlr import models
    from vlmlr.models import greedy_text
    from vlmlr.hooks import QwenPositionMirror
    model, proc = models.load_qwen(a.model)
    tok = proc.tokenizer
    fl = QwenPositionMirror(model)

    def read(inp, vit):
        fl.vit = vit
        try:
            return greedy_text(model, tok, inp, 8).strip()
        finally:
            fl.vit = ""

    words = read_jsonl(os.path.join(a.data, "words.jsonl"))
    if a.limit:
        words = words[: a.limit]
    t0 = time.time()
    with ResumableWriter(a.out, ("task", "idx", "cond")) as out, torch.inference_mode():
        out.done = {(t, str(i), c) for t, i, c in out.done}
        img = Image.open(os.path.join(a.data, "words", f"{words[0]['idx']}.png")).convert("RGB")
        inp = models.build_inputs(proc, img, READ_PROMPT)
        fl.self_check(inp)
        assert inp["image_grid_thw"].tolist() == [[1, H // 14, W // 14]], f"word image was resized: {inp['image_grid_thw'].tolist()}"
        b1, b2, bw = fl.forward_logits(model, inp), fl.forward_logits(model, inp), fl.forward_logits(model, inp, vit="w")
        assert torch.equal(b1, b2) and not torch.equal(b1, bw), "orig not reproducible, or vit_w changed nothing"
        for w in words:
            img0 = Image.open(os.path.join(a.data, "words", f"{w['idx']}.png")).convert("RGB")
            for c in CONDS:
                if ("read", w["idx"], c) in out:
                    continue
                img = ImageOps.mirror(img0) if c == "hflip" else img0
                text = read(models.build_inputs(proc, img, READ_PROMPT), "w" if c == "vit_w" else "")
                out.write({"task": "read", "idx": w["idx"], "cond": c, "word": w["word"], "output": text,
                           "judge": judge_read(text, w["word"])})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


# ---------------------------------------------------------------- analyses

def summary(a):
    out = {}
    for tag, p in (("7B", a.m7b), ("3B", a.m3b)):
        if not p:
            continue
        recs = read_jsonl(p)
        print(f"\n== {tag}")
        for c in CONDS:
            acc, rev, n = read_acc(recs, c)
            print(f"  reading {c:6s}: correct {acc:5.1f}%  reversed {rev:5.1f}%  (n={n})")
        for r in [r for r in recs if r["task"] == "read" and r["cond"] == "vit_w" and r["judge"] != "correct"][:8]:
            print(f"    vit_w misread: {r.get('word')} → {r.get('output')!r} ({r['judge']})")
        out[tag] = {c: dict(zip(("correct", "reversed", "n"), read_acc(recs, c))) for c in CONDS}
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def prefix(a):
    """Post hoc: longest common prefix of each output with the reversed and with the forward spelling of the word;
    control: the reversed spelling of a random other word (seed 0)."""
    for p in a.files:
        rows = [r for r in read_jsonl(p) if r.get("task") == "read"]
        words = sorted({r["word"] for r in rows})
        rnd = random.Random(0)
        print(f'== {p.split("/")[-1]}')
        for cond in ("orig", "vit_w", "hflip"):
            out = [(re.sub("[^a-z]", "", r["output"].lower()), r["word"]) for r in rows if r["cond"] == cond]
            rev = [_common_prefix(o, w[::-1]) for o, w in out]
            fwd = [_common_prefix(o, w) for o, w in out]
            ctl = [_common_prefix(o, rnd.choice([x for x in words if x != w])[::-1]) for o, w in out]
            n = len(out)

            def f(xs, k):
                return sum(x >= k for x in xs) / n * 100
            print(f"  {cond:6s} n={n}: prefix of the reversed word ≥2 {f(rev, 2):5.1f}%  ≥3 {f(rev, 3):5.1f}% | prefix of the word ≥2 {f(fwd, 2):5.1f}%  ≥3 {f(fwd, 3):5.1f}% | "
                  f"control (another word reversed) ≥2 {f(ctl, 2):5.1f}%  ≥3 {f(ctl, 3):5.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.extra word_reading", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-data", help="draw the 100 word images")
    m.add_argument("--font", default=next((f for f in FONTS if os.path.exists(f)), None))
    m.add_argument("--out", default="data/word_reading")
    r = sub.add_parser("run", help="GPU: read the words under orig, vit_w, hflip")
    r.add_argument("--model", required=True)
    r.add_argument("--data", default="data/word_reading")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary", help="reading accuracy under each condition")
    s.add_argument("--m7b", required=True)
    s.add_argument("--m3b")
    s.add_argument("--json")
    p = sub.add_parser("prefix", help="post hoc reversed-prefix analysis")
    p.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    {"make-data": make_data, "run": run, "summary": summary, "prefix": prefix}[a.cmd](a)


R = "results/extra/word_reading/"
OUTPUTS = [
    ("outputs/extra/word_reading.txt", ["summary", "--m7b", R + "qwen2.5-vl-7b.jsonl", "--m3b", R + "qwen2.5-vl-3b.jsonl"]),
    ("outputs/extra/word_reading_reversed_prefix.txt", ["prefix", R + "qwen2.5-vl-3b.jsonl", R + "qwen2.5-vl-7b.jsonl"]),
]
