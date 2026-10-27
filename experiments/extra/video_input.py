"""The left-right mechanism of Qwen2.5-VL-7B in video input mode (paper Section 4.7).

Each CV-Bench validation image (187 questions) is repeated as a 4-frame video (temporal patch 2, so t = 2 in the
ViT grid) and asked the same question under orig, vit_w / vit_h (ViT 2D-RoPE width / height ids mirrored),
llm_w (LLM M-RoPE width component of the video tokens mirrored) and hflip / vflip (every frame flipped).
Change rates are compared with the image-mode predictions of the main table on the same questions.

  python -m experiments.extra video_input run --model Qwen/Qwen2.5-VL-7B-Instruct --parquet data/cvbench/test_2d.parquet \\
      --out results/extra/video_input/qwen2.5-vl-7b.jsonl
  python -m experiments.extra video_input summary --m7b <video 7B> --img7b <main-table 7B>
"""
import argparse
import json
import time

from experiments.extra._shared import load_cvbench_validation
from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, read_jsonl

N_FRAMES = 4
CONDS = {  # condition -> (ViT mirrored axes, LLM mirrored axes, pixel transform)
    "orig": ("", "", None),
    "vit_w": ("w", "", None),
    "llm_w": ("", "w", None),
    "vit_h": ("h", "", None),
    "hflip": ("", "", "hflip"),
    "vflip": ("", "", "vflip"),
}
REAL = {"lr": "hflip", "ud": "vflip"}


def _by(recs, key="cond"):
    out = {}
    for r in recs:
        out.setdefault(r["idx"], {})[r[key]] = r
    return out


def change_rate(recs, cond, axis, key="cond", consistent=True):
    """Share (%) of items of `axis` whose answer under cond differs from orig; consistent=True keeps only items
    whose answer changes under the real flip."""
    rs = []
    for d in _by(recs, key).values():
        if "orig" not in d or cond not in d or d["orig"]["axis"] != axis:
            continue
        if consistent:
            real = d.get(REAL[axis])
            if real is None or real["pred"] == d["orig"]["pred"]:
                continue
        rs.append(d[cond]["pred"] != d["orig"]["pred"])
    return (100 * sum(rs) / len(rs) if rs else float("nan")), len(rs)


def run(a):
    import torch
    from transformers import AutoProcessor
    from vlmlr import models
    from vlmlr.hooks import QwenPositionMirror
    from vlmlr.positions import mirror_mrope_positions
    model, _ = models.load_qwen(a.model)
    proc = AutoProcessor.from_pretrained(a.model)          # default processor settings for video input
    ids = models.letter_ids(proc.tokenizer, "AB")
    fl = QwenPositionMirror(model)
    fl.image_token_id = model.config.video_token_id      # the LLM-side mirror acts on the video tokens

    def inputs(img, text):
        chat = proc.apply_chat_template([{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]}],
                                        add_generation_prompt=True)
        return proc(text=[chat], videos=[[img] * N_FRAMES], return_tensors="pt").to("cuda")

    def fwd(inp, cond):
        vit, llm, _ = CONDS[cond]
        return fl.forward_logits(model, inp, llm=llm, vit=vit)

    items = load_cvbench_validation(a.parquet)
    items = items[: a.limit] if a.limit else items
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("idx", "cond")) as out, torch.inference_mode():
        for it in items:
            img0 = C.to_image(it["image"])
            rel = C.gold_relation(it)
            axis = C.AXIS[rel]
            choices = C.CHOICES[axis]
            text = C.prompt(it["question"], choices, False)
            cache = {}
            for c, (_, _, tf) in CONDS.items():
                if (it["idx"], c) in out:
                    continue
                if tf not in cache:
                    cache[tf] = inputs(C.transform(img0, tf), text)
                inp = cache[tf]
                if not checked:
                    thw = inp["video_grid_thw"]
                    assert "pixel_values_videos" in inp and int((inp["input_ids"] == model.config.image_token_id).sum()) == 0
                    assert int(thw[0, 0]) == N_FRAMES // 2, f"video_grid_thw {thw.tolist()}: t is not {N_FRAMES // 2}"
                    assert torch.equal(fl.orig_rot_pos_emb(thw), fl.vit_rotary(thw, "")), "re-created ViT ids differ"
                    base = fwd(inp, "orig")
                    assert torch.equal(base, fwd(inp, "orig")), "orig not reproducible"
                    for cc in ("vit_w", "llm_w"):
                        assert not torch.equal(base, fwd(inp, cc)), f"{cc} changed nothing"
                    pos, _ = fl.orig_rope_index(inp["input_ids"], None, thw, inp.get("second_per_grid_ts"), inp["attention_mask"])
                    flp = mirror_mrope_positions(pos.clone(), inp["input_ids"], fl.image_token_id, "w")
                    assert flp.max() == pos.max() and not torch.equal(flp, pos), "LLM-side video position mirror self-check failed"
                    checked = True
                lp = models.option_logprobs(fwd(inp, c), ids)
                out.write({"idx": it["idx"], "cond": c, "axis": axis, "gold": rel, "choices": choices,
                           "pred": models.pick(choices, lp), "logp": lp, "source": it["source"]})
            out.flush()
    print(f"done: {out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


def summary(a):
    out = {}
    for tag, vp, ip in (("7B", a.m7b, a.img7b),):
        video = read_jsonl(vp)
        ids = {r["idx"] for r in video}
        image = [r for r in read_jsonl(ip) if r["idx"] in ids]
        print(f"\n== {tag} (video mode, {len(ids)} items)")
        for ax, conds in (("lr", ("vit_w", "llm_w")), ("ud", ("vit_h",))):
            for c in conds:
                v, nv = change_rate(video, c, ax)
                va, na = change_rate(video, c, ax, consistent=False)
                i, ni = change_rate(image, c, ax, key="variant")
                print(f"  {ax} {c:6s}: video geometry-consistent {v:5.1f}% (n={nv}), all {va:5.1f}% (n={na}) | image geometry-consistent {i:5.1f}% (n={ni})")
            acc_v = sum(r["pred"] == r["gold"] for r in video if r["cond"] == "orig" and r["axis"] == ax)
            n_v = sum(1 for r in video if r["cond"] == "orig" and r["axis"] == ax)
            print(f"  {ax} video orig accuracy {100 * acc_v / max(n_v, 1):.1f}% (n={n_v})")
        out[tag] = {f"{ax}_{c}": change_rate(video, c, ax) for ax, cs in (("lr", ("vit_w", "llm_w")), ("ud", ("vit_h",))) for c in cs}
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.extra video_input", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="GPU: six conditions in video mode on the CV-Bench validation split")
    r.add_argument("--model", required=True)
    r.add_argument("--parquet", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("summary", help="change rates in video vs image mode")
    for k in ("m7b", "img7b"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--json")
    a = ap.parse_args(argv)
    {"run": run, "summary": summary}[a.cmd](a)


R, Q = "results/extra/video_input/", "results/main_table/qwen2.5-vl/"
OUTPUTS = [("outputs/extra/video_input.txt", ["summary", "--m7b", R + "qwen2.5-vl-7b.jsonl", "--img7b", Q + "qwen2.5-vl-7b-instruct.jsonl"])]
