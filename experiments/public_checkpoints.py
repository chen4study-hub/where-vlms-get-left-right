"""Public spatially fine-tuned checkpoints of Qwen2.5-VL-3B and the component swap (paper Section 4.7, Fig. 5 and
Appendix E).

Part 1: four public checkpoints fine-tuned from Qwen2.5-VL-3B (SpatialLadder-3B, SpaceQwen2.5-VL-3B,
SpaceR-3B, VLM-R1-REC-500steps) and the base model, CV-Bench 2D Relation 650 questions x 6 variants (orig,
hflip, vflip, vit_w, vit_h, llm_w) x 2 option orders; options are scored in fp32 (lm_head input times the rows of
A / B). Each checkpoint is first checked by greedy generation on 30 questions (its generated answer must agree with
the logit answer on >= 80 %). CV-Bench 3D depth check: the Depth questions, run with the runner of
experiments.depth_shortcut on the checkpoint directory.
Part 2: hybrids of SpatialLadder-3B (S) and its base (B): vision side (ViT + merger) from S with the LLM from B,
the reverse, the merger on the other side, or only ViT blocks 28-31 from S. Each hybrid is saved, reloaded and
checked tensor by tensor, then evaluated like part1.

  python -m experiments.public_checkpoints check-config
  python -m experiments.public_checkpoints run --model hongxingli/SpatialLadder-3B --revision 0819c3ad... \\
      --parquet data/cvbench/test_2d.parquet --out results/public_checkpoints/part1/cv2d_spatialladder-3b.jsonl
  python -m experiments.public_checkpoints generate --model ... --parquet ... --out gen_<name>.jsonl
  python -m experiments.public_checkpoints build-hybrid --cfg h_vitS --out <dir> --report build_h_vitS.json
  python -m experiments.public_checkpoints verify --names base,spatialladder-3b,... --dir <dir> --items <depth item dir>
  python -m experiments.public_checkpoints summary --base cv2d_base.jsonl --model NAME RUN GEN ...   # appendix table of the public checkpoints
  python -m experiments.public_checkpoints depth3d --dir <dir> --names base,...                           # CV-Bench 3D depth check
  python -m experiments.public_checkpoints component-swap --dir <part2 dir> --part1 <part1 dir>   # appendix table of the component swap
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import time
from collections import Counter

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter

BASE = {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": None, "license": "qwen-research", "note": "base model"}
MODELS = [
    {"repo": "hongxingli/SpatialLadder-3B", "revision": "0819c3adf8827a2ea6c0348d49a23503ecb1f428",
     "license": "apache-2.0", "note": "SFT + RL; weights of all 32 ViT blocks, the merger and the LLM changed; 4 shards (possibly stored in fp32)"},
    {"repo": "remyxai/SpaceQwen2.5-VL-3B-Instruct", "revision": "332ff3fbc40d0e62af64b1bd2e0a606b3cca028e",
     "license": "qwen-research (license_name on the card, linking to the Qwen2.5-VL-3B LICENSE)",
     "note": "SFT; all weights changed; the gguf files of the repository are not downloaded"},
    {"repo": "RUBBISHLIKE/SpaceR-3B", "revision": "079d963625710049514e93c99ca855bc8a43a19d",
     "license": "not stated (empty card); research evaluation only, as for the base (qwen-research)",
     "note": "RL; attention layers changed only; the card names no base model, check-config compares the config"},
    {"repo": "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps", "revision": "b9a4c5aa915faddc7e129c3f61d630ceef73a911",
     "license": "not stated; research evaluation only, as for the base (qwen-research)", "note": "RL (REC task); attention layers changed only"},
]
# structural fields compared with the base model on the *loaded* config (AutoConfig.to_dict()); two checkpoints
# omit depth / fullatt_block_indexes / window_size in their json and get the defaults 32 / [7, 15, 23, 31] / 112
ARCH_KEYS = [("architectures",), ("hidden_size",), ("num_hidden_layers",), ("vision_config", "depth"),
             ("vision_config", "hidden_size"), ("vision_config", "fullatt_block_indexes"), ("vision_config", "window_size")]
VIT_DEPTH = 32

# variant -> (pixel transform, ViT mirrored axes, LLM mirrored axes)
VARIANTS = {"orig": (None, "", ""), "hflip": ("hflip", "", ""), "vflip": ("vflip", "", ""),
            "vit_w": (None, "w", ""), "vit_h": (None, "h", ""), "llm_w": (None, "", "w")}
MIRROR = {"lr": ("vit_w", "hflip"), "ud": ("vit_h", "vflip")}


def snapshot(repo, revision=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, local_files_only=True)


def model_name(repo):
    return repo.split("/")[1].lower()


# ---------------------------------------------------------------- structure check

def config_get(cfg, path):
    """Value at path; top-level fields missing in newer configs are looked up in text_config."""
    def walk(c):
        for k in path:
            if not isinstance(c, dict) or k not in c:
                return None
            c = c[k]
        return c
    v = walk(cfg)
    if v is None and len(path) == 1 and isinstance(cfg, dict) and isinstance(cfg.get("text_config"), dict):
        v = walk(cfg["text_config"])
    return v


def arch_mismatch(cfg, base_cfg):
    return [".".join(p) for p in ARCH_KEYS if config_get(cfg, p) != config_get(base_cfg, p)]


_BLOCK = re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.")


def vit_blocks(weight_keys):
    return sorted({int(m.group(1)) for k in weight_keys for m in [_BLOCK.search(k)] if m})


def check_models(models, load_cfg, load_keys, base_cfg):
    """{repo: [problems]}: structural fields differing from the base, or ViT blocks other than 0..31."""
    bad = {}
    for m in models:
        mis = arch_mismatch(load_cfg(m), base_cfg)
        blocks = vit_blocks(load_keys(m))
        if blocks != list(range(VIT_DEPTH)):
            mis.append(f"the weights hold {len(blocks)} visual.blocks ({blocks[:1]}…{blocks[-1:]}), expected 0–{VIT_DEPTH - 1}")
        if mis:
            bad[m["repo"]] = mis
    return bad


def loaded_config(path):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(path).to_dict()


def weight_keys(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return list(json.load(open(idx))["weight_map"])
    with open(os.path.join(path, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)) if k != "__metadata__"]


def check_config(a):
    base_path = snapshot(BASE["repo"])
    base = loaded_config(base_path)
    path = {m["repo"]: snapshot(m["repo"], m["revision"]) for m in MODELS}
    for m in MODELS:
        raw = json.load(open(os.path.join(path[m["repo"]], "config.json")))
        missing = [k for k in ("depth", "fullatt_block_indexes", "window_size") if k not in raw.get("vision_config", {})]
        print(f"[config] {m['repo']}@{m['revision'][:8]}: vision_config of the raw json lacks {missing or 'nothing'} (defaults "
              f"used when loading); torch_dtype={raw.get('torch_dtype')}; license: {m['license']}")
    print(f"[config] ViT blocks in the base weights: {len(vit_blocks(weight_keys(base_path)))}")
    bad = check_models(MODELS, lambda m: loaded_config(path[m["repo"]]), lambda m: weight_keys(path[m["repo"]]), base)
    for m in MODELS:
        print(f"[config] {m['repo']}: {'loaded structure as in the base, 32 ViT blocks in the weights' if m['repo'] not in bad else 'differs: ' + '; '.join(bad[m['repo']])}")
    if bad:
        raise SystemExit(f"structure differs from the base model, the hooks do not apply: {bad}")


# ---------------------------------------------------------------- part1 metrics

def word_pred(lp, choices):
    """Predicted relation word; None for an exact tie (fp32)."""
    if lp[0] == lp[1]:
        return None
    return choices[int(lp[1] > lp[0])]


def _wilson_pct(k, n):
    from vlmlr.stats import wilson
    return tuple(100 * x for x in wilson(k, n))


def _by_item(recs, order="fwd"):
    out = {}
    for r in recs:
        if r.get("order", "fwd") == order:
            out.setdefault(r["idx"], {})[r["variant"]] = r
    return out


def agreement(recs, axis, order="fwd"):
    """On items whose answer changes under the real flip (no ties), share where the ViT-only mirror gives the flip's
    answer. Returns (%, n, Wilson interval %)."""
    v, real = MIRROR[axis]
    sub = [d for d in _by_item(recs, order).values() if d.get("orig", {}).get("axis") == axis
           and all(c in d and d[c]["pred"] is not None for c in ("orig", v, real)) and d["orig"]["pred"] != d[real]["pred"]]
    k = sum(d[v]["pred"] == d[real]["pred"] for d in sub)
    return (100 * k / len(sub) if sub else float("nan")), len(sub), _wilson_pct(k, len(sub))


def change_rate(recs, variant, axis=None, order="fwd"):
    ds = [d for d in _by_item(recs, order).values() if variant in d and "orig" in d
          and (axis is None or d["orig"]["axis"] == axis) and d[variant]["pred"] is not None and d["orig"]["pred"] is not None]
    k = sum(d[variant]["pred"] != d["orig"]["pred"] for d in ds)
    return (100 * k / len(ds) if ds else float("nan")), len(ds), _wilson_pct(k, len(ds))


def correct(r):
    return 0.5 if r["pred"] is None else float(r["pred"] == r["gold"])


def accuracy(recs, axis=None):
    """orig accuracy (%), both option orders pooled."""
    xs = [r for r in recs if r["variant"] == "orig" and (axis is None or r["axis"] == axis)]
    if not xs:
        return float("nan"), 0
    return 100 * sum(correct(r) for r in xs) / len(xs), len(xs)


def first_option_share(recs):
    xs = [r for r in recs if r["variant"] == "orig" and r["pred"] is not None]
    return 100 * sum(r["pred"] == r["choices"][0] for r in xs) / len(xs) if xs else float("nan")


def paired_gain(ft, base, axis=None, B=2000, seed=0):
    """orig accuracy of the checkpoint minus the base (points, both orders), paired bootstrap 95 % interval."""
    from vlmlr.stats import bootstrap_mean, percentile_ci

    def key(r):
        return (r["idx"], r.get("order", "fwd"))
    b = {key(r): correct(r) for r in base if r["variant"] == "orig" and (axis is None or r["axis"] == axis)}
    f = {key(r): correct(r) for r in ft if r["variant"] == "orig" and (axis is None or r["axis"] == axis)}
    ids = [k for k in f if k in b]
    if not ids:
        return float("nan"), (float("nan"), float("nan"))
    d = [f[k] - b[k] for k in ids]
    return 100 * sum(d) / len(d), percentile_ci(bootstrap_mean(d, B=B, seed=seed, scale=100))


LETTER_RE = re.compile(r"(?:^|[\s(\[>:])\(?([AB])\)?(?=[\s).,:<\]]|$)", re.M)


def parse_generation(text, choices):
    """Final answer index of a generation: the content of <answer>...</answer> if present, else the text without
    <think>...</think>; a letter (A / B) or an option word; the last occurrence wins; None if nothing is found."""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    body = m[-1] if m else re.sub(r"<think>.*?(</think>|$)", " ", text, flags=re.S | re.I)
    hits = [(mm.start(1), ord(mm.group(1)) - 65) for mm in LETTER_RE.finditer(body)]
    for i, c in enumerate(choices):
        hits += [(mm.start(), i) for mm in re.finditer(rf"\b{c}\b", body, re.I)]
    return max(hits)[1] if hits else None


def generation_agreement(pairs, thr=80.0):
    """pairs [(generated index or None, logit index or None)] -> (agreement %, parsed count, passes >= thr)."""
    if not pairs:
        return float("nan"), 0, False
    k = sum(g is not None and g == p for g, p in pairs)
    r = 100 * k / len(pairs)
    return r, sum(g is not None for g, _ in pairs), r >= thr


# ---------------------------------------------------------------- CV-Bench 3D depth check

def _two_option_pred(lp):
    if lp[0] == lp[1]:
        return None
    return 0 if lp[0] > lp[1] else 1


def depth_rule(recs):
    """CV-Bench 3D depth, non-ambiguous items: accuracy of the 'higher means farther' rule, of the model, their
    agreement, and model accuracy where the rule is wrong."""
    o = [dict(r, pred=r["pred"] if "pred" in r else _two_option_pred(r["lp"])) for r in recs
         if r.get("cond", "orig") == "orig" and r.get("task", "Depth") == "Depth"]
    na = [r for r in o if r["group"] != "ambiguous"]
    if not na:
        return None

    def ok(r):
        return r["pred"] is not None and r["pred"] == r["gold"]

    def follow(r):
        return r["pred"] is not None and r["pred"] == (r["gold"] if r["group"] == "consistent" else 1 - r["gold"])
    cnt = [r for r in na if r["group"] == "counter"]
    return {"n_all": len(o), "n": len(na), "rule": 100 * sum(r["group"] == "consistent" for r in na) / len(na),
            "model": 100 * sum(map(ok, na)) / len(na), "agree_rule": 100 * sum(map(follow, na)) / len(na),
            "n_counter": len(cnt), "model_counter": 100 * sum(map(ok, cnt)) / len(cnt) if cnt else float("nan"),
            "model_all": 100 * sum(map(ok, o)) / len(o)}


# ---------------------------------------------------------------- part2: component swap

GROUPS = ("vit", "merger", "llm")
CONFIGS = {
    "base": {"vit": "B", "merger": "B", "llm": "B"},
    "ladder": {"vit": "S", "merger": "S", "llm": "S"},
    "h_vitS": {"vit": "S", "merger": "S", "llm": "B"},          # vision side (ViT + merger) from S
    "h_llmS": {"vit": "B", "merger": "B", "llm": "S"},          # LLM from S
    "h_vitS_mB": {"vit": "S", "merger": "B", "llm": "B"},
    "h_llmS_mS": {"vit": "B", "merger": "S", "llm": "S"},
    "h_vitS_last4": {"vit": "B", "merger": "B", "llm": "B", "vit_blocks_S": (28, 29, 30, 31)},
}
HYBRIDS = [k for k in CONFIGS if k.startswith("h_")]
MAIN_HYBRIDS = ("h_vitS", "h_llmS")
PART1_NAME = {"base": "base", "ladder": "spatialladder-3b"}
_MERGER = re.compile(r"(^|\.)visual\.merger\.")
_VISUAL = re.compile(r"(^|\.)visual\.")
_BLOCKNUM = re.compile(r"visual\.blocks\.(\d+)\.")


def group_of(name):
    if _MERGER.search(name):
        return "merger"
    if _VISUAL.search(name):
        return "vit"
    return "llm"


def source_of(cfg, name):
    spec = CONFIGS[cfg]
    m = _BLOCKNUM.search(name)
    if m and int(m.group(1)) in spec.get("vit_blocks_S", ()):
        return "S"
    return spec[group_of(name)]


def assemble(cfg, sb, ss):
    assert list(sb) == list(ss), "B and S have different parameter names"
    return {n: (ss[n] if source_of(cfg, n) == "S" else sb[n]) for n in sb}


def diff_table(names, equal_bs):
    out = {g: {"n": 0, "n_diff": 0} for g in GROUPS}
    blocks = set()
    for n in names:
        g = group_of(n)
        out[g]["n"] += 1
        if not equal_bs[n]:
            out[g]["n_diff"] += 1
            m = re.search(r"visual\.blocks\.(\d+)\.", n)
            if m:
                blocks.add(int(m.group(1)))
    out["vit"]["diff_blocks"] = sorted(blocks)
    return out


def verify_hybrid(cfg, names, eq_h_b, eq_h_s, equal_bs):
    """Every tensor of the reloaded hybrid equals its intended source; every group meant to come from S has at least
    one tensor that differs from B. Returns (ok, per-group counts, first 10 bad tensors)."""
    stat = {g: {"n": 0, "from_S": 0, "from_B": 0, "same_both": 0, "bad": 0} for g in GROUPS}
    bad = []
    for n in names:
        g, src = group_of(n), source_of(cfg, n)
        s = stat[g]
        s["n"] += 1
        if equal_bs[n]:
            s["same_both"] += 1
            ok = eq_h_b[n] and eq_h_s[n]
        elif src == "S":
            ok = eq_h_s[n] and not eq_h_b[n]
            s["from_S"] += ok
        else:
            ok = eq_h_b[n] and not eq_h_s[n]
            s["from_B"] += ok
        if not ok:
            s["bad"] += 1
            bad.append(n)
    need = {g: sum(1 for n in names if group_of(n) == g and source_of(cfg, n) == "S" and not equal_bs[n]) for g in GROUPS}
    bad = [f"<group {g} should come from S, but no tensor does>" for g in GROUPS if need[g] and stat[g]["from_S"] == 0] + bad
    return not bad, stat, bad[:10]


def orig_scores(recs, axis=None):
    return {(r["idx"], r.get("order", "fwd")): correct(r) for r in recs
            if r["variant"] == "orig" and (axis is None or r["axis"] == axis)}


def decompose(sc, B=2000, seed=0):
    """Accuracy per configuration, differences to the base with paired bootstrap intervals (one resample of
    (idx, option order) shared by all configurations), R_V = (h_vitS - base) / (ladder - base),
    R_L = (h_llmS - base) / (ladder - base), shares of the gain, and the interaction."""
    keys = sorted(set.intersection(*(set(v) for v in sc.values())))
    n = len(keys)
    cfgs = list(sc)
    X = {c: [sc[c][k] for k in keys] for c in cfgs}
    acc = {c: 100 * sum(X[c]) / n for c in cfgs}

    def stats(ix):
        a = {c: 100 * sum(X[c][i] for i in ix) / len(ix) for c in cfgs}
        g = a["ladder"] - a["base"]
        d = {c: a[c] - a["base"] for c in cfgs if c != "base"}
        rv = d["h_vitS"] / g if "h_vitS" in d and g else float("nan")
        rl = d["h_llmS"] / g if "h_llmS" in d and g else float("nan")
        share = {c: d[c] / g if g else float("nan") for c in d}
        return d, rv, rl, share

    from vlmlr.stats import resamples
    d0, rv0, rl0, sh0 = stats(range(n))
    bs = [stats(ix) for ix in resamples(n, B, seed)]

    def ci(vals):
        v = sorted(x for x in vals if x == x)
        return (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (float("nan"), float("nan"))
    out = {"n": n, "acc": acc,
           "delta": {c: (d0[c], ci([b[0][c] for b in bs])) for c in d0},
           "R_V": (rv0, ci([b[1] for b in bs])), "R_L": (rl0, ci([b[2] for b in bs])),
           "share": {c: (sh0[c], ci([b[3][c] for b in bs])) for c in sh0}}
    if all(c in d0 for c in ("ladder", "h_vitS", "h_llmS")):
        out["interaction"] = d0["ladder"] - d0["h_vitS"] - d0["h_llmS"]
    return out


def mismatch(acc, acc_base, first):
    """Sanity check of a hybrid model: flagged if its accuracy is more than 10 points below the base model or it picks
    the first option in more than 75 % of the answers."""
    return acc < acc_base - 10 or first > 75


# ---------------------------------------------------------------- GPU runs

def _load(a):
    """bf16 model and processor at a fixed revision (the hub cache or a local directory)."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from vlmlr.models import QWEN_MAX_PIXELS
    kw = {"revision": a.revision} if a.revision else {}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", **kw).eval()
    return model, AutoProcessor.from_pretrained(a.model, max_pixels=QWEN_MAX_PIXELS, **kw)


def run(a):
    import torch
    from vlmlr import models
    from vlmlr.models import FP32OptionScorer
    from vlmlr.hooks import QwenPositionMirror
    model, proc = _load(a)
    ids = [proc.tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    assert all(len(t) == 1 for t in ids), ids
    ids = [t[0] for t in ids]
    fl = QwenPositionMirror(model)
    scorer = FP32OptionScorer(model, ids)

    def fwd(inp, vit, llm):
        fl.forward_logits(model, inp, llm=llm, vit=vit)
        return scorer.scores()

    items = C.load_items(a.parquet, a.limit)
    checked, t0 = False, time.time()
    with ResumableWriter(a.out, ("idx", "variant", "order")) as out, torch.inference_mode():
        for it in items:
            img0 = C.to_image(it["image"])
            gold = C.gold_relation(it)
            axis = C.AXIS[gold]
            for order in ("fwd", "swap"):
                choices = C.CHOICES[axis] if order == "fwd" else C.CHOICES[axis][::-1]
                text = C.prompt(it["question"], choices, False)
                if not checked:
                    inp = models.build_inputs(proc, img0, text)
                    fl.self_check(inp)
                    b1, b2 = fwd(inp, "", ""), fwd(inp, "", "")
                    assert torch.equal(b1, b2), "orig not reproducible"
                    model.model.rope_deltas = None
                    ref = model(**inp).logits[0, -1].float()[ids]
                    assert (ref - b1).abs().max() < 0.25, f"fp32 scores far from the model logits: {b1.tolist()} vs {ref.tolist()}"
                    for v, l in (("w", ""), ("", "w")):
                        assert not torch.equal(b1, fwd(inp, v, l)), f"vit={v!r} llm={l!r} changed nothing"
                    checked = True
                for var, (tf, vit, llm) in VARIANTS.items():
                    if (it["idx"], var, order) in out:
                        continue
                    lp = torch.log_softmax(fwd(models.build_inputs(proc, C.transform(img0, tf), text), vit, llm), -1).tolist()
                    out.write({"idx": it["idx"], "variant": var, "order": order, "axis": axis, "gold": gold,
                               "choices": choices, "pred": word_pred(lp, choices), "lp": lp})
            out.flush()
    print(f"[run] done {out.n} forwards in {time.time() - t0:.0f}s -> {a.out}")


def generate(a):
    import torch
    from vlmlr import models
    from vlmlr.models import FP32OptionScorer, greedy_text
    model, proc = _load(a)
    tok = proc.tokenizer
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")]
    scorer = FP32OptionScorer(model, ids)
    items = C.load_items(a.parquet, 0)
    random.Random(a.seed).shuffle(items)
    items = items[: a.n]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "w") as f, torch.inference_mode():
        for it in items:
            gold = C.gold_relation(it)
            choices = C.CHOICES[C.AXIS[gold]]
            inp = models.build_inputs(proc, C.to_image(it["image"]), C.prompt(it["question"], choices, False))
            model.model.rope_deltas = None
            model(**inp)
            lp = scorer.scores().tolist()
            logit_k = None if lp[0] == lp[1] else int(lp[1] > lp[0])
            txt = greedy_text(model, tok, inp, a.max_new)
            f.write(json.dumps({"idx": it["idx"], "gold": gold, "choices": choices, "logit_k": logit_k,
                                "gen_k": parse_generation(txt, choices), "text": txt[:2000]}, ensure_ascii=False) + "\n")
    print(f"[gen] {len(items)} items in {time.time() - t0:.0f}s -> {a.out}")


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.json", "chat_template.jinja", "generation_config.json")
VISION_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def build_hybrid(a):
    """Assemble a hybrid from B and S (bf16, as in part1), save it, reload it and check every tensor. Tokenizer,
    chat template and generation config come from the LLM side, the image preprocessor from the vision side."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration as M
    t0 = time.time()
    lad = next(m for m in MODELS if m["repo"] == "hongxingli/SpatialLadder-3B")
    pb = a.base or snapshot(BASE["repo"])
    ps = a.ladder or snapshot(lad["repo"], lad["revision"])
    kw = dict(torch_dtype=torch.bfloat16, device_map=a.device, attn_implementation="sdpa")
    mb, ms = M.from_pretrained(pb, **kw).eval(), M.from_pretrained(ps, **kw).eval()
    sb, ss = mb.state_dict(), ms.state_dict()
    assert list(sb) == list(ss), "B and S have different parameter names"
    names = list(sb)
    with torch.no_grad():
        equal_bs = {n: torch.equal(sb[n], ss[n]) for n in names}
    diff = diff_table(names, equal_bs)
    print("[build] B vs S: " + "; ".join(f"{g} {d['n_diff']}/{d['n']}" for g, d in diff.items())
          + f"; ViT blocks that differ: {diff['vit']['diff_blocks']}", flush=True)
    spec = CONFIGS[a.cfg]
    sd = {n: t.cpu() for n, t in assemble(a.cfg, sb, ss).items()}
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    mb.save_pretrained(a.out, state_dict=sd, safe_serialization=True)
    del sd
    llm_side, vit_side = (ps if spec["llm"] == "S" else pb), (ps if spec["vit"] == "S" else pb)
    files = {}
    for f in TOKENIZER_FILES + VISION_FILES:
        src = vit_side if f in VISION_FILES else llm_side
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(a.out, f))
        files[f] = {"from": "S" if src == ps else "B", "same_in_B_S": _md5(os.path.join(pb, f)) == _md5(os.path.join(ps, f))}
    print(f"[build] {a.cfg} ({spec}) saved to {a.out}; processor files that differ between B and S: "
          f"{[f for f, v in files.items() if not v['same_in_B_S']] or 'none'}", flush=True)
    sh = M.from_pretrained(a.out, **kw).eval().state_dict()
    assert list(sh) == names, "reloaded parameter names differ from B"
    with torch.no_grad():
        eq_h_b = {n: torch.equal(sh[n], sb[n]) for n in names}
        eq_h_s = {n: torch.equal(sh[n], ss[n]) for n in names}
    ok, stat, bad = verify_hybrid(a.cfg, names, eq_h_b, eq_h_s, equal_bs)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump({"cfg": a.cfg, "spec": spec, "ok": ok, "stat": stat, "bad": bad, "diff_B_S": diff, "files": files,
               "base": pb, "ladder": ps, "seconds": round(time.time() - t0)}, open(a.report, "w"), ensure_ascii=False, indent=1)
    print(f"[build] tensor check after reloading: {'passed' if ok else 'failed'}; " +
          "; ".join(f"{g}: from S {s['from_S']}, from B {s['from_B']}, same in both {s['same_both']}, wrong {s['bad']}" for g, s in stat.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)
    if not ok:
        raise SystemExit(f"[build] tensor check failed: {bad}")


# ---------------------------------------------------------------- completeness of the output files

def _read_strict(p):
    if not os.path.exists(p):
        return None, [f"missing file {os.path.basename(p)}"]
    raw = open(p, "rb").read()
    bad, rs = [], []
    if raw and not raw.endswith(b"\n"):
        bad.append(f"{os.path.basename(p)} does not end with a newline (the last line may be partial)")
    for k, line in enumerate(raw.decode("utf-8", "replace").split("\n"), 1):
        if not line.strip():
            continue
        try:
            rs.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(f"{os.path.basename(p)} line {k} is not complete JSON")
    return rs, bad


def _item_ids(p):
    if not os.path.exists(p):
        return None
    return [json.loads(x)["idx"] for x in open(p) if x.strip()]


def _check_file(path, expected, keyf, tiesf):
    rs, bad = _read_strict(path)
    name = os.path.basename(path)
    if rs is None:
        return bad, 0, 0
    keys = Counter(keyf(r) for r in rs)
    dup = [k for k, c in keys.items() if c > 1]
    if dup:
        bad.append(f"{name} has {len(dup)} duplicate keys, e.g. {dup[:2]}")
    if expected is None:
        bad.append(f"{name}: item file not found, expected keys unknown")
    else:
        miss, extra = expected - set(keys), set(keys) - expected
        if miss:
            bad.append(f"{name} lacks {len(miss)} keys, e.g. {sorted(miss, key=str)[:2]}")
        if extra:
            bad.append(f"{name} has {len(extra)} unexpected keys, e.g. {sorted(extra, key=str)[:2]}")
        if len(rs) != len(expected):
            bad.append(f"{name} has {len(rs)} lines, expected {len(expected)}")
    return bad, sum(map(tiesf, rs)), len(rs)


def verify(a):
    """Every output file is complete: no partial line, unique keys, exactly the expected keys (computed from the
    item files). Prints ALL FILES COMPLETE only if every model passes."""
    if a.ids2d:
        i2 = _item_ids(a.ids2d)
    elif os.path.exists(a.parquet):
        import pyarrow.parquet as pq
        i2 = [r["idx"] for r in pq.read_table(a.parquet, columns=["idx", "task"]).to_pylist() if r["task"] == "Relation"]
    else:
        i2 = None
    vars2 = list(VARIANTS)
    conds3 = ("orig", "vit_h", "vit_w")
    exp = {"2d": None if i2 is None else {(i, v, o) for i in i2 for v in vars2 for o in ("fwd", "swap")}}
    for k, f in (("val", "items_val.jsonl"), ("test", "items_test_depth.jsonl")):
        x = _item_ids(os.path.join(a.items, f))
        exp[k] = None if x is None else {(i, c) for i in x for c in conds3}
    print("[verify] expected keys: " + ", ".join(f"{k} {'item file not found' if v is None else len(v)}" for k, v in exp.items()))

    def k2(r):
        return (r.get("idx"), r.get("variant"), r.get("order"))

    def kc(r):
        return (r.get("idx"), r.get("cond"))

    def tie_pred(r):
        return r.get("pred") is None

    def tie_lp(r):
        return "lp" in r and len(r["lp"]) == 2 and r["lp"][0] == r["lp"][1]
    allbad = {}
    for n in a.names.split(","):
        files = ((f"cv2d_{n}.jsonl", exp["2d"], k2, tie_pred), (f"depth3d_{n}_val.jsonl", exp["val"], kc, tie_lp),
                 (f"depth3d_{n}_test.jsonl", exp["test"], kc, tie_lp))
        bad, info = [], []
        for f, e, kf, tf in files:
            b, ties, rows = _check_file(os.path.join(a.dir, f), e, kf, tf)
            bad += b
            info.append(f"{f} {rows} lines / {ties} ties")
        print(f"[verify] {n}: {'complete' if not bad else 'incomplete'} ({'; '.join(info)})")
        for b in bad:
            print(f"    - {b}")
        if bad:
            allbad[n] = bad
    if allbad:
        print(f"[verify] {len(allbad)} models with incomplete output files")
        sys.exit(1)
    print("ALL FILES COMPLETE")


# ---------------------------------------------------------------- analyses

def _read(p, optional=False):
    if not os.path.exists(p):
        if optional:
            return []
        raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
    return [json.loads(x) for x in open(p)]


def _describe(name, recs, base=None, gen=None):
    row = {"acc": accuracy(recs)[0], "acc_lr": accuracy(recs, "lr")[0], "acc_ud": accuracy(recs, "ud")[0],
           "first": first_option_share(recs), "ties": sum(r["pred"] is None for r in recs)}
    row["agree_lr"], row["agree_lr_n"], row["agree_lr_ci"] = agreement(recs, "lr")
    row["agree_lr_swap"] = agreement(recs, "lr", "swap")[0]
    row["llm_w"] = change_rate(recs, "llm_w", "lr")[0]
    if base is not None:
        row["gain"] = paired_gain(recs, base)
    if gen is not None:
        r, n_parse, ok = generation_agreement([(g["gen_k"], g["logit_k"]) for g in gen])
        row["gen_agree"], row["gen_parsed"], row["gen_ok"] = r, n_parse, ok
    print(f"\n== {name}")
    print(f"  orig accuracy {row['acc']:.1f}% (left-right {row['acc_lr']:.1f}, up-down {row['acc_ud']:.1f}; both option orders); "
          f"first option chosen {row['first']:.1f}%; ties {row['ties']}")
    lo, hi = row["agree_lr_ci"]
    print(f"  lr agreement (ViT positions mirrored vs real flip): {row['agree_lr']:.1f}% [{lo:.1f}, {hi:.1f}] (n={row['agree_lr_n']}; "
          f"options swapped {row['agree_lr_swap']:.1f}%)")
    print(f"  left-right items, llm_w change rate {row['llm_w']:.1f}%")
    if "gain" in row:
        print(f"  relative to the base: gain {row['gain'][0]:+.1f} [{row['gain'][1][0]:+.1f}, {row['gain'][1][1]:+.1f}]")
    if gen is not None:
        print(f"  generation check: {len(gen)} items, parsed {row['gen_parsed']}, agrees with the logit answer {row['gen_agree']:.1f}%")
    return row


def summary(a):
    """Table E.1: accuracy, agreement of vit_w with hflip, llm_w change, gain over the base, generation check."""
    base = _read(a.base)
    res = {"base": _describe("base (Qwen2.5-VL-3B)", base)}
    for name, run_p, gen_p in a.model:
        recs = _read(run_p)
        res[name] = _describe(name, recs, base, _read(gen_p))
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


def depth3d(a):
    """CV-Bench 3D depth check per model: the 'higher means farther' rule and the model on the non-ambiguous items."""
    if a.names:
        names = a.names.split(",")
    else:
        names = sorted({os.path.basename(p)[len("depth3d_"):].rsplit("_", 1)[0] for p in glob.glob(os.path.join(a.dir, "depth3d_*.jsonl"))})
        if not names:
            raise FileNotFoundError(f"per-item predictions not found in {a.dir} (run the experiment first, see README)")
    res, lines = {}, []
    for n in names:
        lines.append(f"\n== {n}")
        d = depth_rule(_read(os.path.join(a.dir, f"depth3d_{n}_val.jsonl"), True) + _read(os.path.join(a.dir, f"depth3d_{n}_test.jsonl"), True))
        res[n] = {"3d": d}
        if d:
            lines.append(f"  3D Depth (val + test) not ambiguous {d['n']} (of {d['n_all']}): rule {d['rule']:.1f}  model {d['model']:.1f}  "
                         f"agrees with the rule {d['agree_rule']:.1f}  on the {d['n_counter']} items where the rule is wrong, model {d['model_counter']:.1f}")
    print("\n".join(lines))
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1)


def component_swap(a):
    """Table E.2: build checks, every configuration, the gain decomposition, and the agreement of the
    re-run base / ladder with part1."""
    def fmt_d(d):
        v, (lo, hi) = d
        return f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    recs = {c: _read(os.path.join(a.dir, f"cv2d_{c}.jsonl")) for c in CONFIGS}
    have = [c for c in CONFIGS if recs[c]]
    lines, res = [], {"configs": {}}
    lines.append("== Build check (each hybrid reloaded from disk and compared tensor by tensor)")
    for c in HYBRIDS:
        p = os.path.join(a.dir, f"build_{c}.json")
        b = json.load(open(p)) if os.path.exists(p) else None
        res["configs"].setdefault(c, {})["build"] = b
        if b:
            st = "; ".join(f"{g} from S {s['from_S']} / from B {s['from_B']} / same in both {s['same_both']} / wrong {s['bad']}" for g, s in b["stat"].items())
            lines.append(f"  {c} {b['spec']}: {st}" + ("" if b["ok"] else f"; tensors that differ {b['bad']}"))
        else:
            lines.append(f"  {c}: build_{c}.json missing")
    if any(res["configs"].get(c, {}).get("build") for c in HYBRIDS):
        d = next(res["configs"][c]["build"]["diff_B_S"] for c in HYBRIDS if res["configs"].get(c, {}).get("build"))
        lines.append("  tensors that differ between B and S: " + "; ".join(f"{g} {v['n_diff']}/{v['n']}" for g, v in d.items() if isinstance(v, dict))
                     + f"; ViT blocks that differ {d['vit']['diff_blocks']}")
    base_acc = accuracy(recs["base"])[0] if recs["base"] else float("nan")
    flags = {}
    lines.append("\n== Configurations (CV-Bench 2D, 650 items × two option orders; main-table metrics; the CV-Bench 3D depth check for base, ladder, h_vitS and h_llmS)")
    for c in have:
        r = recs[c]
        row = {"acc": accuracy(r)[0], "acc_lr": accuracy(r, "lr")[0], "acc_ud": accuracy(r, "ud")[0],
               "first": first_option_share(r), "ties": sum(x["pred"] is None for x in r if x["variant"] == "orig"),
               "agree_lr": agreement(r, "lr")[0]}
        row["depth_rule"] = depth_rule(_read(os.path.join(a.dir, f"depth3d_{c}_val.jsonl"), True) + _read(os.path.join(a.dir, f"depth3d_{c}_test.jsonl"), True))
        flags[c] = c in HYBRIDS and mismatch(row["acc"], base_acc, row["first"])
        row["mismatch"] = flags[c]
        res["configs"].setdefault(c, {}).update(row)
        depth_txt = f"3D model {row['depth_rule']['model']:.1f} agrees with the rule {row['depth_rule']['agree_rule']:.1f} items where the rule is wrong {row['depth_rule']['model_counter']:.1f}" if row["depth_rule"] else ""
        lines.append(f"  {c:10s} {CONFIGS[c]}: accuracy {row['acc']:.1f} (left-right {row['acc_lr']:.1f} up-down {row['acc_ud']:.1f})"
                     f" first option {row['first']:.1f}% ties {row['ties']}{' [mismatch]' if flags[c] else ''}\n"
                     f"             agreement lr {row['agree_lr']:.1f}" + (f"\n             {depth_txt}" if depth_txt else ""))
    if all(recs[c] for c in ("base", "ladder") + MAIN_HYBRIDS):
        for ax, name in ((None, "all items"), ("lr", "left-right"), ("ud", "up-down")):
            dec = decompose({c: orig_scores(recs[c], ax) for c in have})
            res.setdefault("decompose", {})[name] = dec
            lines.append(f"\n== Decomposition of the gain ({name}, n = {dec['n']}, paired bootstrap 95% intervals; the merger goes with the ViT)")
            for c, d in dec["delta"].items():
                sh, (lo, hi) = dec["share"][c]
                lines.append(f"  {c:12s} − base: {fmt_d(d)}; share of the ladder gain {sh:.2f} [{lo:.2f}, {hi:.2f}]")
            lines.append(f"  R_V (share of the gain kept with the vision side from S) {dec['R_V'][0]:.2f} [{dec['R_V'][1][0]:.2f}, {dec['R_V'][1][1]:.2f}]; "
                         f"R_L (LLM side) {dec['R_L'][0]:.2f} [{dec['R_L'][1][0]:.2f}, {dec['R_L'][1][1]:.2f}]; interaction {dec.get('interaction', float('nan')):+.1f}")
    else:
        lines.append("\n(base, ladder, h_vitS or h_llmS missing: no decomposition)")
    if a.part1:
        lines.append("\n== Re-run base / ladder compared row by row with the runs of the public checkpoints (same idx, variant, option order)")
        for c, n1 in PART1_NAME.items():
            old = {(r["idx"], r["variant"], r["order"]): r for r in _read(os.path.join(a.part1, f"cv2d_{n1}.jsonl"))}
            new = {(r["idx"], r["variant"], r["order"]): r for r in recs.get(c, [])}
            ks = [k for k in new if k in old]
            if ks:
                same = sum(new[k]["pred"] == old[k]["pred"] for k in ks)
                dmax = max(max(abs(x - y) for x, y in zip(new[k]["lp"], old[k]["lp"])) for k in ks)
                lines.append(f"  {c}: {len(ks)} rows, same prediction {100 * same / len(ks):.2f}%, max log-probability difference {dmax:.4f}")
                res["configs"][c]["vs_part1"] = {"n": len(ks), "same": same, "max_lp_diff": dmax}
    print("\n".join(lines))
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.public_checkpoints", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-config", help="CPU: loaded configs and ViT blocks of the checkpoints match the base model")
    for name in ("run", "generate"):
        p = sub.add_parser(name, help="GPU: 6 variants x 2 option orders" if name == "run" else "GPU: generation check")
        p.add_argument("--model", required=True)
        p.add_argument("--revision")
        p.add_argument("--parquet", required=True)
        p.add_argument("--out", required=True)
    sub.choices["run"].add_argument("--limit", type=int, default=0)
    g = sub.choices["generate"]
    g.add_argument("--n", type=int, default=30)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-new", type=int, default=512)
    b = sub.add_parser("build-hybrid", help="GPU: assemble, save and verify a hybrid of SpatialLadder-3B and its base")
    b.add_argument("--cfg", required=True, choices=HYBRIDS)
    b.add_argument("--out", required=True)
    b.add_argument("--report", required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--base", help="local directory of the base model (default: the Hugging Face cache)")
    b.add_argument("--ladder", help="local directory of SpatialLadder-3B (default: the Hugging Face cache)")
    v = sub.add_parser("verify", help="completeness of the output files")
    v.add_argument("--names", required=True)
    v.add_argument("--dir", required=True)
    v.add_argument("--items", required=True, help="directory with items_val.jsonl, items_test_depth.jsonl")
    v.add_argument("--parquet", default="data/cvbench/test_2d.parquet")
    v.add_argument("--ids2d", help="jsonl of 2D item ids instead of the parquet")
    s = sub.add_parser("summary", help="Table E.1")
    s.add_argument("--base", required=True)
    s.add_argument("--model", nargs=3, action="append", metavar=("NAME", "RUN", "GEN"), default=[])
    s.add_argument("--json")
    au = sub.add_parser("depth3d", help="CV-Bench 3D depth check")
    au.add_argument("--dir", required=True)
    au.add_argument("--names", default="")
    au.add_argument("--json")
    cs = sub.add_parser("component-swap", help="Table E.2")
    cs.add_argument("--dir", required=True)
    cs.add_argument("--part1")
    cs.add_argument("--json")
    a = ap.parse_args(argv)
    {"check-config": check_config, "run": run, "generate": generate, "build-hybrid": build_hybrid, "verify": verify,
     "summary": summary, "depth3d": depth3d, "component-swap": component_swap}[a.cmd](a)


S1, G, S2 = "results/public_checkpoints/part1/", "results/public_checkpoints/part1_generation_check/", \
    "results/public_checkpoints/part2_component_swap/"
_NAMES = [model_name(m["repo"]) for m in MODELS]
OUTPUTS = [
    ("outputs/public_checkpoints/summary.txt",
     ["summary", "--base", S1 + "cv2d_base.jsonl"] + sum((["--model", n, S1 + f"cv2d_{n}.jsonl", G + f"gen_{n}.jsonl"] for n in _NAMES), []), True),
    ("outputs/public_checkpoints/depth3d.txt", ["depth3d", "--dir", S1, "--names", ",".join(["base"] + _NAMES)], True),
    ("outputs/public_checkpoints/component_swap.txt", ["component-swap", "--dir", S2, "--part1", S1], True),
]


if __name__ == "__main__":
    main()
