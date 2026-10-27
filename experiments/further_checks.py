"""Further checks (paper Sections 4.1-4.7; protocols in Appendix B).

GPU runs (command run, --exp; per-item files under results/further_checks/<run>/):
  vertical-content    What'sUp: vertical content-only control on the up-down items (pixels flipped upside down,
                      then the ViT and / or LLM height ids flipped back); the horizontal content-only control on the
                      left-right items as reference. Qwen2.5-VL-3B / 7B and InternVL3-2B (single tile).
  block-sets          Qwen2.5-VL: width mirror in chosen sets of ViT blocks only (vlmlr.hooks.QwenLayerwiseMirror),
                      CV-Bench left-right.
  internvl-tiling     InternVL3-2B with its default tiling (up to 12 tiles plus a thumbnail): ViT width mirror per
                      tile, LLM-side width ids and token order mirrored within each tile or across the whole image
                      (the tile columns reversed too).
  qwen3-mirrors       Qwen3-VL-4B: ViT mirrors (absolute embedding and 2D-RoPE together), LLM-side ids, token order
                      (the DeepStack features move with the visual tokens), on CV-Bench and What'sUp.
  qwen3-permutations  Qwen3-VL-4B: same-axis non-mirror permutations of the ViT position code (half-grid shift and
                      three random permutations of the merge units, vlmlr.permutations, applied to the 2D-RoPE and the
                      absolute embedding together) against the ViT width mirror; left-right items of CV-Bench and
                      What'sUp (height permutations on CV-Bench as the off-axis reference); pixels and the LLM side
                      unchanged.
  run-option-order    InternVL3-2B: the 19 conditions of the main table with the two options in reverse order.
  Scoring as in experiments.controls_permutation (bf16 option letters, ties to the first option; fp32 kept as lp32).
  python -m experiments.further_checks run --exp vertical-content --data whatsup --model q7b --whatsup <snapshot> \\
      --out results/further_checks/vertical_content/q7b_whatsup.jsonl
  python -m experiments.further_checks run --exp block-sets --data cv2d --model q7b --parquet <dir>/test_2d.parquet \\
      --out results/further_checks/block_sets/q7b_cv2d.jsonl
  python -m experiments.further_checks run --exp internvl-tiling --data cv2d --model ivl2b ... \\
      --out results/further_checks/internvl_tiling/ivl2b_cv2d.jsonl
  python -m experiments.further_checks run --exp qwen3-mirrors --data cv2d --model q3vl4b ... \\
      --out results/further_checks/qwen3_mirrors/q3vl4b_cv2d.jsonl
  python -m experiments.further_checks run --exp qwen3-permutations --data cv2d --model q3vl4b ... \\
      --out results/further_checks/qwen3_permutations/q3vl4b_cv2d.jsonl
  python -m experiments.further_checks run --exp qwen3-permutations --data whatsup --model q3vl4b ... \\
      --out results/further_checks/qwen3_permutations/q3vl4b_whatsup.jsonl
  python -m experiments.further_checks run-option-order --parquet <dir>/test_2d.parquet \\
      --out results/further_checks/internvl_option_order/ivl2b_swap.jsonl
  Depth, base model on all 600 CV-Bench 3D Depth items (orig, vit_h, vit_w; fp32 option scores):
  python -m experiments.further_checks run-depth-all --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --items items_val.jsonl,items_test.jsonl --out results/depth_shortcut/cvbench3d_all/base_7b.jsonl
Analyses (CPU):
  gpu-summary            vertical-content, block-sets, internvl-tiling, run-option-order, qwen3-mirrors
                                                                         -> outputs/further_checks/gpu_experiments.txt
  box-reclassification   RefCOCO boxes under vit_w / llm_w, with the class 'coincides with the box for the flipped image'
  object-boxes           COCO boxes of the two objects of each CV-Bench COCO question (input of object-order)
  object-order           same-axis width permutations split by whether they reverse the two objects' order (post hoc)
  depth-conflicts        depth conflict items: errors corrected by vit_h / vit_w, strength-matched comparison,
                         heuristic-following rate against the base rate (post hoc)
  depth-shared-images    CV-Bench 3D Depth: source images shared by the validation and the held-out half
  vertical-content-cvbench  vertical / horizontal content-only control on CV-Bench (post hoc)
  qwen3-permutations     flip rates of the mirror and the permutations, their difference and ratio, split by
                         whether a permutation reverses the two objects' order, What'sUp destinations
"""
import argparse
import collections
import io
import json
import math
import os
import random
import re
import time

from vlmlr import cvbench as C
from vlmlr import permutations as PM
from vlmlr.records import read_jsonl

REPO = {"q3b": "Qwen/Qwen2.5-VL-3B-Instruct", "q7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "ivl2b": "OpenGVLab/InternVL3-2B-hf", "q3vl4b": "Qwen/Qwen3-VL-4B-Instruct"}
FAMILY = {"q3b": "qwen", "q7b": "qwen", "ivl2b": "internvl", "q3vl4b": "qwen3"}
RUN_ROOT = "results/further_checks"          # one directory per GPU run, named as below
RUN_DIR = {"vertical-content": "vertical_content", "block-sets": "block_sets", "internvl-tiling": "internvl_tiling",
           "qwen3-mirrors": "qwen3_mirrors", "qwen3-permutations": "qwen3_permutations",
           "run-option-order": "internvl_option_order"}
MAIN_WH = {"q3b": "results/whatsup/qwen2.5-vl_internvl3-2b/qwen25vl3b.jsonl",
           "q7b": "results/whatsup/qwen2.5-vl_internvl3-2b/qwen25vl7b.jsonl",
           "ivl2b": "results/whatsup/qwen2.5-vl_internvl3-2b/internvl32b.jsonl"}
MAIN_BLK = {"q3b": "results/blocks_dose/blocks_qwen2.5-vl-3b-instruct.jsonl",
            "q7b": "results/blocks_dose/blocks_qwen2.5-vl-7b-instruct.jsonl"}
MAIN_CV = {"q3b": "results/main_table/qwen2.5-vl/qwen2.5-vl-3b-instruct.jsonl",
           "q7b": "results/main_table/qwen2.5-vl/qwen2.5-vl-7b-instruct.jsonl",
           "ivl2b": "results/main_table/internvl3-2b/internvl3-2b-hf.jsonl",
           "ivl8b": "results/main_table/internvl3-8b/internvl3-8b-hf.jsonl"}
DEPTH_TEST = {"7b": "results/depth_shortcut/cvbench3d_test/test_qwen2.5-vl-7b-instruct.jsonl",
              "3b": "results/depth_shortcut/cvbench3d_test/test_qwen2.5-vl-3b-instruct.jsonl"}
# base model on all 600 CV-Bench 3D Depth items, orig / vit_h / vit_w, fp32 (run-depth-all)
DEPTH_ALL = {"7b": "results/depth_shortcut/cvbench3d_all/base_7b.jsonl",
             "3b": "results/depth_shortcut/cvbench3d_all/base_3b.jsonl"}
CTRL_DIR = "results/permutation_controls"
OBJECT_BOXES = "results/further_checks/object_boxes.jsonl"
COCO_ANN = "datasets/coco/annotations/instances_val2017.json"
BOX_RUNS = {"7b": "results/extra/box_readout/qwen2.5-vl-7b.jsonl", "3b": "results/extra/box_readout/qwen2.5-vl-3b.jsonl"}
CVBENCH_3D = "datasets/nyu-visionx__CV-Bench/test_3d.jsonl"
REPRO_MIN = 1.0
INV = {"left": "right", "right": "left", "above": "below", "below": "above", "on": "under", "under": "on"}


# ================================================================ conditions of the GPU runs
# keys as in experiments.controls_permutation.COND; also blocks (block-sets: ViT blocks with mirrored width
# positions), tile (internvl-tiling: 'tile' within each tile, 'img' across the whole image).

def _c(tf=None, vh="", vw="", emb="", pos="", axes=("lr", "ud"), **kw):
    d = dict(tf=tf, vh=vh, vw=vw, emb=emb, pos=pos, axes=axes)
    d.update(kw)
    return d


LR, UD = ("lr",), ("ud",)
COND_VERTICAL = {
    "orig": _c(),
    "hflip": _c(tf="hflip", axes=LR),
    "hflip_both_w": _c(tf="hflip", vw="mirror", pos="w", axes=LR),
    "vflip": _c(tf="vflip", axes=UD),
    "vflip_vit_h": _c(tf="vflip", vh="mirror", axes=UD),
    "vflip_llm_h": _c(tf="vflip", pos="h", axes=UD),
    "vflip_both_h": _c(tf="vflip", vh="mirror", pos="h", axes=UD),
}
GLOBAL = (7, 15, 23, 31)          # global-attention blocks of the Qwen2.5-VL ViT (32 blocks; the others are windowed)
N_VIT = 32


def block_sets(n=N_VIT, gl=GLOBAL):
    last = lambda L: frozenset(range(n - L, n))
    return {
        "b31": frozenset([n - 1]),
        "g4": frozenset(gl),
        "all": frozenset(range(n)),
        "g3_early": frozenset(gl[:-1]),
        "win24_30": frozenset(range(n - 8, n - 1)),
        "lastg2": frozenset(gl[-2:]),
        "lastg3": frozenset(gl[-3:]),
        "last2": last(2), "last4": last(4), "last8": last(8), "last16": last(16),
        "nonglobal": frozenset(range(n)) - frozenset(gl),
    }


COND_BLOCKS = {"orig": _c(axes=LR)}
COND_BLOCKS.update({f"blk_{k}": _c(vw="mirror", blocks=v, axes=LR) for k, v in block_sets().items()})
BLOCKS_REPRO = {"blk_b31": "orig_vit_blk31", "blk_g4": "orig_vit_global", "blk_all": "orig_vit_all"}
COND_TILING = {
    "orig": _c(axes=LR), "hflip": _c(tf="hflip", axes=LR),
    "vit_w_tile": _c(vw="mirror", axes=LR),
    "llm_w_tile": _c(pos="w", tile="tile", axes=LR),
    "llm_w_img": _c(pos="w", tile="img", axes=LR),
    "order_only_w_tile": _c(emb="w", pos="w", tile="tile", axes=LR),
    "order_only_w_img": _c(emb="w", pos="w", tile="img", axes=LR),
}
COND_Q3_CV = {
    "orig": _c(), "hflip": _c(tf="hflip"), "vflip": _c(tf="vflip"),
    "vit_w": _c(vw="mirror"), "vit_h": _c(vh="mirror"),
    "llm_w": _c(pos="w"), "llm_h": _c(pos="h"),
    "order_only_w": _c(emb="w", pos="w", axes=LR), "order_and_id_w": _c(emb="w", axes=LR),
    "hflip_both_w": _c(tf="hflip", vw="mirror", pos="w", axes=LR),
}
COND_Q3_WH = {k: COND_Q3_CV[k] for k in ("orig", "hflip", "vit_w", "llm_w", "order_only_w")}
COND_Q3PERM_WH = {"orig": _c(axes=LR), "hflip": _c(tf="hflip", axes=LR), "vit_w": _c(vw="mirror", axes=LR)}
COND_Q3PERM_WH.update({f"vit_w_{k}": _c(vw=k, axes=LR) for k in PM.PERMS})
COND_Q3PERM_CV = dict(COND_Q3PERM_WH, **{f"vit_h_{k}": _c(vh=k, axes=LR) for k in PM.PERMS})
PLAN = {("vertical-content", "whatsup"): COND_VERTICAL, ("block-sets", "cv2d"): COND_BLOCKS,
        ("internvl-tiling", "cv2d"): COND_TILING, ("qwen3-mirrors", "cv2d"): COND_Q3_CV,
        ("qwen3-mirrors", "whatsup"): COND_Q3_WH, ("qwen3-permutations", "cv2d"): COND_Q3PERM_CV,
        ("qwen3-permutations", "whatsup"): COND_Q3PERM_WH}


def conds_for(exp, data, axis):
    return [n for n, c in PLAN[(exp, data)].items() if axis in c["axes"]]


# ================================================================ statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, (max(0.0, c - h), min(1.0, c + h))


# ================================================================ GPU runs

class InternMulti:
    """InternVL3 with tiling: absolute embedding mirrored in every tile (vw); LLM-side contents (emb) and positions
    (pos) permuted by vlmlr.permutations.multitile_perm. The tile grid is re-created by tile_grid and checked
    against the number of tiles and image tokens before the forward pass."""

    def __init__(self, model):
        from vlmlr import positions as P
        self.P = P
        self.emb_mod = model.model.vision_tower.embeddings
        self.tok = model.config.image_token_id
        g = self.emb_mod.patch_embeddings.patch_shape[0]
        self.coords = P.pixel_shuffle_coords(model.model.pixel_shuffle, g, model.config.downsample_ratio)
        self.in_tile = P.mirror_permutation(self.coords, "w").tolist()
        self.cfg, self.layout, self.hits = _c(), None, {"vit": 0, "emb": 0}
        self._pe0, self.emb_mod.interpolate_pos_encoding = self.emb_mod.interpolate_pos_encoding, self._pe
        self._feat0, model.model.get_image_features = model.model.get_image_features, self._feat

    def set(self, cfg, layout):
        self.cfg, self.layout, self.hits = cfg, layout, {"vit": 0, "emb": 0}

    def perm(self):
        c, r, thumb = self.layout
        return PM.multitile_perm(self.in_tile, c, r, thumb, self.cfg.get("tile", "tile"))

    def _pe(self, embeddings, height, width):
        pe = self._pe0(embeddings, height, width)
        self.hits["vit"] += 1
        return PM.abs_pe_remap(pe, self.cfg["vh"], self.cfg["vw"]) if (self.cfg["vh"] or self.cfg["vw"]) else pe

    def _feat(self, *a, **k):
        import torch
        out = self._feat0(*a, **k)                       # (tiles, 256, D)
        if self.cfg["emb"]:
            self.hits["emb"] += 1
            n, t, d = out.shape
            p = torch.tensor(self.perm(), device=out.device)
            out = out.reshape(n * t, d)[p].reshape(n, t, d)
        return out

    def positions(self, input_ids):
        import torch
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
        if not self.cfg["pos"]:
            return pos
        p = torch.tensor(self.perm(), device=input_ids.device)
        for i in range(input_ids.shape[0]):
            spans = self.P.image_spans(input_ids[i], self.tok)
            assert len(spans) == 1 and spans[0][1] - spans[0][0] == len(p), f"image span {spans} vs permutation {len(p)}"
            s, e = spans[0]
            pos[i, s:e] = s + p
        return pos


class Q3Ctrl:
    """Qwen3-VL: the vision encoder's 2D-RoPE (rot_pos_emb) and absolute embedding (fast_pos_embed_interpolate) both
    permuted by vlmlr.permutations.qwen3_vit_perm; LLM-side positions (get_rope_index), visual tokens and DeepStack
    features (get_image_features) permuted on the merged grid as for Qwen2.5-VL."""

    def __init__(self, model):
        self.inner, self.visual = model.model, model.model.visual
        self.tok = model.config.image_token_id
        self.merge = self.visual.spatial_merge_size
        self.cfg, self.hits = _c(), {"rot": 0, "abs": 0, "emb": 0, "pos": 0}
        self._rot0, self.visual.rot_pos_emb = self.visual.rot_pos_emb, self._rot
        self._abs0, self.visual.fast_pos_embed_interpolate = self.visual.fast_pos_embed_interpolate, self._abs
        self._feat0, self.inner.get_image_features = self.inner.get_image_features, self._feat
        self._rope0, self.inner.get_rope_index = self.inner.get_rope_index, self._rope

    def set(self, cfg):
        self.cfg, self.hits = cfg, {"rot": 0, "abs": 0, "emb": 0, "pos": 0}

    def _vperm(self, grid_thw, device):
        import torch
        vh, vw = self.cfg["vh"], self.cfg["vw"]
        axes = ("h" if vh else "") + ("w" if vw else "")
        ps, off = [], 0
        for t, h, w in grid_thw.tolist():
            assert t == 1, "single-frame images only"
            if vh in ("", "mirror") and vw in ("", "mirror"):
                p = PM.qwen3_vit_perm(h, w, self.merge, axes)
            else:                        # same-axis permutations (qwen3-permutations)
                p = PM.qwen3_vit_perm_kinds(h, w, self.merge, vh, vw)
            ps.append(torch.tensor(p) + off)
            off += h * w
        return torch.cat(ps).to(device)

    def _rot(self, grid_thw):
        out = self._rot0(grid_thw)
        if self.cfg["vh"] or self.cfg["vw"]:
            self.hits["rot"] += 1
            out = out[self._vperm(grid_thw, out.device)]
        return out

    def _abs(self, grid_thw):
        out = self._abs0(grid_thw)
        if self.cfg["vh"] or self.cfg["vw"]:
            self.hits["abs"] += 1
            out = out[self._vperm(grid_thw, out.device)]
        return out

    def _feat(self, pixel_values, image_grid_thw=None):
        import torch
        from vlmlr import positions as P
        embeds, deep = self._feat0(pixel_values, image_grid_thw)
        if self.cfg["emb"]:
            self.hits["emb"] += 1
            perms = P.image_permutations(image_grid_thw, self.merge, self.cfg["emb"])
            embeds = P.permute_embeds(embeds, perms)
            full, off = [], 0
            for p in perms:
                full.append(p.to(deep[0].device) + off)
                off += len(p)
            full = torch.cat(full)
            deep = [d[full] for d in deep]
        return embeds, deep

    def _rope(self, input_ids=None, image_grid_thw=None, *a, **k):
        from vlmlr import positions as P
        pos, deltas = self._rope0(input_ids, image_grid_thw, *a, **k)
        if self.cfg["pos"] and input_ids is not None:
            self.hits["pos"] += 1
            pos = P.permute_positions(pos, input_ids, self.tok, P.image_permutations(image_grid_thw, self.merge, self.cfg["pos"]))
        return pos, deltas


def load_model(m, a):
    import torch
    import transformers
    from vlmlr import models
    fam = FAMILY[m]
    if fam == "qwen":
        return models.load_qwen(REPO[m], max_pixels=a.max_pixels)
    if fam == "internvl":
        return models.load_internvl(REPO[m])
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        REPO[m], torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    proc = transformers.AutoProcessor.from_pretrained(REPO[m])
    ip = proc.image_processor
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size = {"shortest_edge": ip.size["shortest_edge"], "longest_edge": a.max_pixels_q3}
    print(f"[load] Qwen3-VL processor size={ip.size} patch={ip.patch_size} merge={ip.merge_size}", flush=True)
    return model, proc


def run(a):
    import torch
    from experiments import controls_permutation as CP
    from vlmlr import models
    from vlmlr.cvbench import transform
    if (a.exp, a.data) not in PLAN:
        raise SystemExit(f"no plan for {a.exp} x {a.data}")
    fam = FAMILY[a.model]
    table = PLAN[(a.exp, a.data)]
    items = CP.pick(CP.load_items(a.data, a), a.limit)
    items = [x for x in items if conds_for(a.exp, a.data, x["axis"])]
    print(f"[items] {a.exp} {a.data} {a.model}: {len(items)} items", flush=True)
    model, proc = load_model(a.model, a)
    n_opt = 4 if a.data == "whatsup" else 2
    letters = models.letter_ids(proc.tokenizer, "ABCD"[:n_opt])
    fp32 = models.FP32OptionScorer(model, letters)
    dose = H = None
    if a.exp == "block-sets":
        from vlmlr.hooks import QwenLayerwiseMirror
        dose = QwenLayerwiseMirror(model)
    elif a.exp == "internvl-tiling":
        H = InternMulti(model)
    elif fam == "qwen":
        H = CP.QwenCtrl(model)
    elif fam == "internvl":
        H = CP.InternCtrl(model)
    else:
        H = Q3Ctrl(model)

    def build(img, it):
        text = models.chat_text(proc, it["prompt"])
        extra = {"crop_to_patches": a.exp == "internvl-tiling"} if fam == "internvl" else {}
        inp = proc(text=[text], images=[img], return_tensors="pt", **extra).to("cuda")
        layout = None
        if a.exp == "internvl-tiling":
            c, r = PM.tile_grid(img.size[1], img.size[0])
            thumb = c * r > 1
            n_t = c * r + thumb
            assert inp["pixel_values"].shape[0] == n_t, f"{inp['pixel_values'].shape[0]} tiles, re-created {c}x{r}+{int(thumb)}"
            assert int((inp["input_ids"] == model.config.image_token_id).sum()) == 256 * n_t
            layout = (c, r, thumb)
        return inp, layout

    def fwd(inp, layout, cfg):
        try:
            if a.exp == "block-sets":
                blocks = cfg.get("blocks", frozenset())
                dose.set(axes="w" if blocks else "", layers=blocks)
                model.model.rope_deltas = None
                lg = model(**inp).logits[0, -1].float()
                hits = {"vit": dose.vit_swap.hits}
            elif a.exp == "internvl-tiling":
                H.set(cfg, layout)
                lg = model(**inp, position_ids=H.positions(inp["input_ids"])).logits[0, -1].float()
                hits = dict(H.hits)
            elif fam == "internvl":
                H.set(cfg)
                lg = model(**inp, position_ids=H.positions(inp["input_ids"])).logits[0, -1].float()
                hits = dict(H.hits)
            else:
                H.set(cfg)
                model.model.rope_deltas = None
                lg = model(**inp).logits[0, -1].float()
                hits = dict(H.hits)
        finally:
            if dose is not None:
                dose.reset()
            elif a.exp == "internvl-tiling":
                H.set(_c(), layout)
            else:
                H.set(_c())
        return lg[letters], fp32(), hits

    def self_check(inp, layout, names, tfcache):
        o1, f1, _ = fwd(inp, layout, _c())
        o2, f2, _ = fwd(inp, layout, _c())
        assert torch.equal(o1, o2) and torch.equal(f1, f2), "orig is not reproducible"
        unchanged = []
        if a.exp == "block-sets":
            allb, _, h = fwd(inp, layout, _c(blocks=frozenset(range(dose.n_vit))))
            dose.set(axes="w", full=True)
            model.model.rope_deltas = None
            try:
                full = model(**inp).logits[0, -1].float()[letters]
            finally:
                dose.reset()
            assert float((allb - full).abs().max()) <= float((o1 - o2).abs().max()) + 1e-6, "all blocks differ from the whole-model mirror"
            assert h["vit"] == dose.n_vit, f"{h['vit']} blocks replaced, expected {dose.n_vit}"
        if fam in ("qwen", "qwen3") and a.exp != "block-sets":
            o3, _, h3 = fwd(inp, layout, dict(_c(), emb="id", pos="id"))
            assert torch.equal(o3, o1) and h3["emb"] == 1 and h3["pos"] == 1, "identity permutation through the hooks differs"
        if fam == "qwen3":
            g = inp["image_grid_thw"]
            for axes in ("w", "h"):
                p = PM.qwen3_vit_perm(int(g[0, 1]), int(g[0, 2]), H.merge, axes)
                assert sorted(p) == list(range(len(p))) and all(p[p[k]] == k for k in range(len(p))), "ViT permutation is not an involution"
        for n in names:
            cfg = table[n]
            if n == "orig" or (cfg["tf"] and cfg == _c(tf=cfg["tf"], axes=cfg["axes"])):
                continue
            if cfg["tf"]:
                ti, tl = tfcache(cfg["tf"])
                base, _, _ = fwd(ti, tl, _c(tf=cfg["tf"]))
                o, _, h = fwd(ti, tl, cfg)
            else:
                base = o1
                o, _, h = fwd(inp, layout, cfg)
            if cfg["vh"] or cfg["vw"]:
                assert sum(h.get(k, 0) for k in ("vit", "rot", "abs")) >= 1, f"{n} did not reach the ViT hook"
            if a.exp == "qwen3-permutations" and (cfg["vh"] or cfg["vw"]) not in ("", "mirror"):
                g = inp["image_grid_thw"]
                gh, gw = int(g[0, 1]), int(g[0, 2])
                pp = PM.qwen3_vit_perm_kinds(gh, gw, H.merge, cfg["vh"], cfg["vw"])
                assert pp != list(range(gh * gw)), f"{n} is the identity"
                assert pp != PM.qwen3_vit_perm(gh, gw, H.merge, "h" if cfg["vh"] else "w"), f"{n} equals the mirror"
                assert h["rot"] >= 1 and h["abs"] >= 1, f"{n} did not reach both ViT position encodings: {h}"
                if torch.equal(o, base):        # a cyclic shift can be a weak edit under RoPE: noted, not an error
                    unchanged.append(n)
                continue
            if a.exp == "block-sets" and n not in ("blk_all", "blk_g4", "blk_b31"):
                if torch.equal(o, base):
                    unchanged.append(n)
                continue
            assert not torch.equal(o, base), f"{n} changed nothing"
        print(f"[self-check] passed; conditions that left the output of the first item unchanged: {unchanged or 'none'}", flush=True)

    done = {(r["idx"], r["cond"]) for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n, checked = time.time(), 0, False
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            names = [c for c in conds_for(a.exp, a.data, it["axis"]) if (it["idx"], c) not in done]
            if not names:
                continue
            img0, cache = CP.image_of(it), {}

            def tfcache(tf, img0=img0, it=it, cache=cache):
                if tf not in cache:
                    cache[tf] = build(transform(img0, tf), it)
                return cache[tf]
            for name in names:
                cfg = table[name]
                inp, layout = tfcache(cfg["tf"])
                if not checked:
                    i0, l0 = tfcache(None)
                    self_check(i0, l0, conds_for(a.exp, a.data, it["axis"]), tfcache)
                    checked = True
                bl, fl, _ = fwd(inp, layout, cfg)
                lp = torch.log_softmax(bl, -1).tolist()
                lp32 = torch.log_softmax(fl, -1).tolist()
                rec = {"idx": it["idx"], "cond": name, "exp": a.exp, "data": a.data, "model": a.model,
                       "axis": it["axis"], "gold": it["gold"], "lp": lp, "lp32": lp32,
                       "n_img": int((inp["input_ids"] == model.config.image_token_id).sum())}
                if layout:
                    rec["tiles"] = list(layout)
                if a.exp == "qwen3-permutations":             # image size and patch grid: merge units of the object-order split
                    rec.update(W=img0.size[0], H=img0.size[1], grid=[int(x) for x in inp["image_grid_thw"][0, 1:].tolist()])
                if a.data == "cv2d":
                    rec.update(choices=it["choices"], pred=it["choices"][int(lp[1] > lp[0])], tie=lp[0] == lp[1])
                else:
                    l4 = bl.tolist()
                    rec.update(rels=it["rels"], logits=l4, pred=it["rels"][max(range(4), key=lambda i: l4[i])])
                f.write(json.dumps(rec) + "\n")
                n += 1
            f.flush()
    print(f"[run] done {n} forwards in {time.time() - t0:.0f}s -> {a.out}", flush=True)


def run_option_order(a):
    """experiments.main_table run for InternVL3-2B with the options of every condition in reverse order
    (--order fwd keeps the order, to check the reproduction of the main run)."""
    from experiments import main_table
    from vlmlr import cvbench as C
    variants0 = C.variants

    def swapped(item):
        rows = variants0(item)
        if a.order == "swap":
            for r in rows:
                r["choices"] = list(r["choices"])[::-1]
        return rows
    C.variants = swapped
    try:
        main_table.run(argparse.Namespace(family="internvl", model=REPO["ivl2b"], parquet=a.parquet, out=a.out, limit=a.limit))
    finally:
        C.variants = variants0


def run_depth_all(a):
    """Base model on all CV-Bench 3D Depth items of the given item tables (validation and held-out half), under
    orig, vit_h and vit_w; fp32 option scores from the lm_head input; one record per item and condition."""
    import torch
    from PIL import Image
    from vlmlr import cvbench as C
    from vlmlr import models
    from vlmlr.hooks import QwenPositionMirror
    model, proc = models.load_qwen(a.model, min_pixels=a.min_pixels, max_pixels=a.max_pixels)
    mirror = QwenPositionMirror(model)
    ids = models.letter_ids(proc.tokenizer, "AB")
    fp32 = models.FP32OptionScorer(model, ids)
    items = [it for p in a.items.split(",") for it in read_jsonl(p) if it["task"] == "Depth"]
    items = items[: a.limit] if a.limit else items
    done = {(r["idx"], r["cond"]) for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.inference_mode():
        for c, vit in (("orig", ""), ("vit_h", "h"), ("vit_w", "w")):         # all orig first, then vit_h, then vit_w
            for it in items:
                if (it["idx"], c) in done:
                    continue
                img = Image.open(os.path.join(a.image_root, it["img"])).convert("RGB")
                inp = models.build_inputs(proc, img, C.prompt(it["question"], it["choices"], False))
                gh, gw = inp["image_grid_thw"][0, 1:].tolist()
                assert (gw * 14, gh * 14) == img.size, f"the processor resized the cached image ({img.size} vs {gh}x{gw})"
                mirror.vit = vit
                model.model.rope_deltas = None
                try:
                    model(**inp, logits_to_keep=1)
                    lg = fp32().tolist()
                finally:
                    mirror.vit = ""
                pred = None if lg[0] == lg[1] else int(lg[1] > lg[0])
                f.write(json.dumps({"idx": it["idx"], "cond": c, "gold": it["gold"], "group": it["group"],
                                    "lg": lg, "pred": pred}) + "\n")
                f.flush()
                n += 1
    print(f"[run] done {n} forwards in {time.time() - t0:.0f}s -> {a.out}", flush=True)


# ================================================================ analyses: helpers

def _read(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
    return read_jsonl(p)


def _index(recs, key="cond"):
    d = collections.defaultdict(dict)
    for r in recs:
        d[r["idx"]][r.get(key) or r.get("variant") or r.get("cond")] = r
    return d


def _run_file(root, run, name):
    """Per-item predictions of one GPU run; a missing file is an error (the predictions are not shipped)."""
    p = os.path.join(root, RUN_DIR[run], name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
    return p


def pct(x):
    return "  nan" if x != x else f"{100 * x:5.1f}"


def boot(stat, ids, B_=10000, seed=0):
    ids = sorted(ids)
    rng = random.Random(seed)
    n = len(ids)
    vals = sorted(stat([ids[rng.randrange(n)] for _ in range(n)]) for _ in range(B_))
    return stat(ids), (vals[int(0.025 * B_)], vals[int(0.975 * B_) - 1])


def boot2(f1, ids1, f2, ids2, B_=10000, seed=0):
    """Unpaired: both groups resampled separately. Returns f1(ids1) - f2(ids2) and its interval."""
    rng = random.Random(seed)
    i1, i2 = sorted(ids1), sorted(ids2)
    vals = sorted(f1([i1[rng.randrange(len(i1))] for _ in i1]) - f2([i2[rng.randrange(len(i2))] for _ in i2])
                  for _ in range(B_))
    return f1(i1) - f2(i2), (vals[int(0.025 * B_)], vals[int(0.975 * B_) - 1])


def _repro_line(new, ref, pairs, L, tag):
    ok = True
    for mine, theirs in pairs.items():
        ks = [i for i in new if mine in new[i] and theirs in ref.get(i, {})]
        if not ks:
            continue
        same = sum(new[i][mine]["pred"] == ref[i][theirs]["pred"] for i in ks)
        good = same / len(ks) >= REPRO_MIN
        ok &= good
        L.append(f"  reproduction {tag} {mine}<->main run {theirs}: {same}/{len(ks)}" + ("" if good else "  <- not reproduced on every item, section skipped"))
    return ok


def _flip(d, ids, c, base="orig"):
    return sum(d[i][c]["pred"] != d[i][base]["pred"] for i in ids) / len(ids) if ids else float("nan")


def _exp_gold(v, pix):
    g, ax = v["orig"]["gold"], v["orig"]["axis"]
    if (pix == "hflip" and ax == "lr") or (pix == "vflip" and ax == "ud"):
        return INV.get(g, g)
    return g


def _gc_new(d, axis, pix):
    """Geometry-consistent items of this run: correct on orig and on the real flip."""
    return sorted(i for i, v in d.items() if "orig" in v and pix in v and v["orig"]["axis"] == axis
                  and v["orig"]["pred"] == v["orig"]["gold"] and v[pix]["pred"] == _exp_gold(v, pix))


# ================================================================ gpu-summary

def _sec_vertical(a, L, R):
    L.append("== What'sUp, vertical content-only control (flip-back rate = answer under pix_both differs from the answer "
             "under pix, all items of the axis; D = FB_h − FB_v, unpaired bootstrap)")
    for m in ("q3b", "q7b", "ivl2b"):
        new = _index(_read(_run_file(a.dir, "vertical-content", f"{m}_whatsup.jsonl")))
        if not new:
            continue
        ref = _index(_read(MAIN_WH[m]), "variant")
        if not _repro_line(new, ref, {"orig": "orig", "hflip": "hflip", "vflip": "vflip"}, L, m):
            continue
        ud = [i for i, v in new.items() if v["orig"]["axis"] == "ud" and "vflip_both_h" in v]
        lr = [i for i, v in new.items() if v["orig"]["axis"] == "lr" and "hflip_both_w" in v]
        fb = lambda c, pix: (lambda I: sum(new[i][c]["pred"] != new[i][pix]["pred"] for i in I) / len(I))
        fv, fh = fb("vflip_both_h", "vflip"), fb("hflip_both_w", "hflip")
        D, ci = boot2(fh, lr, fv, ud, a.B)
        R.setdefault("vertical_content", {})[m] = {"FB_h": fh(lr), "FB_v": fv(ud), "D": D, "D_ci": ci}
        L.append(f"  {m:5s} up-down n={len(ud)}: FB_v {pct(fv(ud))} (vit_h alone {pct(fb('vflip_vit_h', 'vflip')(ud))}, "
                 f"llm_h alone {pct(fb('vflip_llm_h', 'vflip')(ud))}); left-right n={len(lr)}: FB_h {pct(fh(lr))}; "
                 f"D {pct(D)} [{pct(ci[0]).strip()}, {pct(ci[1]).strip()}]")
        G = _gc_new(new, "ud", "vflip")
        if G:
            back = sum(new[i]["vflip_both_h"]["pred"] == new[i]["orig"]["pred"] for i in G) / len(G)
            L.append(f"        GC up-down n={len(G)}: vflip_both_h back to the orig answer {pct(back)}")


def _sec_blocks(a, L, R):
    L.append("\n== Width mirror in chosen sets of ViT blocks (GC left-right items = main-run orig and hflip both "
             "correct; flips relative to the main-run orig; paired bootstrap)")
    for m in ("q7b", "q3b"):
        new = _index(_read(_run_file(a.dir, "block-sets", f"{m}_cv2d.jsonl")))
        if not new:
            continue
        blk = _index(_read(MAIN_BLK[m]))
        p1 = _index(_read(MAIN_CV[m]), "variant")
        if not _repro_line(new, blk, dict(BLOCKS_REPRO, orig="base_orig"), L, m):
            continue
        G = [i for i in sorted(new) if i in p1 and p1[i]["orig"]["axis"] == "lr"
             and p1[i]["orig"]["pred"] == p1[i]["orig"]["expected"] and p1[i]["hflip"]["pred"] == p1[i]["hflip"]["expected"]]
        o = {i: p1[i]["orig"]["pred"] for i in G}
        fr = lambda c: (lambda I: sum(new[i][c]["pred"] != o[i] for i in I) / len(I))
        r = {k: fr(f"blk_{k}")(G) for k in block_sets()}
        r["ratio_g4_all"], r["ratio_g4_all_ci"] = boot(lambda I: fr("blk_g4")(I) / max(fr("blk_all")(I), 1e-9), G, a.B)
        r["d_last8_b31"], r["d_last8_b31_ci"] = boot(lambda I: fr("blk_last8")(I) - fr("blk_b31")(I), G, a.B)
        R.setdefault("block_sets", {})[m] = r
        L.append(f"  {m} n={len(G)}: " + "  ".join(f"{k} {pct(r[k]).strip()}" for k in block_sets()))
        L.append(f"      g4/all {r['ratio_g4_all']:.2f} [{r['ratio_g4_all_ci'][0]:.2f}, {r['ratio_g4_all_ci'][1]:.2f}]; "
                 f"last8 − b31 {pct(r['d_last8_b31'])} [{pct(r['d_last8_b31_ci'][0]).strip()}, {pct(r['d_last8_b31_ci'][1]).strip()}]")


def _sec_tiling(a, L, R):
    new = _index(_read(_run_file(a.dir, "internvl-tiling", "ivl2b_cv2d.jsonl")))
    if not new:
        return
    L.append("\n== InternVL3-2B, default tiling (GC left-right items re-determined from the orig and hflip of this "
             "tiled run; flips relative to this run's orig)")
    G = _gc_new(new, "lr", "hflip")
    tiles = collections.defaultdict(int)
    for v in new.values():
        tiles[tuple(v["orig"].get("tiles", []))] += 1
    L.append(f"  left-right items {sum(v['orig']['axis'] == 'lr' for v in new.values())}, GC {len(G)}; tile layouts "
             f"(columns, rows, thumbnail) "
             + str(dict(sorted(tiles.items(), key=lambda t: -t[1])[:6])))
    for c in ("vit_w_tile", "llm_w_tile", "llm_w_img", "order_only_w_tile", "order_only_w_img"):
        k = sum(new[i][c]["pred"] != new[i]["orig"]["pred"] for i in G)
        p, ci = wilson(k, len(G))
        R.setdefault("internvl_tiling", {})[c] = {"k": k, "n": len(G), "rate": p, "ci": ci}
        L.append(f"    {c:18s} {k}/{len(G)} {pct(p)} [{pct(ci[0]).strip()}, {pct(ci[1]).strip()}]")


def _sec_option_order(a, L, R):
    sw = C.main_rows(_read(_run_file(a.dir, "run-option-order", "ivl2b_swap.jsonl")))
    if not sw:
        return
    L.append("\n== InternVL3-2B, options in reverse order (measured as in the Qwen2.5-VL run with reversed options)")
    orig = _index(C.main_rows(_read(MAIN_CV["ivl2b"])), "variant")
    swp = _index(sw, "variant")
    fwd = C.main_rows(_read(_run_file(a.dir, "run-option-order", "ivl2b_fwd.jsonl")))
    if fwd:
        same = sum(r["pred"] == orig[r["idx"]][r["variant"]]["pred"] for r in fwd)
        L.append(f"  reproduction (original order, first 3 items): {same}/{len(fwd)} as in the main run"
                 + ("" if same >= REPRO_MIN * len(fwd) else "  <- not reproduced on every item"))
    if len(swp) < 650:
        L.append(f"  (reversed order: only {len(swp)} items)")
        return

    def q(d):
        def G(ax, pix):
            return [i for i, v in d.items() if v["orig"]["axis"] == ax and v["orig"]["pred"] == v["orig"]["expected"]
                    and v[pix]["pred"] == v[pix]["expected"]]
        lr, ud = G("lr", "hflip"), G("ud", "vflip")
        alr = [i for i, v in d.items() if v["orig"]["axis"] == "lr"]
        f = lambda I, c: sum(d[i][c]["pred"] != d[i]["orig"]["pred"] for i in I) / len(I)
        return {"agree_vit_w_hflip_lr": sum(d[i]["vit_w"]["pred"] == d[i]["hflip"]["pred"] for i in alr) / len(alr),
                "flip_vit_w_lr": f(lr, "vit_w"), "flip_llm_w_lr": f(lr, "llm_w"), "flip_vit_h_ud": f(ud, "vit_h"),
                "flip_llm_h_ud": f(ud, "llm_h"), "flip_vit_w_ud": f(ud, "vit_w")}
    qo, qs = q(orig), q(swp)
    big = [k for k in qo if abs(qs[k] - qo[k]) > 0.05]
    R["internvl_option_order"] = {"orig": qo, "swap": qs, "over_5_points": big}
    for k in qo:
        L.append(f"  {k:22s} original {pct(qo[k])}  reversed {pct(qs[k])}  diff {pct(qs[k] - qo[k])}")
    L.append("  quantities differing by more than 5 points between the two orders: " + ("none" if not big else ", ".join(big)))
    ties = sum(1 for r in sw if r["logp"][0] == r["logp"][1])
    L.append(f"  reversed order: {ties} ties (of {len(sw)} rows)")


def _sec_qwen3(a, L, R):
    for data in ("cv2d", "whatsup"):
        new = _index(_read(_run_file(a.dir, "qwen3-mirrors", f"q3vl4b_{data}.jsonl")))
        if not new:
            continue
        L.append(f"\n== Qwen3-VL-4B, {data} (GC re-determined from this run's orig and pixel flip; flips relative to this run's orig)")
        G = _gc_new(new, "lr", "hflip")
        alr = [i for i, v in new.items() if v["orig"]["axis"] == "lr"]
        L.append(f"  left-right items {len(alr)}, GC {len(G)}; orig accuracy (left-right) "
                 f"{pct(sum(new[i]['orig']['pred'] == new[i]['orig']['gold'] for i in alr) / len(alr))}")
        for c in ("vit_w", "llm_w", "order_only_w", "order_and_id_w"):
            if not G or c not in new[G[0]]:
                continue
            k = sum(new[i][c]["pred"] != new[i]["orig"]["pred"] for i in G)
            p, ci = wilson(k, len(G))
            R.setdefault("qwen3_mirrors", {}).setdefault(data, {})[c] = {"k": k, "n": len(G), "rate": p, "ci": ci}
            L.append(f"    {c:15s} {k}/{len(G)} {pct(p)} [{pct(ci[0]).strip()}, {pct(ci[1]).strip()}]")
        if alr and "vit_w" in new[alr[0]]:
            ag = sum(new[i]["vit_w"]["pred"] == new[i]["hflip"]["pred"] for i in alr) / len(alr)
            ag0 = sum(new[i]["orig"]["pred"] == new[i]["hflip"]["pred"] for i in alr) / len(alr)
            L.append(f"    vit_w agrees with hflip {pct(ag)} (orig with hflip {pct(ag0)})")
        if alr and "hflip_both_w" in new[alr[0]]:
            fb = sum(new[i]["hflip_both_w"]["pred"] != new[i]["hflip"]["pred"] for i in alr) / len(alr)
            L.append(f"    hflip_both_w flip-back rate {pct(fb)}")
        Gu = _gc_new(new, "ud", "vflip")
        for c in ("vit_h", "llm_h", "vit_w", "llm_w"):
            if Gu and c in new[Gu[0]]:
                L.append(f"    up-down GC n={len(Gu)}: {c} flips {pct(_flip(new, Gu, c))}")


def gpu_summary(a):
    L, R = [], {}
    for f in (_sec_vertical, _sec_blocks, _sec_tiling, _sec_option_order, _sec_qwen3):
        f(a, L, R)
    print("\n".join(L))
    if a.json:
        json.dump(R, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


# ================================================================ box-reclassification (RefCOCO, post hoc)

def _wilson3(k, n, z=1.96):
    p, (lo, hi) = wilson(k, n, z)
    return p, lo, hi


def _paired_boot(x, y, B=10000, seed=0):
    n = len(x)
    rng = random.Random(seed)
    d = [u - v for u, v in zip(x, y)]
    vals = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def box_reclassification(a):
    """Boxes under vit_w / llm_w classified as switch (the other object), same, hflip_raw (coincides, IoU >= 0.5, with
    the box the model gives for the flipped image, in raw coordinates), mirror_coords, or unmatched (none of these);
    none: no box in the output. Each box gets the first class that fits, in the order switch, same, hflip_raw,
    mirror_coords; the cross table lists every combination of classes that fit."""
    from experiments.extra import box_readout as L
    fmt = lambda p, lo, hi: f"{100 * p:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    out = []
    for m in ("7b", "3b"):
        d = collections.defaultdict(dict)
        for r in read_jsonl(getattr(a, f"m{m}")):
            d[r["id"]][r["cond"]] = r
        st = collections.Counter()
        ok, wide = [], []
        for k, v in sorted(d.items()):
            o = v["orig"]
            boxes = {c: v[c]["box"] for c in L.CONDS}
            s, A, Bx = L.item_status(boxes, o["gt"], o["W"])
            st[s] += 1
            if s != "orig_wrong":
                wide.append((k, boxes, A, Bx, o["W"]))
            if s == "ok":
                ok.append((k, boxes, A, Bx, o["W"]))
        out.append(f"== {m}: items {len(d)}; status " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
        raw = lambda box, hb: L.iou(box, hb) >= L.THR
        for c in ("vit_w", "llm_w"):
            main_cnt, multi = collections.Counter(), collections.Counter()
            for k, bx, A, Bx, W in ok:
                box, hb = bx[c], bx["hflip"]
                if box is None:
                    main_cnt["none"] += 1
                    continue
                hit = {"switch": L.iou(box, Bx) >= L.THR, "same": L.iou(box, A) >= L.THR,
                       "hflip_raw": raw(box, hb), "mirror_coords": L.iou(box, L.mirror(A, W)) >= L.THR}
                order = ("switch", "same", "hflip_raw", "mirror_coords")
                main_cnt[next((x for x in order if hit[x]), "unmatched")] += 1
                multi["+".join(x for x in order if hit[x]) or "unmatched"] += 1
            n = len(ok)
            show = lambda C: "  ".join(f"{x} {C[x]} ({100 * C[x] / n:.1f}%)" for x in
                                       ("switch", "same", "hflip_raw", "mirror_coords", "unmatched", "none") if C[x])
            out.append(f"  {c} usable items {n}")
            out.append(f"    classes (first that fits: switch, same, hflip_raw, mirror_coords; else unmatched): {show(main_cnt)}")
            out.append("    combinations of classes that fit (cross table): " + "  ".join(f"{k} {v}" for k, v in multi.most_common()))
        for name, items in (("usable items", ok), ("all items with a correct orig box", wide)):
            x = {c: [int(raw(bx[c], bx["hflip"])) for k, bx, A, Bx, W in items] for c in ("orig", "vit_w", "llm_w")}
            n = len(items)
            out.append(f"  coincides with the raw H-flip box (IoU ≥ 0.5), {name} n={n}: " + "  ".join(
                f"{c} {sum(x[c])}/{n} {fmt(*_wilson3(sum(x[c]), n))}" for c in ("orig", "vit_w", "llm_w")))
            for c2 in ("orig", "llm_w"):
                dlt, lo, hi = _paired_boot(x["vit_w"], x[c2])
                out.append(f"    vit_w − {c2}: {100 * dlt:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] points (paired bootstrap B=10000)")
    print("\n".join(out))


# ================================================================ object-boxes / object-order (post hoc)
OBJ_PAT = re.compile(r"relative positions of the (.+?) and the (.+?) in the image")
RED = " (annotated by the red box)"
SYN = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "tv monitor": "tv",
       "television": "tv", "dining table": "dining table", "hand bag": "handbag"}
PERIM_MIN = 0.6


GRID_UNITS = {}      # model -> {(W, H): (width units, height units)} read from the run records (Qwen3-VL-4B)


def merge_units(model, W, H):
    """(units along the width, units along the height) of the merged grid the model sees for a W x H image."""
    from vlmlr.geometry import smart_resize
    if model in GRID_UNITS:
        return GRID_UNITS[model][(W, H)]
    if model == "ivl2b":
        return 16, 16
    hb, wb = smart_resize(H, W, 28, 56 * 56, 1280 * 28 * 28)
    return wb // 28, hb // 28


def _red_mask(img, d=3):
    import numpy as np
    x = np.asarray(img.convert("RGB")).astype(int)
    m = (x[..., 0] > 180) & (x[..., 1] < 60) & (x[..., 2] < 60)
    M = m.copy()
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            M |= np.roll(np.roll(m, dy, 0), dx, 1)
    return M


def _perim_red(M, b):
    H, W = M.shape
    x0, y0, x1, y1 = [int(round(v)) for v in b]
    x0, x1 = max(0, min(W - 1, x0)), max(0, min(W - 1, x1 - 1))
    y0, y1 = max(0, min(H - 1, y0)), max(0, min(H - 1, y1 - 1))
    pts = ([(x, y0) for x in range(x0, x1 + 1)] + [(x, y1) for x in range(x0, x1 + 1)]
           + [(x0, y) for y in range(y0, y1 + 1)] + [(x1, y) for y in range(y0, y1 + 1)])
    return sum(bool(M[y, x]) for x, y in pts) / len(pts)


def object_boxes(a):
    """COCO boxes (instances_val2017.json, scaled to the CV-Bench image) of the two objects of every CV-Bench COCO
    question. The subject A must be unique, or, on questions with a red box, the instance whose box edges are red
    (share of red pixels on the edges >= 0.6, and no second instance as red); every instance of the reference
    object B is kept. One JSON line per item."""
    import pyarrow.parquet as pq
    from PIL import Image
    A = json.load(open(a.ann))
    cats = {c["name"]: c["id"] for c in A["categories"]}
    imgs = {im["id"]: im for im in A["images"]}
    by_img = {}
    for o in A["annotations"]:
        by_img.setdefault(o["image_id"], []).append(o)

    def norm(name):
        n = name.replace(RED, "").strip().lower()
        n = SYN.get(n, n)
        if n in cats:
            return n
        return n[:-1] if n.endswith("s") and n[:-1] in cats else None
    rows = [r for r in pq.read_table(a.parquet).to_pylist() if r["task"] == "Relation" and r["source"] == "COCO"]
    drop = collections.Counter()
    n_ok = 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            img = Image.open(io.BytesIO(r["image"]["bytes"]))
            W, H = img.size
            iid = int(re.search(r"(\d+)\.jpg", r["source_filename"]).group(1))
            im = imgs.get(iid)
            if im is None:
                drop["image not in COCO"] += 1
                continue
            sx, sy = W / im["width"], H / im["height"]
            an, bn = OBJ_PAT.search(r["question"]).groups()
            ca, cb = norm(an), norm(bn)
            if ca is None or cb is None:
                drop["class name not in COCO"] += 1
                continue

            def inst(cn):
                return [[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for o in by_img.get(iid, [])
                        if o["category_id"] == cats[cn] and not o.get("iscrowd") for x, y, w, h in [o["bbox"]]]
            IA, IB = inst(ca), inst(cb)
            rec = {"idx": r["idx"], "W": W, "H": H, "coco_id": iid, "A": ca, "B": cb, "red": RED.strip() in an}
            if rec["red"]:
                if not IA:
                    drop["no instance of A"] += 1
                    continue
                M = _red_mask(img)
                sc = sorted(((_perim_red(M, b), b) for b in IA), key=lambda t: -t[0])
                rec["perim"] = sc[0][0]
                if sc[0][0] < PERIM_MIN or (len(sc) > 1 and sc[1][0] >= PERIM_MIN):
                    drop["red box not matched to exactly one instance of A"] += 1
                    continue
                rec["boxA"] = sc[0][1]
            else:
                if len(IA) != 1:
                    drop["no red box and A not unique"] += 1
                    continue
                rec["boxA"] = IA[0]
            if not IB:
                drop["no instance of B"] += 1
                continue
            rec["boxesB"] = IB
            f.write(json.dumps(rec) + "\n")
            n_ok += 1
    print(f"== Boxes of the two objects: CV-Bench 2D Relation COCO {len(rows)} items -> {n_ok} mapped")
    print("  dropped: " + ("; ".join(f"{k} {v}" for k, v in sorted(drop.items())) or "none"))


def _order_status(rec, model, axis, kind):
    """'rev' / 'keep' (the permutation reverses / keeps the order of the two objects' merge units), 'samecol', or
    'ambig' (several instances of B disagree)."""
    Uw, Uh = merge_units(model, rec["W"], rec["H"])
    U, size, k = (Uw, rec["W"], 0) if axis == "lr" else (Uh, rec["H"], 1)
    um = PM.unit_map(U, kind)
    cell = lambda b: min(U - 1, max(0, int(((b[k] + b[k + 2]) / 2) / size * U)))
    ua = cell(rec["boxA"])
    out = set()
    for b in rec["boxesB"]:
        ub = cell(b)
        out.add("samecol" if ua == ub else ("rev" if (ua - ub) * (um[ua] - um[ub]) < 0 else "keep"))
    return out.pop() if len(out) == 1 else "ambig"


def _cluster_boot(by_item, stat, B=10000, seed=0):
    ids = sorted(by_item)
    rng = random.Random(seed)
    n = len(ids)
    vals = []
    for _ in range(B):
        v = stat([ids[rng.randrange(n)] for _ in range(n)])
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def _rate(pairs):
    return sum(f for _, f in pairs) / len(pairs) if pairs else float("nan")


def _order_analyze(items, new, ref, model, axis, kinds):
    from experiments import controls_permutation as CP
    a_, off = ("w", "h") if axis == "lr" else ("h", "w")
    G = [i for i in CP.geo_subset(ref, axis) if i in new and i in items]
    o = {i: ref[i]["orig"]["pred"] for i in G}
    pairs = collections.defaultdict(list)
    drop = collections.Counter()
    for i in G:
        for kd in kinds:
            s = _order_status(items[i], model, axis, kd)
            fl = int(new[i][f"vit_{a_}_{kd}"]["pred"] != o[i])
            if s in ("samecol", "ambig"):
                drop[s] += 1
            pairs[i].append((s, fl))
    mir = {i: int(ref[i][f"vit_{a_}"]["pred"] != o[i]) for i in G}
    offr = {i: sum(new[i][f"vit_{off}_{kd}"]["pred"] != o[i] for kd in PM.PERMS) / len(PM.PERMS) for i in G}
    grp = lambda ids, g: [x for i in ids for x in pairs[i] if x[0] == g]
    delta = lambda ids: _rate(grp(ids, "rev")) - _rate(grp(ids, "keep"))
    rev, keep, same = grp(G, "rev"), grp(G, "keep"), grp(G, "samecol")
    res = {"n_items": len(G), "n_rev": len(rev), "n_keep": len(keep), "n_samecol": len(same), "drop": dict(drop),
           "rev": _rate(rev), "keep": _rate(keep), "samecol": _rate(same),
           "D": delta(G), "D_ci": _cluster_boot(pairs, delta) if G else (float("nan"),) * 2,
           "mirror": sum(mir.values()) / len(G) if G else float("nan"),
           "offaxis": sum(offr.values()) / len(G) if G else float("nan")}
    return res


def _seed_table(items, new, ref, m):
    """Geometry-consistent left-right items that the mirror flips: (number of shuf seeds reversing the two objects,
    number flipping the answer) -> count; also the number of items per reversing count and the items skipped (same
    unit or ambiguous)."""
    from experiments import controls_permutation as CP
    G = [i for i in CP.geo_subset(ref, "lr") if i in new and i in items]
    o = {i: ref[i]["orig"]["pred"] for i in G}
    tab, nrev, skip = collections.Counter(), collections.Counter(), 0
    for i in G:
        if ref[i]["vit_w"]["pred"] == o[i]:
            continue
        st = [_order_status(items[i], m, "lr", k) for k in PM.SHUF]
        if any(s in ("samecol", "ambig") for s in st):
            skip += 1
            continue
        r = sum(s == "rev" for s in st)
        fl = sum(new[i][f"vit_w_{k}"]["pred"] != o[i] for k in PM.SHUF)
        tab[(r, fl)] += 1
        nrev[r] += 1
    return tab, nrev, skip


def _whatsup_dest(new, ref):
    """What'sUp, geometry-consistent left-right items: {(condition, gold): (items, changed, to the mirror relation,
    to another relation)} for the mirror and the width permutations."""
    from experiments import controls_permutation as CP
    G = [i for i in CP.geo_subset(ref, "lr") if i in new]
    o = {i: ref[i]["orig"]["pred"] for i in G}
    w = {}
    for kd in ["mirror"] + list(PM.PERMS):
        for gl in ("left", "right", "all"):
            ids = [i for i in G if gl == "all" or ref[i]["orig"]["expected"] == gl]
            get = (lambda i: ref[i]["vit_w"]["pred"]) if kd == "mirror" else (lambda i, kd=kd: new[i][f"vit_w_{kd}"]["pred"])
            ch = [i for i in ids if get(i) != o[i]]
            mr = sum(get(i) == INV.get(o[i]) for i in ch)
            w[(kd, gl)] = (len(ids), len(ch), mr, len(ch) - mr)
    return w


def object_order(a):
    """Same-axis width permutations on the geometry-consistent left-right CV-Bench COCO items, split by whether they
    reverse the order of the two objects' merge units (boxes from object-boxes); flip rates relative to the main-run
    orig; bootstrap over items (all permutations of an item resampled together). Also: for the items the mirror flips, the number of
    shuf seeds that reverse the objects against the number that flip the answer; What'sUp: where changed answers go."""
    from experiments import controls_permutation as CP
    items = {r["idx"]: r for r in read_jsonl(a.boxes)}
    p = pct
    L = [f"== Object order: CV-Bench 2D Relation COCO items, {len(items)} mapped; geometry-consistent subset; "
         f"flips relative to the main-run orig; cluster bootstrap (by item) B={a.B} seed 0"]
    L.append("\n-- left-right items (width permutations vit_w_*)")
    for m in CP.MODELS:
        new, ref = CP._pair(a.new, m, "cv2d")
        for tag, kinds in (("pooled", PM.PERMS), ("shift", ("shift",)), ("shuf×3", PM.SHUF)):
            r = _order_analyze(items, new, ref, m, "lr", kinds)
            L.append(f"  {m:5s} {tag:8s} items {r['n_items']}  reversed n={r['n_rev']} flips {p(r['rev'])}  kept n={r['n_keep']} flips {p(r['keep'])}"
                     f"  Δ {p(r['D'])} [{p(r['D_ci'][0]).strip()}, {p(r['D_ci'][1]).strip()}]"
                     f"  same column n={r['n_samecol']} flips {p(r['samecol'])}  ambiguous {r['drop'].get('ambig', 0)}")
            L.append(f"        reference: mirror {p(r['mirror'])}  off-axis permutations {p(r['offaxis'])}")
    L.append("\n== Seeds that reverse the objects: geometry-consistent left-right items flipped by the mirror; items whose "
             "number of flipping shuf seeds equals the number of shuf seeds that reverse the two objects' column order")
    for m in CP.MODELS:
        new, ref = CP._pair(a.new, m, "cv2d")
        tab, nrev, skip = _seed_table(items, new, ref, m)
        n = sum(nrev.values())
        L.append(f"  {m}: items {n} ({skip} items with the same column or ambiguous not counted)")
        eq = sum(tab[(r, r)] for r in range(4))
        L.append(f"      items with flips = reversals {eq}/{n} = {p(eq / n if n else float('nan'))}%; of the items flipped by "
                 f"3/3 seeds, reversed by 3/3 "
                 f"{tab[(3, 3)]}/{sum(tab[(r, 3)] for r in range(4))}")
    L.append("\n== What'sUp (no boxes): geometry-consistent left-right items, changed answers that go to the mirror relation / "
             "an off-axis relation (width conditions)")
    for m in CP.MODELS:
        new, ref = CP._pair(a.new, m, "whatsup")
        w = _whatsup_dest(new, ref)
        for kd in ["mirror"] + list(PM.PERMS):
            L.append(f"  {m:5s} {kd:6s} " + "  ".join(
                f"{gl}: items {w[(kd, gl)][0]} changed {w[(kd, gl)][1]} -> mirror {w[(kd, gl)][2]} "
                f"({p(w[(kd, gl)][2] / w[(kd, gl)][1] if w[(kd, gl)][1] else float('nan')).strip()}%)"
                f" off-axis {w[(kd, gl)][3]}" for gl in ("left", "right", "all")))
    print("\n".join(L))


# ================================================================ depth-conflicts (post hoc)

def _ci3(t):
    return f"{pct(t[0]).strip()} [{pct(t[1]).strip()}, {pct(t[2]).strip()}]"


def _boot3(stat, ids, B=10000, seed=0):
    ids = sorted(ids)
    rng = random.Random(seed)
    n = len(ids)
    vals = []
    for _ in range(B):
        v = stat([ids[rng.randrange(n)] for _ in range(n)])
        if v == v:
            vals.append(v)
    vals.sort()
    return stat(ids), vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def _heur(gold, group):
    """Answer of the 'higher in the image means farther' heuristic (asked which object is closer)."""
    return gold if group == "consistent" else 1 - gold


def _depth_all(path):
    """{idx: {cond: record}} of the base-model run on all Depth items (records of other sets are skipped)."""
    R = collections.defaultdict(dict)
    for r in read_jsonl(path):
        if r.get("set", "cvb3d") == "cvb3d":
            R[r["idx"]][r["cond"]] = r
    return R


def depth_conflicts(a):
    from experiments import controls_permutation as CP
    base = dict(x.split("=", 1) for x in a.base)
    L = ["== Depth conflict items: CV-Bench 3D Depth, 600 items (base model); orig errors on the conflict (counter) items "
         "corrected (paired bootstrap, by item)"]
    for m in ("7b", "3b"):
        R = _depth_all(base[m])
        grp = {i: v["orig"]["group"] for i, v in R.items()}
        con = [i for i in R if grp[i] == "counter"]
        err = [i for i in con if R[i]["orig"]["pred"] != R[i]["orig"]["gold"]]
        cor = [i for i in con if R[i]["orig"]["pred"] == R[i]["orig"]["gold"]]
        fix = {c: {i: int(R[i][c]["pred"] == R[i][c]["gold"]) for i in err} for c in ("vit_h", "vit_w")}
        brk = {c: {i: int(R[i][c]["pred"] != R[i][c]["gold"]) for i in cor} for c in ("vit_h", "vit_w")}
        L.append(f"  {m}: conflict items {len(con)}, orig accuracy {len(cor)}/{len(con)}; orig wrong on {len(err)} items")
        for c in ("vit_h", "vit_w"):
            k = sum(fix[c].values())
            b = sum(brk[c].values())
            L.append(f"    {c}: corrected {k}/{len(err)} {_ci3(_wilson3(k, len(err)))}; broken (of the {len(cor)} items correct on orig) {b} {_ci3(_wilson3(b, len(cor)))}")
        d = _boot3(lambda ids: sum(fix["vit_h"][i] - fix["vit_w"][i] for i in ids) / len(ids), err)
        L.append(f"    correction rate vit_h − vit_w: {_ci3(d)} points (paired on the same errors)")
        db = _boot3(lambda ids: sum(brk["vit_h"][i] - brk["vit_w"][i] for i in ids) / len(ids), cor)
        L.append(f"    breaking rate vit_h − vit_w: {_ci3(db)} points")
        both = sum(fix["vit_h"][i] and fix["vit_w"][i] for i in err)
        L.append(f"    corrected by both {both}; vit_h only {sum(fix['vit_h'][i] and not fix['vit_w'][i] for i in err)}; "
                 f"vit_w only {sum(fix['vit_w'][i] and not fix['vit_h'][i] for i in err)}")

    L.append("\n== Held-out half, 314 items (permutation controls), heuristic-following / opposing groups; four height "
             "permutations vs four width permutations (averaged per item); paired bootstrap")
    for m in ("q7b", "q3b"):
        R = CP.load(os.path.join(a.new, f"{m}_depth.jsonl"))
        side, ch = {}, {}
        for i, v in R.items():
            o = v["orig"]
            if o["group"] not in ("consistent", "counter"):
                continue
            po = CP.depth_pred(o)
            if po is None:
                continue
            side[i] = "follow" if po == _heur(o["gold"], o["group"]) else "against"
            ch[i] = {}
            for ax in "hw":
                for k in PM.PERMS:
                    pk = CP.depth_pred(v[f"vit_{ax}_{k}"])
                    ch[i][f"{ax}_{k}"] = float("nan") if pk is None else float(pk != po)
        ids = sorted(side)
        nf = sum(side[i] == "follow" for i in ids)
        L.append(f"  {m}: items {len(ids)} (following {nf}, opposing {len(ids) - nf}; orig ties removed)")

        def avg(i, keys):
            xs = [ch[i][k] for k in keys if ch[i][k] == ch[i][k]]
            return sum(xs) / len(xs) if xs else float("nan")

        def gap(keys, norm=False):
            def f(I):
                fo = [avg(i, keys) for i in I if side[i] == "follow"]
                ag = [avg(i, keys) for i in I if side[i] == "against"]
                if not fo or not ag:
                    return float("nan")
                g = sum(fo) / len(fo) - sum(ag) / len(ag)
                if norm:
                    tot = sum(avg(i, keys) for i in I) / len(I)
                    return g / tot if tot else float("nan")
                return g
            return f
        for ax in "hw":
            for k in PM.PERMS:
                key = [f"{ax}_{k}"]
                tot = sum(avg(i, key) for i in ids) / len(ids)
                fo = [avg(i, key) for i in ids if side[i] == "follow"]
                ag = [avg(i, key) for i in ids if side[i] == "against"]
                L.append(f"    vit_{ax}_{k:6s} total change {pct(tot)}  following {pct(sum(fo) / len(fo))}  opposing {pct(sum(ag) / len(ag))}  gap {pct(sum(fo) / len(fo) - sum(ag) / len(ag))}")
        Hk = [f"h_{k}" for k in PM.PERMS]
        Wk = [f"w_{k}" for k in PM.PERMS]
        for tag, keys in (("height permutations (all four)", Hk), ("width permutations (all four)", Wk)):
            L.append(f"    {tag}: total change {pct(sum(avg(i, keys) for i in ids) / len(ids))}  gap {_ci3(_boot3(gap(keys), ids))}"
                     f"  gap_norm {_ci3(_boot3(gap(keys, True), ids))}")
        did = _boot3(lambda I: gap(Hk)(I) - gap(Wk)(I), ids)
        didn = _boot3(lambda I: gap(Hk, True)(I) - gap(Wk, True)(I), ids)
        L.append(f"    DiD = gap_h − gap_w: {_ci3(did)}; DiD_norm: {_ci3(didn)} (ratio, shown x 100)")
        tots = {k: sum(avg(i, [k]) for i in ids) / len(ids) for k in Hk + Wk}
        hk, wk = min(((h, w) for h in Hk for w in Wk), key=lambda t: abs(tots[t[0]] - tots[t[1]]))
        dm = _boot3(lambda I: gap([hk])(I) - gap([wk])(I), ids)
        L.append(f"    strength-matched: {hk} (total change {pct(tots[hk])}) vs {wk} ({pct(tots[wk])}): gap difference {_ci3(dm)}")

    L.append("\n== Accuracy, gold base rate and the model's heuristic-following rate (consistent + counter items only; "
             "following rate − base rate, paired bootstrap by item)")
    srcs = []
    for m in ("7b", "3b"):
        rows = [(r["idx"], r["group"], r["gold"], None if r["lp"][0] == r["lp"][1] else int(r["lp"][1] > r["lp"][0]))
                for r in read_jsonl(DEPTH_TEST[m]) if r["cond"] == "orig" and r["task"] == "Depth"]
        srcs.append((f"{m} held-out half", rows))
        R = _depth_all(base[m])
        srcs.append((f"{m} all 600 items (base model)", [(i, v["orig"]["group"], v["orig"]["gold"], v["orig"]["pred"])
                                               for i, v in R.items() if "orig" in v]))
    for name, rows in srcs:
        rows = [r for r in rows if r[1] in ("consistent", "counter") and r[3] is not None]
        acc = {g: [int(pr == gd) for i, gg, gd, pr in rows if gg == g] for g in ("consistent", "counter")}
        n = len(rows)
        nb = sum(r[1] == "consistent" for r in rows)
        fol = {r[0]: int(r[3] == _heur(r[2], r[1])) for r in rows}
        cons = {r[0]: int(r[1] == "consistent") for r in rows}
        d = _boot3(lambda I: sum(fol[i] - cons[i] for i in I) / len(I), list(fol))
        L.append(f"  {name}: n={n} (consistent {nb}, conflict {n - nb})")
        L.append(f"    accuracy: consistent {_ci3(_wilson3(sum(acc['consistent']), len(acc['consistent'])))}  conflict {_ci3(_wilson3(sum(acc['counter']), len(acc['counter'])))}")
        L.append(f"    gold base rate (share of consistent items) {pct(nb / n)}; model following rate {_ci3(_wilson3(sum(fol.values()), n))}; following rate − base rate {_ci3(d)} points")
    print("\n".join(L))


# ================================================================ depth-shared-images

def depth_shared_images(a):
    """CV-Bench 3D (test_3d.jsonl of nyu-visionx/CV-Bench): the first 600 rows are the Depth questions, item index
    1438 + row. Source images (source_filename) shared by the validation and the held-out half
    (vlmlr.depth.cvbench3d_split); groups from the base-model run on all Depth items."""
    from vlmlr.depth import cvbench3d_split as split_of
    rows = read_jsonl(a.cvbench3d)[:600]
    assert all(r["task"] == "Depth" for r in rows)
    grp = {i: v["orig"]["group"] for i, v in _depth_all(a.base).items() if "orig" in v}
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        by[r["source_filename"]].append(1438 + k)
    shared = sorted(i for ids in by.values() if len({split_of(i) for i in ids}) > 1 for i in ids)
    n_img = sum(1 for ids in by.values() if len({split_of(i) for i in ids}) > 1)
    L = [f"== Shared source images: Depth, 600 items, {len(by)} source images (600 distinct CV-Bench filenames)",
         f"  validation half {sum(split_of(1438 + k) == 'val' for k in range(600))} items, held-out half {sum(split_of(1438 + k) == 'test' for k in range(600))} items",
         f"  source images shared by the two halves {n_img}, covering {len(shared)} items: held-out half {sum(split_of(i) == 'test' for i in shared)}, validation half {sum(split_of(i) == 'val' for i in shared)}"]
    for half in ("test", "val"):
        c = collections.Counter(grp.get(i) for i in shared if split_of(i) == half)
        L.append(f"  {half} half, items with a shared source image by group: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print("\n".join(L))


# ================================================================ vertical-content-cvbench (post hoc)

def vertical_content_cvbench(a):
    """CV-Bench main runs: flip-back rate FB = share of items whose answer under pix_both differs from that under pix
    (all items of the axis), vertical (vflip_both_h) and horizontal (hflip_both_w); D = FB_h - FB_v, unpaired bootstrap."""
    L = ["== CV-Bench, content-only control (post hoc): flip-back rate FB = answer under pix_both differs from the answer "
         "under pix (all items of the axis); D = FB_h − FB_v, unpaired bootstrap B=10000 seed 0"]
    for m in ("q3b", "q7b", "ivl2b", "ivl8b"):
        p = MAIN_CV[m]
        d = _index(_read(p), "variant")
        ud = [i for i, v in d.items() if v["orig"]["axis"] == "ud"]
        lr = [i for i, v in d.items() if v["orig"]["axis"] == "lr"]
        fb = lambda c, pix: (lambda I: sum(d[i][c]["pred"] != d[i][pix]["pred"] for i in I) / len(I))
        fv, fh = fb("vflip_both_h", "vflip"), fb("hflip_both_w", "hflip")
        D, ci = boot2(fh, lr, fv, ud)
        L.append(f"  {m:5s} up-down n={len(ud)} FB_v {pct(fv(ud))} (vit_h alone {pct(fb('vflip_vit_h', 'vflip')(ud))}, "
                 f"llm_h alone {pct(fb('vflip_llm_h', 'vflip')(ud))}); left-right n={len(lr)} FB_h {pct(fh(lr))}"
                 f" (vit_w alone {pct(fb('hflip_vit_w', 'hflip')(lr))}); D {pct(D)} [{pct(ci[0]).strip()}, {pct(ci[1]).strip()}]")
        for ax, pix, both in (("ud", "vflip", "vflip_both_h"), ("lr", "hflip", "hflip_both_w")):
            G = [i for i, v_ in d.items() if v_["orig"]["axis"] == ax and v_["orig"]["pred"] == v_["orig"]["expected"]
                 and v_[pix]["pred"] == v_[pix]["expected"]]
            k, n = sum(d[i][both]["pred"] == d[i]["orig"]["pred"] for i in G), len(G)
            pp, (lo, hi) = wilson(k, n)
            L.append(f"        GC {ax} n={n}: {both} back to the orig answer {k}/{n} {pct(pp)} [{pct(lo).strip()}, {pct(hi).strip()}]")
    print("\n".join(L))


# ================================================================ qwen3-permutations

def _q3_index(recs):
    """Records of a qwen3-permutations run -> (ref, new): ref holds orig / hflip / vit_w with 'expected' (the gold relation of the
    image; on hflip its mirror), new every condition. Only left-right items are run."""
    ref, new = {}, {}
    for r in recs:
        assert r["axis"] == "lr", "the qwen3-permutations runs contain left-right items only"
        r = dict(r, expected=INV[r["gold"]] if r["cond"] == "hflip" else r["gold"])
        new.setdefault(r["idx"], {})[r["cond"]] = r
        if r["cond"] in ("orig", "hflip", "vit_w"):
            ref.setdefault(r["idx"], {})[r["cond"]] = r
    return ref, new


def _q3_units(recs):
    """{(W, H): (width units, height units)} from the patch grid the processor gave (merge 2)."""
    t = {}
    for r in recs:
        k, v = (r["W"], r["H"]), (r["grid"][1] // 2, r["grid"][0] // 2)
        if t.setdefault(k, v) != v:
            raise ValueError(f"two grids for image size {k}: {t[k]} vs {v}")
    return t


def qwen3_permutations(a):
    """Qwen3-VL-4B, left-right items: flip rates (relative to this run's orig, on the items correct on orig and on
    hflip) of the ViT width mirror and of the same-axis permutations, mirror - max(shift, shuf) with a paired
    bootstrap, shuf / mirror ratio; on CV-Bench the split by whether a permutation reverses the order of the two
    objects' merge units; on What'sUp where the changed answers go."""
    from experiments import controls_permutation as CP
    m = "q3vl4b"
    p = pct

    def ci(c):
        return f"[{p(c[0]).strip()}, {p(c[1]).strip()}]"
    L = ["== Qwen3-VL-4B: same-axis non-mirror permutations on the ViT side (2D-RoPE and absolute position embedding "
         "permuted together; pixels and the LLM side unchanged)"]
    for data in ("cv2d", "whatsup"):
        recs = _read(os.path.join(a.dir, f"{m}_{data}.jsonl"))
        ref, new = _q3_index(recs)
        L.append(f"\n== {data}: Qwen3-VL-4B, {len(new)} left-right items ({len(recs)} records)")
        old = {(r["idx"], r["cond"]): r for r in _read(os.path.join(a.ref_dir, f"{m}_{data}.jsonl"))}
        rp = {}
        for r in recs:
            x = old.get((r["idx"], r["cond"])) if r["cond"] in ("orig", "hflip", "vit_w") else None
            if x is None:
                continue
            d = rp.setdefault(r["cond"], {"n": 0, "same": 0, "max_diff": 0.0})
            d["n"] += 1
            d["same"] += r["pred"] == x["pred"]
            d["max_diff"] = max(d["max_diff"], max(abs(u - v) for u, v in zip(r["lp"], x["lp"])))
        L.append("  reproduction (item by item against the mirror run of the same model): " + "; ".join(
            f"{k} same prediction {v['same']}/{v['n']}, max bf16 log-probability difference {v['max_diff']:.2g}" for k, v in rp.items()))
        r = CP.perm_compare(new, ref, "lr", a.B)
        G = [i for i in CP.geo_subset(ref, "lr") if i in new and all(f"vit_w_{k}" in new[i] for k in PM.PERMS)]
        o = {i: ref[i]["orig"]["pred"] for i in G}
        fm = {i: float(ref[i]["vit_w"]["pred"] != o[i]) for i in G}
        fs = {i: CP.mean(new[i][f"vit_w_{k}"]["pred"] != o[i] for k in PM.SHUF) for i in G}
        km, ks = int(sum(fm.values())), sum(new[i]["vit_w_shift"]["pred"] != o[i] for i in G)
        wm, wsh = CP.wilson(km, len(G)), CP.wilson(ks, len(G))

        def ratio(ids):
            mm = CP.mean(fm[i] for i in ids)
            return CP.mean(fs[i] for i in ids) / mm if mm else float("nan")
        rr, rci = CP.boot(ratio, G, a.B) if G else (float("nan"), (float("nan"),) * 2)
        per_seed = {k: CP.mean(new[i][f"vit_w_{k}"]["pred"] != o[i] for i in G) for k in PM.SHUF}
        L.append(f"  geometry-consistent left-right items n={r['n']} (orig and hflip of this run both correct)")
        L.append(f"  flip rates: mirror vit_w {p(r['mirror'])} {ci(wm[1])} ({km}/{len(G)})  half-grid cyclic shift {p(r['shift'])} {ci(wsh[1])}"
                 f" ({ks}/{len(G)})  random merge-unit column permutation (mean of 3 seeds) {p(r['shuf'])}  per seed "
                 + " ".join(f"{k} {p(v).strip()}" for k, v in per_seed.items()))
        L.append(f"  mirror − max(shift, random) = {p(r['D'])} points {ci(r['D_ci'])} (paired bootstrap, B={a.B}, seed 0); "
                 f"without ties {p(r['D_notie'][0])} (n={r['D_notie'][1]})")
        L.append(f"  random permutations / mirror = {rr:.2f} [{rci[0]:.2f}, {rci[1]:.2f}] (bootstrap, B={a.B}, seed 0)")
        L.append("  by gold: " + "  ".join(f"{g}: n={v['n']} mirror {p(v['mirror'])} permutation (max) {p(v['ctrl'])}"
                                       for g, v in r["by_gold"].items()))
        if data == "cv2d":
            L.append("  off-axis (height permutations on the left-right items): " + "  ".join(f"{k} {p(v)}" for k, v in r["offaxis"].items()))
        L.append("  share of changed answers that become the mirror relation: " + "  ".join(
            f"{k} {p(x[0])} (changed {x[1]})" for k, x in r["mirror_rel"].items()))
        if data == "whatsup":
            w = _whatsup_dest(new, ref)
            L.append("  where the changed answers go (by gold; mirror = the mirror relation word, off-axis = on / under):")
            for kd in ["mirror"] + list(PM.PERMS):
                L.append(f"    {kd:6s} " + "  ".join(
                    f"{gl}: items {w[(kd, gl)][0]} changed {w[(kd, gl)][1]} -> mirror {w[(kd, gl)][2]}"
                    f" ({p(w[(kd, gl)][2] / w[(kd, gl)][1] if w[(kd, gl)][1] else float('nan')).strip()}%) off-axis {w[(kd, gl)][3]}"
                    for gl in ("left", "right", "all")))
            continue
        items = {x["idx"]: x for x in read_jsonl(a.boxes)}
        GRID_UNITS[m] = _q3_units(recs)
        miss = [i for i in items if i in new and (items[i]["W"], items[i]["H"]) not in GRID_UNITS[m]]
        assert not miss, f"{len(miss)} located items without a grid record"
        L.append(f"  split by object order (the {len(items)} CV-Bench items with both objects located; cluster bootstrap by item, B={a.B}, seed 0):")
        for tag, kinds in (("pooled", PM.PERMS), ("shift", ("shift",)), ("shuf×3", PM.SHUF)):
            x = _order_analyze(items, new, ref, m, "lr", kinds)
            L.append(f"    {tag:8s} items {x['n_items']}  reversed n={x['n_rev']} flips {p(x['rev'])}  kept n={x['n_keep']} "
                     f"flips {p(x['keep'])}  diff {p(x['D'])} {ci(x['D_ci'])}  same column n={x['n_samecol']} flips {p(x['samecol'])}"
                     f"  ambiguous {x['drop'].get('ambig', 0)}")
            L.append(f"           reference: mirror {p(x['mirror'])}  off-axis permutations {p(x['offaxis'])}")
        tab, nrev, skip = _seed_table(items, new, ref, m)
        n = sum(nrev.values())
        L.append(f"  located items flipped by the mirror {n} ({skip} with the same column or ambiguous not counted)")
        L.append(f"    of the items flipped by 3/3 seeds, reversed by 3/3 {tab[(3, 3)]}/{sum(tab[(k, 3)] for k in range(4))}")
    print("\n".join(L))


# ================================================================ command line

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.further_checks", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="GPU run (see the module docstring)")
    p.add_argument("--exp", choices=sorted({e for e, _ in PLAN}), required=True)
    p.add_argument("--data", choices=["cv2d", "whatsup"], default="cv2d")
    p.add_argument("--model", choices=sorted(REPO), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--parquet", default="datasets/nyu-visionx__CV-Bench/test_2d.parquet")
    p.add_argument("--whatsup", default="datasets/amitakamath2__whatsup_vlms")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--max-pixels-q3", type=int, default=1280 * 32 * 32)
    p = sub.add_parser("run-option-order", help="GPU run: InternVL3-2B main table with the options reversed")
    p.add_argument("--parquet", default="datasets/nyu-visionx__CV-Bench/test_2d.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--order", choices=["swap", "fwd"], default="swap")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("run-depth-all", help="GPU run: base model on all CV-Bench 3D Depth items, orig / vit_h / vit_w")
    p.add_argument("--model", required=True)
    p.add_argument("--items", default="items_val.jsonl,items_test.jsonl")
    p.add_argument("--image-root", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--min-pixels", type=int, default=56 * 56)
    p = sub.add_parser("gpu-summary", help="summary of the GPU runs except qwen3-permutations")
    p.add_argument("--dir", default=RUN_ROOT)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--json")
    p = sub.add_parser("box-reclassification")
    p.add_argument("--m7b", default=BOX_RUNS["7b"])
    p.add_argument("--m3b", default=BOX_RUNS["3b"])
    p = sub.add_parser("object-boxes", help="COCO boxes of the two objects (needs instances_val2017.json)")
    p.add_argument("--ann", default=COCO_ANN)
    p.add_argument("--parquet", default="datasets/nyu-visionx__CV-Bench/test_2d.parquet")
    p.add_argument("--out", default=OBJECT_BOXES)
    p = sub.add_parser("object-order")
    p.add_argument("--boxes", default=OBJECT_BOXES)
    p.add_argument("--new", default=CTRL_DIR)
    p.add_argument("--B", type=int, default=10000)
    p = sub.add_parser("depth-conflicts")
    p.add_argument("--base", nargs=2, default=[f"7b={DEPTH_ALL['7b']}", f"3b={DEPTH_ALL['3b']}"], metavar="SIZE=FILE")
    p.add_argument("--new", default=CTRL_DIR)
    p = sub.add_parser("depth-shared-images")
    p.add_argument("--cvbench3d", default=CVBENCH_3D)
    p.add_argument("--base", default=DEPTH_ALL["7b"])
    sub.add_parser("vertical-content-cvbench")
    p = sub.add_parser("qwen3-permutations", help="Qwen3-VL-4B same-axis permutations")
    p.add_argument("--dir", default=os.path.join(RUN_ROOT, RUN_DIR["qwen3-permutations"]))
    p.add_argument("--ref-dir", default=os.path.join(RUN_ROOT, RUN_DIR["qwen3-mirrors"]), help="qwen3-mirrors runs (reproduction check)")
    p.add_argument("--boxes", default=OBJECT_BOXES)
    p.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    {"run": run, "run-option-order": run_option_order, "run-depth-all": run_depth_all, "gpu-summary": gpu_summary,
     "box-reclassification": box_reclassification, "object-boxes": object_boxes, "object-order": object_order,
     "depth-conflicts": depth_conflicts, "depth-shared-images": depth_shared_images,
     "vertical-content-cvbench": vertical_content_cvbench, "qwen3-permutations": qwen3_permutations}[a.cmd](a)


OUTPUTS = [
    ("outputs/further_checks/gpu_experiments.txt", ["gpu-summary"], True),
    ("outputs/further_checks/object_boxes.txt", ["object-boxes"], False),     # writes the box file object-order reads
    ("outputs/further_checks/box_reclassification.txt", ["box-reclassification"], False),
    ("outputs/further_checks/permutation_order_strata.txt", ["object-order"], False),
    ("outputs/further_checks/depth_conflict_items.txt", ["depth-conflicts"], False),
    ("outputs/further_checks/depth_shared_source_images.txt", ["depth-shared-images"], False),
    ("outputs/further_checks/vertical_content_only_cvbench.txt", ["vertical-content-cvbench"], False),
    ("outputs/further_checks/qwen3_permutations.txt", ["qwen3-permutations"], True),
]

if __name__ == "__main__":
    main()
