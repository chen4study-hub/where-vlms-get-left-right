"""Hooks that change where the model thinks each image token is, without touching the pixels.

All hooks wrap methods of the loaded model instance; no transformers source is modified. They are written
against transformers 4.57.6 (Qwen2_5_VLForConditionalGeneration, InternVLForConditionalGeneration).

  QwenPositionMirror     whole-model mirror of the ViT 2D-RoPE ids and/or the LLM M-RoPE ids (Qwen2.5-VL)
  InternVLPositionMirror mirror of InternViT's absolute position embedding and/or the LLM 1D positions
  QwenLayerwiseMirror    mirrored ViT positions in a chosen subset of ViT blocks only (dose curves)
  QwenTokenOrder         LLM side: permute the visual tokens' contents and their position ids independently

Each class keeps the original methods (`orig_*`) so the self-checks can compare against them.
"""
import torch

from . import positions as P


class QwenPositionMirror:
    """Mirror the positions that Qwen2.5-VL assigns to image patches.

    llm: axes of the LLM-side M-RoPE ids to mirror ('', 'w', 'h', 'hw'), via model.model.get_rope_index;
    vit: axes of the vision encoder's 2D-RoPE ids to mirror, via visual.rot_pos_emb.
    Set `llm` / `vit` before a forward pass and reset them to '' afterwards.
    """

    def __init__(self, model):
        self.inner, self.visual = model.model, model.model.visual
        self.image_token_id = model.config.image_token_id
        self.orig_rope_index, self.orig_rot_pos_emb = self.inner.get_rope_index, self.visual.rot_pos_emb
        self.llm = self.vit = ""
        self.inner.get_rope_index = self._rope_index
        self.visual.rot_pos_emb = self._rot_pos_emb

    def _rope_index(self, input_ids=None, *args, **kwargs):
        pos, deltas = self.orig_rope_index(input_ids, *args, **kwargs)
        if self.llm and input_ids is not None:
            pos = P.mirror_mrope_positions(pos.clone(), input_ids, self.image_token_id, self.llm)
        return pos, deltas

    def _rot_pos_emb(self, grid_thw):
        if not self.vit:
            return self.orig_rot_pos_emb(grid_thw)
        return self.vit_rotary(grid_thw, self.vit)

    def vit_rotary(self, grid_thw, axes):
        """Rotary angles of the vision encoder for patch ids mirrored along `axes` (same layout as rot_pos_emb)."""
        ids = P.vit_patch_ids(grid_thw.cpu(), self.visual.spatial_merge_size, axes).to(grid_thw.device)
        return self.visual.rotary_pos_emb(grid_thw[:, 1:].max())[ids].flatten(1)

    def self_check(self, inputs):
        """Unmirrored re-created ViT ids equal transformers' element-wise; the LLM mirror is a permutation inside
        each image span that keeps the largest id."""
        g = inputs["image_grid_thw"]
        assert torch.equal(self.orig_rot_pos_emb(g), self.vit_rotary(g, "")), \
            "re-created ViT position ids differ from the transformers implementation"
        base, _ = self.orig_rope_index(inputs["input_ids"], g, None, None, inputs["attention_mask"])
        flp = P.mirror_mrope_positions(base.clone(), inputs["input_ids"], self.image_token_id, "w")
        assert flp.max() == base.max() and not torch.equal(flp, base), "LLM position mirror self-check failed"

    def forward_logits(self, model, inputs, llm="", vit=""):
        """Next-token logits (fp32) of the last position with the given mirrors switched on."""
        self.llm, self.vit = llm, vit
        model.model.rope_deltas = None
        try:
            return model(**inputs).logits[0, -1].float()
        finally:
            self.llm = self.vit = ""


class InternVLPositionMirror:
    """InternVL3 (-hf): ViT positions enter once, as a learned absolute embedding; the LLM uses 1D RoPE.

    vit: axes to mirror in embeddings.interpolate_pos_encoding (CLS untouched).
    LLM: pass `positions(input_ids, axes)` as position_ids; inside each tile token k takes the position of the
    token at its mirrored grid cell. The merged grid is derived from the model's own pixel_shuffle.
    Experiments use a single 448 x 448 tile (crop_to_patches=False), where a pixel hflip is exactly a mirror of
    the 32 x 32 patch grid.
    """

    def __init__(self, model):
        self.emb = model.model.vision_tower.embeddings
        if self.emb.position_embeddings is None:
            raise RuntimeError("this InternViT has no absolute position embedding")
        self.image_token_id = model.config.image_token_id
        gh, gw = self.emb.patch_embeddings.patch_shape
        assert gh == gw, f"non-square patch grid {gh}x{gw}"
        self.coords = P.pixel_shuffle_coords(model.model.pixel_shuffle, gh, model.config.downsample_ratio)
        self.n_tokens = len(self.coords)
        self.perms = {ax: P.mirror_permutation(self.coords, ax) for ax in ("w", "h", "hw")}
        self.vit, self.calls = "", 0
        self.orig_pos_encoding, self.emb.interpolate_pos_encoding = self.emb.interpolate_pos_encoding, self._pos_encoding

    def _pos_encoding(self, embeddings, height, width):
        self.calls += 1
        pe = self.orig_pos_encoding(embeddings, height, width)
        return P.mirror_abs_pos_embed(pe, self.vit) if self.vit else pe

    def positions(self, input_ids, axes):
        return P.internvl_llm_positions(input_ids, self.image_token_id, self.perms[axes] if axes else None)

    def forward_logits(self, model, inputs, llm="", vit="", pass_positions=True):
        self.vit, self.calls = vit, 0
        try:
            kw = {"position_ids": self.positions(inputs["input_ids"], llm)} if pass_positions else {}
            logits = model(**inputs, **kw).logits[0, -1].float()
        finally:
            self.vit = ""
        assert self.calls == 1, f"the ViT position embedding should be used once per forward, got {self.calls}"
        return logits

    def self_check(self, model, inputs, full=True):
        """Single tile; mirrored LLM positions are a permutation; the embedding mirror is an involution; both
        mirrors change the output. full=True also checks that passing arange equals passing nothing."""
        ids = inputs["input_ids"]
        n_img = int((ids == self.image_token_id).sum())
        assert n_img == self.n_tokens and inputs["pixel_values"].shape[0] == 1, \
            f"not a single tile: {n_img} image tokens, {inputs['pixel_values'].shape[0]} tiles"
        for ax in ("w", "h", "hw"):
            p = self.positions(ids, ax)
            assert sorted(p[0].tolist()) == list(range(ids.shape[1]))
            if full:
                assert not torch.equal(p, self.positions(ids, ""))
        pe = self.orig_pos_encoding(torch.zeros(1, 1 + self.emb.patch_embeddings.num_patches, 1), *self.emb.image_size)
        assert P.mirror_abs_pos_embed(pe, "") is pe and torch.equal(
            P.mirror_abs_pos_embed(P.mirror_abs_pos_embed(pe, "w"), "w"), pe), "position embedding mirror self-check failed"
        if full:
            none = self.forward_logits(model, inputs, pass_positions=False)
            ar, ar2 = self.forward_logits(model, inputs), self.forward_logits(model, inputs)
            tol, d = float((ar - ar2).abs().max()), float((none - ar).abs().max())
            assert d <= tol, f"explicit arange position_ids differ from the default (by {d})"
        o = self.forward_logits(model, inputs)
        assert not torch.equal(self.forward_logits(model, inputs, vit="w"), o) and \
            not torch.equal(self.forward_logits(model, inputs, llm="w"), o), "mirroring did not change the output"
        return n_img


class LayerSwap:
    """Replace the kwarg position_embeddings of selected layers by `alt`.

    If `rotary` is given, its forward is wrapped so that, while armed, it also computes (cos, sin) for alt_pos.
    """

    def __init__(self, rotary, layers):
        self.alt_pos, self.alt, self.sel, self.hits = None, None, frozenset(), 0
        if rotary is not None:
            orig = rotary.forward

            def fwd(x, position_ids, *a, **k):
                out = orig(x, position_ids, *a, **k)
                if self.alt_pos is not None:
                    self.alt = orig(x, self.alt_pos, *a, **k)
                return out
            rotary.forward = fwd
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._hook(i), with_kwargs=True)

    def _hook(self, i):
        def h(_mod, args, kwargs):
            if self.alt is not None and i in self.sel:
                if "position_embeddings" not in kwargs:
                    raise RuntimeError(f"layer {i} received no position_embeddings kwarg")
                kwargs["position_embeddings"] = self.alt
                self.hits += 1
            return args, kwargs
        return h

    def arm(self, alt_pos, sel):
        self.alt_pos, self.sel = alt_pos, frozenset(sel)

    def disarm(self):
        self.alt_pos, self.alt, self.sel = None, None, frozenset()


class QwenLayerwiseMirror:
    """Mirrored ViT positions in a subset of blocks only (Qwen2.5-VL).

    visual.forward builds rot_pos_emb -> window re-order -> (cos, sin); the mirrored copy is built the same way
    from the window index and handed to the selected blocks.
    set(axes, layers): axes 'w' / 'h', layers = indices of the blocks that see mirrored positions; full=True
    replaces the positions of the whole ViT instead (used only by the self-check).
    """

    def __init__(self, model):
        self.vis = model.model.visual
        self.n_vit = len(self.vis.blocks)
        self.vit_swap = LayerSwap(None, self.vis.blocks)
        self._rot0, self.vis.rot_pos_emb = self.vis.rot_pos_emb, self._rot_pos_emb
        self._win0, self.vis.get_window_index = self.vis.get_window_index, self._window_index
        self.reset()

    def set(self, axes="", layers=frozenset(), full=False):
        self.axes, self.layers, self.full = axes, frozenset(layers), full
        self.vit_swap.hits = 0

    def reset(self):
        self.set()
        self.vit_swap.disarm()
        self._vit_raw = None

    def _rot_pos_emb(self, grid_thw):
        out = self._rot0(grid_thw)
        if self.axes:
            ids = P.vit_patch_ids(grid_thw.cpu(), self.vis.spatial_merge_size, self.axes).to(grid_thw.device)
            raw = self.vis.rotary_pos_emb(grid_thw[:, 1:].max())[ids].flatten(1)
            if self.full:
                return raw
            self._vit_raw = raw
        return out

    def _window_index(self, grid_thw):
        wi, cu = self._win0(grid_thw)
        if self._vit_raw is not None:
            raw, u = self._vit_raw, self.vis.spatial_merge_unit
            n = raw.shape[0]
            raw = raw.reshape(n // u, u, -1)[wi, :, :].reshape(n, -1)
            emb = torch.cat((raw, raw), dim=-1)
            self.vit_swap.alt, self.vit_swap.sel = (emb.cos(), emb.sin()), self.layers
            self._vit_raw = None
        return wi, cu

    def forward_logits(self, model, inputs, **cfg):
        """Returns (fp32 last-position logits, number of ViT blocks that received the mirrored positions)."""
        self.set(**cfg)
        model.model.rope_deltas = None
        try:
            logits = model(**inputs).logits[0, -1].float()
            hits = self.vit_swap.hits
        finally:
            self.reset()
        return logits, hits

    def self_check(self, model, inputs, log=print):
        """L = 0 equals no hook; L = all blocks equals the whole-ViT replacement; the number of replaced blocks
        equals the number selected."""
        base, _ = self.forward_logits(model, inputs)
        base2, _ = self.forward_logits(model, inputs)
        tol = float((base - base2).abs().max())
        log(f"[self-check] max difference between two identical forward passes {tol:.2e}")
        n = self.n_vit
        z, h0 = self.forward_logits(model, inputs, axes="w", layers=frozenset())
        assert float((z - base).abs().max()) <= tol and h0 == 0, "L=0 differs from the original"
        all_l, h = self.forward_logits(model, inputs, axes="w", layers=frozenset(range(n)))
        full, _ = self.forward_logits(model, inputs, axes="w", full=True)
        d = float((all_l - full).abs().max())
        assert d <= tol, f"L=all differs from the whole-ViT replacement (by {d})"
        assert h == n, f"{h} blocks replaced, expected {n}"
        assert not torch.equal(all_l, base), "mirroring changed nothing"
        log(f"[self-check] L=0 unchanged, L={n} differs from the whole-ViT replacement by {d:.2e}, {h} blocks replaced")


class QwenTokenOrder:
    """LLM side of Qwen2.5-VL: permute the visual tokens' contents (sigma) and position ids (tau) separately.

    get_image_features is wrapped to re-order each image's merged tokens (after the merger, before they are
    scattered into inputs_embeds); get_rope_index is wrapped to re-order the (t, h, w) ids of each image span.
    rope_deltas and the attention mask are unchanged. Kinds are those of positions.grid_permutation.
    """

    def __init__(self, model):
        self.inner = model.model
        self.image_token_id = model.config.image_token_id
        self.merge = self.inner.visual.spatial_merge_size
        self.emb_kind = self.pos_kind = ""
        self.hits = [0, 0]
        self.orig_image_features, self.inner.get_image_features = self.inner.get_image_features, self._image_features
        self.orig_rope_index, self.inner.get_rope_index = self.inner.get_rope_index, self._rope_index

    def set(self, emb="", pos=""):
        self.emb_kind, self.pos_kind, self.hits = emb, pos, [0, 0]

    def _image_features(self, pixel_values, image_grid_thw=None):
        out = self.orig_image_features(pixel_values, image_grid_thw)
        if self.emb_kind:
            out = P.permute_embeds(out, P.image_permutations(image_grid_thw, self.merge, self.emb_kind))
            self.hits[0] += 1
        return out

    def _rope_index(self, input_ids=None, image_grid_thw=None, *a, **k):
        pos, deltas = self.orig_rope_index(input_ids, image_grid_thw, *a, **k)
        if self.pos_kind and input_ids is not None:
            pos = P.permute_positions(pos, input_ids, self.image_token_id,
                                      P.image_permutations(image_grid_thw, self.merge, self.pos_kind))
            self.hits[1] += 1
        return pos, deltas

    def forward_logits(self, model, inputs, emb="", pos=""):
        """Returns (fp32 last-position logits, [content hook calls, position hook calls], rope_deltas)."""
        self.set(emb, pos)
        model.model.rope_deltas = None
        try:
            logits = model(**inputs).logits[0, -1].float()
            hits, deltas = list(self.hits), model.model.rope_deltas
        finally:
            self.set()
        return logits, hits, deltas

    def self_check(self, model, inputs):
        """Identity permutations through the full hook path equal no hook; id-only 'w' equals the M-RoPE mirror;
        rope_deltas unchanged; order-only 'w' changes the output."""
        base, _, d0 = self.forward_logits(model, inputs)
        base2, _, _ = self.forward_logits(model, inputs)
        tol = float((base - base2).abs().max())
        ident, h, d1 = self.forward_logits(model, inputs, "id", "id")
        assert h == [1, 1] and float((ident - base).abs().max()) <= tol, "identity permutation differs from the original"
        assert torch.equal(d0, d1), "rope_deltas changed"
        ids, g = inputs["input_ids"], inputs["image_grid_thw"]
        p0, _ = self.orig_rope_index(ids, g, None, None, attention_mask=inputs["attention_mask"])
        mine = P.permute_positions(p0, ids, self.image_token_id, P.image_permutations(g, self.merge, "w"))
        assert torch.equal(mine, P.mirror_mrope_positions(p0.clone(), ids, self.image_token_id, "w")), \
            "id-only 'w' differs from the M-RoPE mirror"
        assert mine.max() == p0.max()
        oo, h, d2 = self.forward_logits(model, inputs, "w", "w")
        assert h == [1, 1] and torch.equal(d0, d2) and not torch.equal(oo, base), "order-only 'w' changed nothing"
        return tol
