"""Position-id arithmetic for the counterfactuals (pure tensor code, no model needed).

Qwen2.5-VL
  * LLM side: M-RoPE gives every token a (t, h, w) position. `mirror_mrope_positions` mirrors the h and/or w
    component inside each image span (w' = wmin + wmax - w), so the set of ids, the largest id and rope_deltas
    are unchanged.
  * ViT side: the vision encoder's 2D-RoPE uses (h, w) patch ids. `vit_patch_ids` re-creates them exactly as
    `Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb` does, optionally mirrored.
InternVL3
  * ViT side: a learned absolute position embedding; `mirror_abs_pos_embed` mirrors it on the patch grid.
  * LLM side: 1D RoPE; `internvl_llm_positions` permutes the positions inside each 256-token tile.
Token order (Qwen2.5-VL, LLM side): `grid_permutation`, `permute_positions` and `permute_embeds` move token
contents and position ids independently.
"""
import torch

AXIS_COMPONENT = {"h": 1, "w": 2}      # index of the h / w component in M-RoPE position ids (t, h, w)


def image_spans(ids_row, image_token_id):
    """Contiguous runs of image tokens in one row of input_ids, as [(start, end), ...]."""
    idx = (ids_row == image_token_id).nonzero().flatten().tolist()
    spans = []
    if not idx:
        return spans
    start = idx[0]
    for p, q in zip(idx, idx[1:] + [None]):
        if q is None or q != p + 1:
            spans.append((start, p + 1))
            start = q
    return spans


# ---------------------------------------------------------------- Qwen2.5-VL

def mirror_mrope_positions(position_ids, input_ids, image_token_id, axes):
    """Mirror the h and/or w component ('h', 'w' or 'hw') of M-RoPE ids inside every image span.

    position_ids: (3, B, L); input_ids: (B, L). Modified in place and returned.
    """
    for ax in axes:
        comp = AXIS_COMPONENT[ax]
        for i in range(input_ids.shape[0]):
            for s, e in image_spans(input_ids[i], image_token_id):
                v = position_ids[comp, i, s:e]
                position_ids[comp, i, s:e] = v.min() + v.max() - v
    return position_ids


def vit_patch_ids(grid_thw, merge, axes):
    """(h, w) ids of the Qwen2.5-VL vision encoder's 2D-RoPE, mirrored along `axes`. Returns an (N, 2) tensor.

    Same construction as rot_pos_emb: the tokens are re-ordered into merge x merge windows; the mirroring acts on
    the id values, the re-ordering on the token order, so the two commute.
    """
    pos = []
    for t, h, w in grid_thw.tolist():
        hp = torch.arange(h).unsqueeze(1).expand(-1, w)
        wp = torch.arange(w).unsqueeze(0).expand(h, -1)
        if "h" in axes:
            hp = (h - 1) - hp
        if "w" in axes:
            wp = (w - 1) - wp
        hp = hp.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        wp = wp.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        pos.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
    return torch.cat(pos, dim=0)


def merged_grids(grid_thw, merge):
    """image_grid_thw (patch grid) -> the (t, H, W) grid of merged tokens that each image has in the LLM."""
    return [(int(t), int(h) // merge, int(w) // merge) for t, h, w in grid_thw.tolist()]


def grid_permutation(H, W, kind):
    """Slot permutation on a row-major H x W grid: slot j takes token perm[j].

    kind: '' or 'id' (identity), 'w' (reverse each row), 'h' (reverse the row order), 'hw', 'col' (column-major
    traversal), 'rowshift' (rotate rows by H // 2), 'rowshuffleN' (rows shuffled with seed N).
    """
    idx = torch.arange(H * W).view(H, W)
    if kind in ("", "id"):
        return idx.flatten()
    if kind == "col":
        return idx.t().flatten()
    if kind == "rowshift":
        return idx.roll(H // 2, dims=0).flatten()
    if kind.startswith("rowshuffle"):
        g = torch.Generator().manual_seed(int(kind[len("rowshuffle"):] or 0))
        return idx[torch.randperm(H, generator=g)].flatten()
    dims = [d for ax, d in (("h", 0), ("w", 1)) if ax in kind]
    return idx.flip(dims).flatten()


def image_permutations(grid_thw, merge, kind):
    """One grid permutation per (single-frame) image."""
    out = []
    for t, H, W in merged_grids(grid_thw, merge):
        if t != 1:
            raise ValueError("token-order permutations are defined for single-frame images only")
        out.append(grid_permutation(H, W, kind))
    return out


def permute_positions(position_ids, input_ids, image_token_id, perms):
    """position_ids (3, B, L): image span i is re-ordered by perms[i] (slot j gets the position of token perm[j])."""
    pos = position_ids.clone()
    for b in range(input_ids.shape[0]):
        spans = image_spans(input_ids[b], image_token_id)
        if len(spans) != len(perms):
            raise ValueError(f"{len(spans)} image spans but {len(perms)} permutations")
        for (s, e), p in zip(spans, perms):
            if e - s != len(p):
                raise ValueError(f"image span of length {e - s} but grid of {len(p)}")
            pos[:, b, s:e] = position_ids[:, b, s:e][:, p.to(pos.device)]
    return pos


def permute_embeds(embeds, perms):
    """Re-order the rows of each image's visual tokens (the output of get_image_features)."""
    return tuple(e[p.to(e.device)] for e, p in zip(embeds, perms))


# ---------------------------------------------------------------- InternVL3

def mirror_abs_pos_embed(pe, axes):
    """Mirror InternViT's absolute position embedding (1, 1 + g*g, D) on the g x g patch grid; the CLS row stays.

    Patches are row-major, so after mirroring along 'w' the patch at (r, c) receives the embedding of (r, g-1-c).
    Returns a new tensor; the parameter is not modified.
    """
    if not axes:
        return pe
    n = pe.shape[1] - 1
    g = int(round(n ** 0.5))
    if g * g != n:
        raise ValueError(f"position embedding is not a square grid: {n} patches")
    patch = pe[:, 1:].reshape(pe.shape[0], g, g, pe.shape[2])
    dims = [d for ax, d in (("h", 1), ("w", 2)) if ax in axes]
    patch = patch.flip(dims)
    return torch.cat([pe[:, :1], patch.reshape(pe.shape[0], n, pe.shape[2])], dim=1)


def pixel_shuffle_coords(shuffle_fn, g, scale=0.5):
    """Grid cell (row, col) of each merged image token, derived from the model's own pixel_shuffle.

    A g x g feature map whose channels hold (row, col) is passed through shuffle_fn; each output token must
    collect exactly one k x k block, otherwise mirroring is not a permutation. Returns an (N, 2) long tensor in
    the order of the image tokens in the LLM.
    """
    r = torch.arange(g).view(g, 1).expand(g, g)
    c = torch.arange(g).view(1, g).expand(g, g)
    feats = torch.stack([r, c], -1).float().unsqueeze(0)
    out = shuffle_fn(feats, scale_factor=scale)
    G = int(g * scale)
    k = int(round(1 / scale))
    out = out.reshape(G * G, -1)
    rows, cols = out[:, 0::2].long(), out[:, 1::2].long()
    if rows.shape[1] != k * k:
        raise ValueError(f"each merged token should hold {k * k} patches, found {rows.shape[1]}")
    a, b = rows // k, cols // k
    if not (bool((a == a[:, :1]).all()) and bool((b == b[:, :1]).all())):
        raise ValueError("pixel_shuffle output tokens are not complete k x k blocks")
    coords = torch.stack([a[:, 0], b[:, 0]], -1)
    if len({tuple(x) for x in coords.tolist()}) != G * G:
        raise ValueError("duplicate merged-grid coordinates")
    return coords


def mirror_permutation(coords, axes):
    """coords (N, 2): (row, col) of each token. Returns m with m[k] = index of the token at the mirror of k."""
    cs = coords.tolist()
    H = max(x[0] for x in cs) + 1
    W = max(x[1] for x in cs) + 1
    where = {tuple(x): i for i, x in enumerate(cs)}
    out = []
    for r, c in cs:
        if "h" in axes:
            r = H - 1 - r
        if "w" in axes:
            c = W - 1 - c
        out.append(where[(r, c)])
    return torch.tensor(out, dtype=torch.long)


def internvl_llm_positions(input_ids, image_token_id, perm):
    """(B, L) position_ids for InternVL's 1D RoPE: text keeps 0..L-1; inside every tile of len(perm) image tokens,
    token k gets the position of token perm[k]. perm=None returns arange (used by the self-check)."""
    B, L = input_ids.shape
    pos = torch.arange(L, device=input_ids.device).unsqueeze(0).repeat(B, 1)
    if perm is None:
        return pos
    n = len(perm)
    perm = perm.to(input_ids.device)
    for i in range(B):
        for s, e in image_spans(input_ids[i], image_token_id):
            if (e - s) % n:
                raise ValueError(f"image span of length {e - s} is not a multiple of the tile size {n}")
            for t in range(s, e, n):
                pos[i, t:t + n] = t + perm
    return pos
