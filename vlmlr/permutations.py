"""Same-axis permutations of the ViT position code and the helpers of the permutation controls (pure code, no model).

Convention (as for the mirror): "the patch in column c receives the position code of column m[c]"; the mirror is
m[c] = n-1-c. The non-mirror permutations move whole merge units (Qwen2.5-VL merges 2x2 patches, InternVL3 a 2x2
pixel shuffle; one unit = 2 patches along an axis) and keep the order inside a unit, so every visual token that
reaches the LLM is still made of the same patches and only the column (row) of its position code changes.

Kinds: '' identity; 'mirror'; 'shift' (cyclic shift by half the units); 'shufN' (units shuffled with seed N);
'const' (every patch gets id 0: with RoPE the relative phase along the axis is then 0, i.e. the axis is removed);
for the absolute embedding of InternViT also 'mean' (embedding averaged along the axis).
"""
import functools
import json
import os

SEEDS = (0, 1, 2)
SHUF = tuple(f"shuf{s}" for s in SEEDS)
PERMS = ("shift",) + SHUF


@functools.lru_cache(maxsize=None)
def _unit_shuffles():
    """Shuffled merge-unit orders used by 'shufN': table[seed][U - 1] is a permutation of range(U), for U = 1..128
    (vlmlr/tables/unit_shuffles.json)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "unit_shuffles.json")
    return json.load(open(path, encoding="utf-8"))


def unit_shuffle(seed, U):
    """Order of the U merge units under 'shuf<seed>'."""
    rows = _unit_shuffles()[str(seed)]
    if not 1 <= U <= len(rows):
        raise ValueError(f"no stored shuffle for an axis of {U} merge units (1..{len(rows)})")
    return list(rows[U - 1])


def axis_map(n, kind, merge=2):
    """Axis of n patches. Returns m: patch c receives the position code of patch m[c]."""
    if not kind:
        return list(range(n))
    if kind == "mirror":
        return [n - 1 - c for c in range(n)]
    if kind == "const":
        return [0] * n
    if n % merge:
        raise ValueError(f"axis length {n} is not a multiple of merge={merge}")
    U = n // merge
    if kind == "shift":
        unit = [(u + U // 2) % U for u in range(U)]
    elif kind.startswith("shuf"):
        unit = unit_shuffle(int(kind[4:]), U)
    else:
        raise ValueError(f"unknown permutation {kind}")
    return [unit[c // merge] * merge + c % merge for c in range(n)]


def unit_map(U, kind):
    """Merge unit u -> unit whose position code it receives (axis_map on 2U patches, 2 patches per unit)."""
    m = axis_map(2 * U, kind)
    return [m[2 * u] // 2 for u in range(U)]


def qwen_vit_pos_ids(grid_thw, merge, h_kind="", w_kind=""):
    """(h, w) ids of the Qwen2.5-VL vision encoder's 2D-RoPE (as rot_pos_emb builds them), each axis re-mapped by
    axis_map. With h_kind = w_kind = '' identical to the original; 'mirror' equals positions.vit_patch_ids."""
    import torch
    pos = []
    for t, h, w in grid_thw.tolist():
        mh = torch.tensor(axis_map(h, h_kind, merge))
        mw = torch.tensor(axis_map(w, w_kind, merge))
        hp = mh[torch.arange(h)].unsqueeze(1).expand(-1, w)
        wp = mw[torch.arange(w)].unsqueeze(0).expand(h, -1)
        hp = hp.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        wp = wp.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        pos.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
    return torch.cat(pos, dim=0)


def abs_pe_remap(pe, h_kind="", w_kind="", merge=2):
    """InternViT absolute position embedding (1, 1+g*g, D): the patch at (r, c) receives the embedding of
    (mh[r], mw[c]); the CLS row stays. Kind 'mean' averages along that axis. Returns a new tensor."""
    import torch
    if not h_kind and not w_kind:
        return pe
    n = pe.shape[1] - 1
    g = int(round(n ** 0.5))
    if g * g != n:
        raise ValueError(f"position embedding is not a square grid: {n}")
    P = pe[:, 1:].reshape(pe.shape[0], g, g, pe.shape[2])
    if h_kind == "mean":
        P = P.mean(1, keepdim=True).expand(-1, g, -1, -1)
    elif h_kind:
        P = P[:, torch.tensor(axis_map(g, h_kind, merge), device=pe.device)]
    if w_kind == "mean":
        P = P.mean(2, keepdim=True).expand(-1, -1, g, -1)
    elif w_kind:
        P = P[:, :, torch.tensor(axis_map(g, w_kind, merge), device=pe.device)]
    return torch.cat([pe[:, :1], P.reshape(pe.shape[0], n, pe.shape[2])], dim=1)


# ---------------------------------------------------------------- InternVL3 default tiling

def tile_grid(height, width, tile=448, min_tiles=1, max_tiles=12):
    """(columns, rows) of InternVL3's default tiling: re-creates get_optimal_tiled_canvas of the GotOcr2 image
    processor (transformers 4.57.6)."""
    ar = [(w, h) for w in range(1, max_tiles + 1) for h in range(1, max_tiles + 1) if min_tiles <= w * h <= max_tiles]
    ar = sorted(ar, key=lambda x: x[0] * x[1])          # as get_all_supported_aspect_ratios (stable sort)
    aspect, area = width / height, width * height
    best, best_diff = (1, 1), float("inf")
    for g in ar:
        d = abs(aspect - g[0] / g[1])
        if d < best_diff:
            best_diff, best = d, g
        elif d == best_diff and area > 0.5 * tile * tile * g[0] * g[1]:
            best = g
    return best


def multitile_perm(in_tile, n_cols, n_rows, thumb, mode):
    """Width permutation on the image span of a tiled image (tiles row-major, thumbnail last). in_tile: the mirror
    permutation inside one tile (length n, e.g. 256). mode 'tile': every tile mirrored inside itself; 'img': the
    whole image mirrored, i.e. token k of tile (r, c) moves to token in_tile[k] of tile (r, C-1-c); the thumbnail
    (a complete small image) is mirrored inside itself. Returns a list of length n * (C*R + thumb)."""
    n = len(in_tile)
    out = []
    for t in range(n_cols * n_rows):
        r, c = divmod(t, n_cols)
        src = r * n_cols + (n_cols - 1 - c) if mode == "img" else t
        out += [src * n + int(in_tile[k]) for k in range(n)]
    if thumb:
        base = n_cols * n_rows * n
        out += [base + int(in_tile[k]) for k in range(n)]
    return out


# ---------------------------------------------------------------- Qwen3-VL

def qwen3_vit_perm(h, w, merge, axes):
    """Order of the patch tokens in the Qwen3-VL vision encoder: merge blocks (bh, bw) row-major, inside a block
    (ih, iw) row-major (transformers 4.57.6, Qwen3VLVisionModel.rot_pos_emb / fast_pos_embed_interpolate).
    Returns perm: token k receives the position encoding of token perm[k] (the patch at (r, c) gets that of
    (r, w-1-c) for 'w', of (h-1-r, c) for 'h')."""
    MH, MW = h // merge, w // merge

    def idx(r, c):
        bh, ih = divmod(r, merge)
        bw, iw = divmod(c, merge)
        return ((bh * MW + bw) * merge + ih) * merge + iw
    perm = [0] * (h * w)
    for r in range(h):
        for c in range(w):
            r2 = h - 1 - r if "h" in axes else r
            c2 = w - 1 - c if "w" in axes else c
            perm[idx(r, c)] = idx(r2, c2)
    return perm


def qwen3_vit_perm_kinds(h, w, merge, h_kind="", w_kind=""):
    """As qwen3_vit_perm, with each axis re-mapped by axis_map: the patch at (r, c) gets the position encoding of
    (mh[r], mw[c]). h_kind = w_kind = 'mirror' gives qwen3_vit_perm(h, w, merge, 'hw')."""
    mh, mw = axis_map(h, h_kind, merge), axis_map(w, w_kind, merge)
    MW = w // merge

    def idx(r, c):
        bh, ih = divmod(r, merge)
        bw, iw = divmod(c, merge)
        return ((bh * MW + bw) * merge + ih) * merge + iw
    perm = [0] * (h * w)
    for r in range(h):
        for c in range(w):
            perm[idx(r, c)] = idx(mh[r], mw[c])
    return perm
