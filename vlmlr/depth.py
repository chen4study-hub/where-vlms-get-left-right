"""Depth questions and the 'higher means farther' heuristic: CV-Bench 3D and EmbSpatial-Bench.

CV-Bench 3D (test_3d.parquet)
  Depth:    "Which object is closer to the camera ..., the X (red box) or the Y (blue box)?", bbox = [X, Y]
  Boxes are pixel xyxy; choices = [X, Y]; answer '(A)' / '(B)'. Items are split by index into a validation half
  and a held-out test half (`cvbench3d_split`).
EmbSpatial-Bench (embspatial_bench.json, images base64-embedded): near-far questions with four object options,
  boxes [x, y, w, h]; the heuristic answer is the object whose box bottom is lowest (close) or highest (far).
A question is heuristic-consistent if the heuristic gives the gold answer and a conflict ('counter') item if
not; differences below 5 % of the image height are 'ambiguous'.
"""
import ast
import functools
import json
import os

from .stats import wilson

AMBIGUOUS = 0.05                 # vertical difference below 5 % of the image height
CONDITIONS = ["orig", "vit_h", "vit_w", "llm_h"]
# condition -> (ViT axes mirrored, LLM axes mirrored, pixels flipped upside down)
CONDITION_SPEC = {"orig": ("", "", False), "vit_h": ("h", "", False), "vit_w": ("w", "", False),
                  "llm_h": ("", "h", False), "vflip": ("", "", True)}

EMBSPATIAL_REPO = "Phineas476/EmbSpatial-Bench"
EMBSPATIAL_REVISION = "e2733cacf0f20371518c687176e881d79644e61e"
EMBSPATIAL_FILE = "embspatial_bench.json"


# ---------------------------------------------------------------- CV-Bench 3D

DEPTH_IDX = range(1438, 2038)    # item indices of the 600 Depth questions of CV-Bench 3D


@functools.lru_cache(maxsize=None)
def _validation_idx():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "cvbench_splits.json")
    return frozenset(json.load(open(path, encoding="utf-8"))["depth_validation"])


def cvbench3d_split(idx):
    """'val' or 'test': fixed split of the Depth questions; the 286 validation items are listed in
    vlmlr/tables/cvbench_splits.json ('depth_validation'), the other 314 form the held-out test half."""
    if idx not in DEPTH_IDX:
        raise ValueError(f"{idx} is not the index of a CV-Bench 3D Depth question")
    return "val" if idx in _validation_idx() else "test"


def gold_index(item):
    return ord(item["answer"].strip().strip("()")) - 65


def _center_y(box):
    return (box[1] + box[3]) / 2


def depth_group(item, H):
    """consistent: the farther object is higher in the image; counter: it is lower; ambiguous: < 5 % of H."""
    g = gold_index(item)                              # the closer object
    near, far = item["bbox"][g], item["bbox"][1 - g]
    d = _center_y(near) - _center_y(far)
    if abs(d) < AMBIGUOUS * H:
        return "ambiguous"
    return "consistent" if d > 0 else "counter"


def higher_index(item):
    """Index of the object whose box centre is higher in the image (smaller y)."""
    a, b = item["bbox"][:2]
    return 0 if _center_y(a) < _center_y(b) else 1


# ---------------------------------------------------------------- predictions

def predict(lp, ties="separate"):
    """Two-option prediction. ties='separate': an exact tie gives None (counted apart); ties='first': it goes to
    the first option (the rule used for the validation half)."""
    if ties == "separate" and lp[0] == lp[1]:
        return None
    return int(lp[1] > lp[0])


def average_orders(lp_fwd, lp_rev):
    """Option scores averaged over the original and the reversed option order (indexed in the original order)."""
    n = len(lp_fwd)
    return [(lp_fwd[i] + lp_rev[n - 1 - i]) / 2 for i in range(n)]


def argmax_or_none(xs):
    m = max(xs)
    return None if sum(x == m for x in xs) > 1 else xs.index(m)


def score(p, g):
    """1 correct, 0 wrong, 0.5 for a tie (the expected score of a random tie-break)."""
    return 0.5 if p is None else float(p == g)


def accuracy(preds, gold, ids):
    ids = [i for i in ids if i in preds]
    return (100 * sum(score(preds[i], gold[i]) for i in ids) / len(ids)) if ids else float("nan"), len(ids)


def count_ties(preds, ids=None):
    return sum(preds[i] is None for i in (ids if ids is not None else preds) if i in preds)


def change_rate(base, other):
    """Share of items whose prediction differs between two conditions ({idx: pred}); ties are left out.
    Returns (%, n, (lo %, hi %))."""
    ids = [i for i in base if i in other and base[i] is not None and other[i] is not None]
    k = sum(base[i] != other[i] for i in ids)
    return (100 * k / len(ids) if ids else float("nan")), len(ids), tuple(100 * x for x in wilson(k, len(ids)))


def direction_share(base, other, higher, ids):
    """Among items whose prediction changed (ties left out), the share where `other` names the object that is
    higher in the image as the closer one: the direction predicted if the model reads 'encoded higher -> farther'
    and the height positions are mirrored. Returns (%, n changed, (lo %, hi %))."""
    ch = [i for i in ids if i in base and i in other and base[i] is not None and other[i] is not None and base[i] != other[i]]
    k = sum(other[i] == higher[i] for i in ch)
    return (100 * k / len(ch) if ch else float("nan")), len(ch), tuple(100 * x for x in wilson(k, len(ch)))


def fix_rates(by, conds=("vit_h", "vit_w", "llm_h")):
    """by: {idx: {cond: {"pred", "gold"}}}. Among orig errors (ties excluded), the share answered correctly under
    each condition; 'harm': share of orig-correct items answered wrongly. Returns {"n_err", "n_ok",
    cond: {"fix": (%, n, (lo, hi)), "harm": (%, n)}}."""
    err = [d for d in by.values() if "orig" in d and d["orig"]["pred"] is not None and d["orig"]["pred"] != d["orig"]["gold"]]
    ok = [d for d in by.values() if "orig" in d and d["orig"]["pred"] is not None and d["orig"]["pred"] == d["orig"]["gold"]]
    out = {"n_err": len(err), "n_ok": len(ok)}
    for c in conds:
        e = [d for d in err if c in d and d[c]["pred"] is not None]
        k = sum(d[c]["pred"] == d[c]["gold"] for d in e)
        o = [d for d in ok if c in d and d[c]["pred"] is not None]
        h = sum(d[c]["pred"] != d[c]["gold"] for d in o)
        out[c] = {"fix": (100 * k / len(e) if e else float("nan"), len(e), tuple(100 * x for x in wilson(k, len(e)))),
                  "harm": (100 * h / len(o) if o else float("nan"), len(o))}
    return out


# ---------------------------------------------------------------- EmbSpatial-Bench

def as_list(v):
    """EmbSpatial stores some lists as their Python repr."""
    return v if isinstance(v, list) else ast.literal_eval(v)


def embspatial_bottoms(item):
    """Box bottom y + h of each option's object; None if an option name has no box (such items are dropped)."""
    bot = {o["name"]: o["bbox"][1] + o["bbox"][3] for o in as_list(item["objects"])}
    try:
        return [bot[n] for n in as_list(item["answer_options"])]
    except KeyError:
        return None


def embspatial_heuristic(relation, bottoms):
    """'Higher means farther': close -> the object with the lowest box bottom (largest y + h), far -> the highest."""
    if relation == "close":
        return max(range(len(bottoms)), key=lambda k: bottoms[k])
    return min(range(len(bottoms)), key=lambda k: bottoms[k])


def embspatial_anti(relation, bottoms):
    """The opposite end: the option a model following the heuristic picks once the height positions are mirrored."""
    return embspatial_heuristic("far" if relation == "close" else "close", bottoms)


def embspatial_group(item):
    """consistent / counter, or None if an option has no box."""
    b = embspatial_bottoms(item)
    if b is None:
        return None
    return "consistent" if embspatial_heuristic(item["relation"], b) == int(item["answer"]) else "counter"


def embspatial_debias(by, conds=("vit_h", "vit_w", "llm_h")):
    """Change rates split by whether the orig answer follows the heuristic ('follow') or not ('against'), ties dropped.
    Returns {cond: {"follow": (k, n, %, ci), "against": (...)}}."""
    out = {}
    for c in conds:
        k = {"follow": [0, 0], "against": [0, 0]}
        for d in by.values():
            o = d.get("orig")
            if not o or c not in d or o["pred"] is None or d[c]["pred"] is None or o.get("heuristic") is None:
                continue
            side = "follow" if o["pred"] == o["heuristic"] else "against"
            k[side][1] += 1
            k[side][0] += d[c]["pred"] != o["pred"]
        r = {}
        for s, (ch, n) in k.items():
            r[s] = (ch, n, 100 * ch / n if n else float("nan"), tuple(100 * x for x in wilson(ch, n)))
        out[c] = r
    return out


def iter_json_array(path, chunk=1 << 20):
    """Stream the elements of a file whose top level is a JSON array (a few MB of memory for any file size)."""
    dec = json.JSONDecoder()
    ws = " \t\r\n,"
    buf, pos, started, eof = "", 0, False, False
    with open(path, encoding="utf-8") as f:
        while True:
            if not eof:
                data = f.read(chunk)
                eof = not data
                buf = buf[pos:] + data
                pos = 0
            if not started:
                while pos < len(buf) and buf[pos] in ws:
                    pos += 1
                if pos >= len(buf):
                    if eof:
                        return
                    continue
                assert buf[pos] == "[", "top level is not a JSON array"
                pos, started = pos + 1, True
            while True:
                while pos < len(buf) and buf[pos] in ws:
                    pos += 1
                if pos >= len(buf) or buf[pos] == "]":
                    break
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except ValueError:
                    break
                yield obj
                pos = end
            if eof or (pos < len(buf) and buf[pos] == "]"):
                return
