"""What'sUp (Controlled Images A, amitakamath2/whatsup_vlms): items, prompt and relation bookkeeping.

Each item is one photograph with four captions (left of / right of / on / under); gold_index marks the correct
one. Scoring takes the largest of the four option-letter logits.
"""
import json
import os
import re

RELATIONS = ("left", "right", "on", "under")
AXIS = {"left": "lr", "right": "lr", "on": "ud", "under": "ud"}
# relation in image coordinates after a pixel transform
TRANSFORMED = {
    None: {r: r for r in RELATIONS},
    "hflip": {"left": "right", "right": "left", "on": "on", "under": "under"},
    "vflip": {"left": "left", "right": "right", "on": "under", "under": "on"},
}
SUFFIX = "\nAnswer with the option's letter from the given choices directly."

# (name, pixel transform, LLM-side mirrored axes, ViT-side mirrored axes)
VARIANTS = [
    ("orig", None, "", ""),
    ("hflip", "hflip", "", ""),
    ("vflip", "vflip", "", ""),
    ("vit_w", None, "", "w"),
    ("llm_w", None, "w", ""),
    ("vit_h", None, "", "h"),
    ("llm_h", None, "h", ""),
]
CONDITIONS = tuple(v[0] for v in VARIANTS)


def main_rows(recs):
    """The records of the What'sUp conditions (CONDITIONS) in a prediction file."""
    return [r for r in recs if r["variant"] in CONDITIONS]


def caption_relation(cap):
    """'A scarf to the left of a armchair' -> 'left'; raises if no relation is recognised."""
    c = cap.lower()
    if re.search(r"\bto the left of\b|\bleft of\b", c):
        return "left"
    if re.search(r"\bto the right of\b|\bright of\b", c):
        return "right"
    if re.search(r"\bunder\b|\bbelow\b|\bbeneath\b", c):
        return "under"
    if re.search(r"\bon\b|\bon top of\b|\babove\b", c):
        return "on"
    raise ValueError(f"no relation in caption: {cap}")


def filename_relation(fn):
    """'controlled_images/scarf_right_of_armchair.jpeg' -> 'right' (cross-check of gold_index)."""
    base = fn.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    for key, rel in (("_left_of_", "left"), ("_right_of_", "right"), ("_under_", "under"), ("_on_", "on")):
        if key in base:
            return rel
    return None


def item_from_row(row, idx):
    rels = [caption_relation(c) for c in row["caption_options"]]
    if sorted(rels) != sorted(RELATIONS):
        raise ValueError(f"item {idx}: options are not one each of left/right/on/under: {rels}")
    gold = rels[int(row["gold_index"])]
    fr = filename_relation(row["filename"])
    if fr is not None and fr != gold:
        raise ValueError(f"item {idx}: file name says {fr}, gold_index says {gold}")
    return {"idx": idx, "filename": row["filename"], "options": list(row["caption_options"]), "rels": rels,
            "gold": gold, "axis": AXIS[gold]}


def load_items(root, limit=0):
    """root: dataset snapshot directory with test.json (JSON array or JSON lines) and controlled_images/."""
    txt = open(os.path.join(root, "test.json")).read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(x) for x in txt.splitlines() if x.strip()]
    items = [item_from_row(r, i) for i, r in enumerate(rows)]
    for it in items:
        it["path"] = os.path.join(root, it["filename"])
    items = [it for it in items if os.path.exists(it["path"])] or items
    return items[:limit] if limit else items


def prompt(options):
    opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(options))
    return f"Which caption best describes the image? Select from the following choices.\n{opts}{SUFFIX}"


def expected(item, tf):
    """Relation implied by the image after pixel transform tf."""
    return TRANSFORMED[tf][item["gold"]] if tf in TRANSFORMED else None


def predicted_relation(item, logits4):
    k = max(range(4), key=lambda i: logits4[i])
    return item["rels"][k]


def geometry_consistent(by, axis):
    """Items of `axis` answered correctly both on the original and on the matching real flip (hflip / vflip)."""
    ref = "hflip" if axis == "lr" else "vflip"
    return [i for i, d in by.items() if d.get("orig") and d.get(ref) and d["orig"]["axis"] == axis
            and d["orig"]["pred"] == d["orig"]["expected"] and d[ref]["pred"] == d[ref]["expected"]]


def change_rate(by, ids, v, mirror=None):
    """Answers of variant v that differ from orig; with `mirror`, also those that became the mirrored relation.
    Returns (k_changed, k_mirrored, n)."""
    k = km = n = 0
    for i in ids:
        d = by[i]
        if v not in d:
            continue
        n += 1
        o, x = d["orig"]["pred"], d[v]["pred"]
        k += x != o
        if mirror:
            km += x == TRANSFORMED[mirror][o] and x != o
    return k, km, n
