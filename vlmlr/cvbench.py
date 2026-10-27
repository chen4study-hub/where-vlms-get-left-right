"""CV-Bench 2D Relation: items, question template, pixel transforms and the counterfactual variants.

650 questions (383 left/right, 267 above/below), all with the fixed template
  Considering the relative positions of {A} and {B} in the image provided,
  where is {A} located with respect to {B}?
"""
import functools
import io
import json
import os
import re

from PIL import Image, ImageOps

TEMPLATE = re.compile(
    r"^Considering the relative positions of (?P<a>.+?) and (?P<b>.+?) in the image provided, "
    r"where is (?P=a) located with respect to (?P=b)\?$")
INVERSE = {"left": "right", "right": "left", "above": "below", "below": "above"}
AXIS = {"left": "lr", "right": "lr", "above": "ud", "below": "ud"}
CHOICES = {"lr": ["left", "right"], "ud": ["above", "below"]}
# the 19 conditions of the main table, in the order of `variants`
CONDITIONS = ("orig", "orig_frame", "swap", "hflip", "vflip",
              "llm_w", "hflip_llm_w", "vit_w", "hflip_vit_w", "both_w", "hflip_both_w",
              "llm_h", "vflip_llm_h", "vit_h", "vflip_vit_h", "both_h", "vflip_both_h", "ids_rot180", "rot180")
IMAGE_FRAME_HINT = (" Answer in terms of the image exactly as it is displayed (image coordinates), "
                    "not the real-world orientation of the scene.")
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def load_items(parquet, limit=0, task="Relation"):
    """Rows of CV-Bench test_2d.parquet (or test_3d.parquet) of one task, in file order."""
    import pyarrow.parquet as pq
    items = [r for r in pq.read_table(parquet).to_pylist() if r["task"] == task]
    return items[:limit] if limit else items


def to_image(cell):
    if isinstance(cell, dict):
        cell = cell.get("bytes")
    return Image.open(io.BytesIO(cell)).convert("RGB")


def transform(img, tf):
    """None, 'hflip', 'vflip' or 'rot180'."""
    if tf is None:
        return img
    if tf == "hflip":
        return ImageOps.mirror(img)
    if tf == "vflip":
        return ImageOps.flip(img)
    if tf == "rot180":
        return img.rotate(180)
    raise ValueError(f"unknown transform {tf}")


def parse(question):
    m = TEMPLATE.match(question.strip())
    if not m:
        raise ValueError(f"question does not match the template: {question[:80]}")
    return m.group("a"), m.group("b")


def build_question(a, b):
    return (f"Considering the relative positions of {a} and {b} in the image provided, "
            f"where is {a} located with respect to {b}?")


def prompt(question, choices, frame_hint=False):
    q = question + (IMAGE_FRAME_HINT if frame_hint else "")
    opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    return f"{q} Select from the following choices.\n{opts}{SUFFIX}"


def gold_relation(item):
    """CV-Bench answers are '(A)' / '(B)'; return the relation word."""
    return item["choices"][ord(item["answer"].strip("()")) - 65]


def variants(item):
    """The 19 conditions run for each question of the main table (CONDITIONS).

    Each variant: variant, question, choices, transform (pixels), llm_flip / vit_flip (position axes mirrored on
    the LLM / ViT side), expected (relation implied by the image geometry; None for position-only variants,
    which are compared with a reference condition instead), frame_hint.
    hflip = content + ViT positions + LLM positions mirrored together; mirroring only some of them shows which
    one the answer follows, and e.g. hflip_both_w mirrors both position sides back so only the content is mirrored.
    """
    a, b = parse(item["question"])
    rel = gold_relation(item)
    q, ch = item["question"], CHOICES[AXIS[rel]]

    def flip_rel(r, axis):
        return INVERSE[r] if AXIS[r] == axis else r
    out = [
        ("orig", q, ch, None, "", "", rel),
        ("orig_frame", q, ch, None, "", "", rel),                       # only the frame hint is added
        ("swap", build_question(b, a), ch, None, "", "", INVERSE[rel]),  # subject and reference swapped
        ("hflip", q, ch, "hflip", "", "", flip_rel(rel, "lr")),
        ("vflip", q, ch, "vflip", "", "", flip_rel(rel, "ud")),
    ]
    for ax, pix in (("w", "hflip"), ("h", "vflip")):
        for side, llm, vit in (("llm", ax, ""), ("vit", "", ax), ("both", ax, ax)):
            out.append((f"{side}_{ax}", q, ch, None, llm, vit, None))          # positions only
            out.append((f"{pix}_{side}_{ax}", q, ch, pix, llm, vit, None))     # pixels flipped, positions back
    out.append(("ids_rot180", q, ch, None, "hw", "hw", None))                  # 180° rotation by positions only
    out.append(("rot180", q, ch, "rot180", "", "", INVERSE[rel]))                # real 180° rotation
    return [{"variant": name, "question": qq, "choices": cc, "transform": tf, "llm_flip": llm, "vit_flip": vit,
             "expected": exp,
             "frame_hint": name == "orig_frame" or name.startswith("rot") or name == "ids_rot180"}
            for name, qq, cc, tf, llm, vit, exp in out]


def main_rows(recs):
    """The records of the main-table conditions (CONDITIONS) in a prediction file."""
    return [r for r in recs if r["variant"] in CONDITIONS]


@functools.lru_cache(maxsize=None)
def _validation_idx():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "cvbench_splits.json")
    return frozenset(json.load(open(path, encoding="utf-8"))["relation_validation"])


def in_validation_split(idx):
    """Validation split of the 650 questions used by the tile and video-input analyses: the 187 item indices listed
    in vlmlr/tables/cvbench_splits.json ('relation_validation'). The other 463 questions were not used by them."""
    return idx in _validation_idx()
