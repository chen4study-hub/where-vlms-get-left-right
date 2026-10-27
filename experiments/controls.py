"""Two control experiments on Qwen2.5-VL-3B / 7B (paper Section 4.7, Fig. 6 and Appendix C).

Swapped option order. On CV-Bench, left and above are always option A, and a bf16 tie goes to the first option.
The main table is re-run with the options of every one of the 19 conditions in reverse order (choices[::-1]);
everything else is as in experiments.main_table. The original order is the main-table run itself. Six key
quantities are compared between the two orders (definitions as in `main_table compare`):
  agree_vit_w_hflip_lr  all left-right items: vit_w gives the same answer as the real horizontal flip
  flip_vit_w_lr         geometry-consistent left-right items: vit_w changes the answer of orig
  flip_llm_w_lr         the same for llm_w
  flip_vit_h_ud         geometry-consistent up-down items (reference vflip): vit_h changes the answer
  flip_llm_h_ud         the same for llm_h
  flip_vit_w_ud         off-axis control: vit_w on up-down items
The geometry-consistent subset (orig and the real flip both correct) is set separately in each order.
Differences use a joint bootstrap over the 650 items; quantities that move by more than 5 points are listed, and
the paper reports their mean over both orders.

Non-spatial recognition (CV-Bench Count, 788 questions with 4-6 options). Conditions orig, hflip, vit_w, vit_h and
llm_w; the answer is the option letter with the highest logit (first one on ties).
drop = accuracy(orig) - accuracy(condition), paired bootstrap; hflip is the reference (a real flip should not
change a count).

  python -m experiments.controls run-order --model Qwen/Qwen2.5-VL-3B-Instruct --parquet data/cvbench/test_2d.parquet \\
      --order swap --out results/controls/swap_3b.jsonl      # --order fwd: the original order (reproduction check)
  python -m experiments.controls run-count --model Qwen/Qwen2.5-VL-3B-Instruct --parquet data/cvbench/test_2d.parquet \\
      --out results/controls/count_3b.jsonl
  python -m experiments.controls summary --orig 3b=<main table 3B> 7b=<main table 7B> --swap 3b=... 7b=... \\
      --count 3b=... 7b=... [--repro 3b=... 7b=...] [--json summary.json]
  python -m experiments.controls cross-check --orig ... --swap ... --count ...   # independent re-computation, two-order means
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

from vlmlr import cvbench as C
from vlmlr.records import ResumableWriter, by_item, read_jsonl
from vlmlr.stats import percentile_ci, resample, resamples

# (name, condition, reference condition, axis, kind)
KEY_QUANTITIES = [
    ("agree_vit_w_hflip_lr", "vit_w", "hflip", "lr", "agree"),
    ("flip_vit_w_lr", "vit_w", "hflip", "lr", "flip"),
    ("flip_llm_w_lr", "llm_w", "hflip", "lr", "flip"),
    ("flip_vit_h_ud", "vit_h", "vflip", "ud", "flip"),
    ("flip_llm_h_ud", "llm_h", "vflip", "ud", "flip"),
    ("flip_vit_w_ud", "vit_w", "hflip", "ud", "flip"),
]
MOVE_POINTS = 5.0                                      # option order: quantities moving by more than this are listed
# Count conditions: (pixel transform, ViT mirror axis, LLM mirror axis)
COUNT_CONDS = {"orig": (None, "", ""), "hflip": ("hflip", "", ""), "vit_w": (None, "w", ""), "vit_h": (None, "h", ""),
               "llm_w": (None, "", "w")}


# ---------------------------------------------------------------- GPU runs

def _load_model(args):
    from vlmlr import models
    from vlmlr.hooks import QwenPositionMirror
    model, proc = models.load_qwen(args.model, max_pixels=args.max_pixels)
    return model, proc, QwenPositionMirror(model)


def run_order(args):
    """The 19 main-table conditions with the options in reverse order (--order swap) or as in the main table (fwd)."""
    import torch
    from vlmlr import models
    model, proc, mirror = _load_model(args)
    ids = models.letter_ids(proc.tokenizer, "AB")
    items = C.load_items(args.parquet, args.limit)
    checked, t0 = False, time.time()
    with ResumableWriter(args.out) as out, torch.inference_mode():
        for it in items:
            img0 = C.to_image(it["image"])
            for v in C.variants(it):
                if (it["idx"], v["variant"]) in out:
                    continue
                choices = v["choices"][::-1] if args.order == "swap" else v["choices"]
                text = C.prompt(v["question"], choices, v["frame_hint"])
                inputs = models.build_inputs(proc, C.transform(img0, v["transform"]), text)
                if not checked:
                    mirror.self_check(inputs)
                    checked = True
                lp = models.option_logprobs(mirror.forward_logits(model, inputs, v["llm_flip"], v["vit_flip"]), ids)
                gold = C.gold_relation(it)
                out.write({"idx": it["idx"], "variant": v["variant"], "order": args.order, "gold_orig": gold,
                           "expected": v["expected"], "pred": models.pick(choices, lp), "logp": lp, "choices": choices,
                           "axis": C.AXIS[gold], "source": it["source"],
                           "n_img_tokens": int((inputs["input_ids"] == model.config.image_token_id).sum())})
            out.flush()
    print(f"[order {args.order}] done {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


def run_count(args):
    """CV-Bench 2D Count under the conditions of COUNT_CONDS."""
    import torch
    from vlmlr import models
    model, proc, mirror = _load_model(args)
    letters = models.letter_ids(proc.tokenizer, "ABCDEF")
    items = C.load_items(args.parquet, args.limit, task="Count")
    checked, t0 = False, time.time()
    with ResumableWriter(args.out, key=("idx", "cond")) as out, torch.inference_mode():
        for it in items:
            img0 = C.to_image(it["image"])
            k = len(it["choices"])
            gold = ord(it["answer"].strip("()")) - 65
            text = C.prompt(it["question"], it["choices"])
            for c, (tf, vit, llm) in COUNT_CONDS.items():
                if (it["idx"], c) in out:
                    continue
                inputs = models.build_inputs(proc, C.transform(img0, tf), text)
                if not checked:
                    mirror.self_check(inputs)
                    checked = True
                lp = models.option_logprobs(mirror.forward_logits(model, inputs, llm, vit), letters[:k])
                out.write({"idx": it["idx"], "cond": c, "gold": gold, "pred": letter_pred(lp), "lp": lp,
                           "n_choices": k, "source": it["source"]})
            out.flush()
    print(f"[count] done {out.n} forwards in {time.time() - t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------- swapped option order

def quantity(R, var, ref, axis, kind, ids=None):
    """R = by_item(records). ids: the items to use (with repeats, for the bootstrap). Returns (%, n)."""
    ids = [i for i in (ids if ids is not None else R) if R[i]["orig"]["axis"] == axis]
    if kind == "agree":
        sub = ids
        k = sum(R[i][var]["pred"] == R[i][ref]["pred"] for i in sub)
    else:
        sub = [i for i in ids if R[i]["orig"]["pred"] == R[i]["orig"]["expected"] and R[i][ref]["pred"] == R[i][ref]["expected"]]
        k = sum(R[i][var]["pred"] != R[i]["orig"]["pred"] for i in sub)
    return (100 * k / len(sub) if sub else float("nan")), len(sub)


def all_quantities(R, ids=None):
    return {name: quantity(R, v, ref, ax, kind, ids) for name, v, ref, ax, kind in KEY_QUANTITIES}


def order_compare(orig_recs, swap_recs, B=10000, seed=0):
    """Key quantities in both orders and their difference (swapped - original), with a joint bootstrap over the
    items (the geometry-consistent subsets are set again in every resample)."""
    Ro, Rs = by_item(orig_recs), by_item(swap_recs)
    ids = sorted(set(Ro) & set(Rs))
    qo, qs = all_quantities(Ro, ids), all_quantities(Rs, ids)
    rng = random.Random(seed)
    diffs = {name: [] for name, *_ in KEY_QUANTITIES}
    for _ in range(B):
        bs = [ids[j] for j in resample(rng, len(ids))]
        a, b = all_quantities(Ro, bs), all_quantities(Rs, bs)
        for name in diffs:
            d = b[name][0] - a[name][0]
            if d == d:
                diffs[name].append(d)
    out = {}
    for name, *_ in KEY_QUANTITIES:
        ds = sorted(diffs[name])
        out[name] = {"orig": qo[name], "swap": qs[name], "diff": qs[name][0] - qo[name][0],
                     "ci": percentile_ci(ds) if ds else (float("nan"), float("nan"))}
    return out


def moved_more_than_5_points(cmp_by_model):
    """Quantities whose value moves by more than MOVE_POINTS points when the options are swapped ('model:quantity')."""
    return [f"{m}:{name}" for m, c in cmp_by_model.items() for name, v in c.items() if not abs(v["diff"]) <= MOVE_POINTS]


def split_by_orig_answer(recs):
    """Post hoc: vit_w flip rate on geometry-consistent left-right items, split by the original answer (word
    left / right) and by its option position (letter A / B)."""
    R = by_item(recs)
    sub = [d for d in R.values() if d["orig"]["axis"] == "lr" and d["orig"]["pred"] == d["orig"]["expected"]
           and d["hflip"]["pred"] == d["hflip"]["expected"]]
    out = {}
    for key, f in (("word", lambda d: d["orig"]["pred"]), ("letter", lambda d: "AB"[d["orig"]["choices"].index(d["orig"]["pred"])])):
        for g in sorted({f(d) for d in sub}):
            xs = [d for d in sub if f(d) == g]
            out[f"{key}={g}"] = (100 * sum(d["vit_w"]["pred"] != d["orig"]["pred"] for d in xs) / len(xs), len(xs))
    return out


def ties(recs):
    return sum(1 for r in recs if r["logp"][0] == r["logp"][1])


# ---------------------------------------------------------------- Count

def letter_pred(lp):
    """Index of the highest option score; the first one on ties."""
    return lp.index(max(lp))


def count_compare(recs, B=10000, seed=0):
    """Accuracy per condition, drop = acc(orig) - acc(condition) with a paired bootstrap interval, and the rate of
    changed answers (pred differs from orig's pred)."""
    by = by_item(recs, cond="cond")
    ids = sorted(i for i, d in by.items() if all(c in d for c in COUNT_CONDS if c != "llm_w") and "orig" in d)
    conds = [c for c in COUNT_CONDS if all(c in by[i] for i in ids)]
    ok = {c: [float(by[i][c]["pred"] == by[i][c]["gold"]) for i in ids] for c in conds}
    n = len(ids)
    acc = {c: 100 * sum(ok[c]) / n for c in conds}
    samples = resamples(n, B, seed)
    out = {"n": n, "acc": acc, "drop": {}, "change": {}}
    for c in conds:
        if c == "orig":
            continue
        d = [ok["orig"][j] - ok[c][j] for j in range(n)]
        bs = sorted(100 * sum(d[j] for j in s) / n for s in samples)
        out["drop"][c] = (100 * sum(d) / n, percentile_ci(bs))
        out["change"][c] = 100 * sum(by[i][c]["pred"] != by[i]["orig"]["pred"] for i in ids) / n
    return out


# ---------------------------------------------------------------- analyses

def _read(p):
    if not (p and os.path.exists(p)):
        raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
    return read_jsonl(p)


def _read_main(p):
    return C.main_rows(_read(p))


def summary(a):
    """Both experiments with their readings; --repro compares a re-run in the original order record by
    record with the main table."""
    orig, swap, count, repro = a.orig, a.swap, a.count, a.repro
    L, res = [], {}
    for m, p in repro.items():
        new = {(r["idx"], r["variant"]): r for r in _read_main(p)}
        old = {(r["idx"], r["variant"]): r for r in _read_main(orig.get(m))}
        ks = [k for k in new if k in old]
        if ks:
            same = sum(new[k]["pred"] == old[k]["pred"] for k in ks)
            dmax = max(max(abs(x - y) for x, y in zip(new[k]["logp"], old[k]["logp"])) for k in ks)
            L.append(f"[reproduction] {m}: re-run in the original order, {len(ks)} rows, same prediction as the main table {same}/{len(ks)}, max log-probability difference {dmax:.4f}")
            res.setdefault("repro", {})[m] = {"n": len(ks), "same": same, "max_lp_diff": dmax}
    cmps = {}
    for m in sorted(set(orig) & set(swap)):
        o, s = _read_main(orig[m]), _read_main(swap[m])
        c = order_compare(o, s, B=a.B)
        cmps[m] = c
        L.append(f"\n== Swapped option order: {m} (original order = main table; ties original {ties(o)} / swapped {ties(s)})")
        for name, v in c.items():
            L.append(f"  {name:22s} original {v['orig'][0]:5.1f} (n={v['orig'][1]})  swapped {v['swap'][0]:5.1f} (n={v['swap'][1]})  "
                     f"diff {v['diff']:+5.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
        for tag, recs in (("original order", o), ("swapped", s)):
            sp = split_by_orig_answer(recs)
            L.append(f"  split by the original answer ({tag}, geometry-consistent left-right items, vit_w flips): " + "; ".join(f"{k} {v[0]:.1f}% (n={v[1]})" for k, v in sp.items()))
        res.setdefault("order", {})[m] = {k: {**v, "ci": list(v["ci"])} for k, v in c.items()}
    if cmps:
        over = moved_more_than_5_points(cmps)
        L.append("\nquantities that move by more than 5 points with the options swapped: " + ("none" if not over else ", ".join(over)))
        res["moved_more_than_5_points"] = over
    for m, p in sorted(count.items()):
        recs = _read(p)
        cc = count_compare(recs, B=a.B)
        L.append(f"\n== CV-Bench Count: {m} (n = {cc['n']})")
        L.append("  accuracy: " + "  ".join(f"{c} {x:.1f}" for c, x in cc["acc"].items()))
        for c, (x, (lo, hi)) in cc["drop"].items():
            L.append(f"  {c:6s} drop {x:+5.1f} [{lo:+.1f}, {hi:+.1f}]  answers changed {cc['change'][c]:.1f}%")
        res.setdefault("count", {})[m] = {"compare": cc}
    print("\n".join(L))
    if a.json:
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


def cross_check(a):
    """Independent re-computation of both experiments, written separately from the functions above; adds the mean
    of the two option orders (used in the paper where a quantity moves by more than 5 points)."""
    def load(p, cond):
        R = defaultdict(dict)
        rows = read_jsonl(p)
        for r in (C.main_rows(rows) if cond == "variant" else rows):
            R[r["idx"]][r[cond]] = r
        return R

    def counts(R, ids, q):
        _, v, ref, ax, kind = q
        k = n = 0
        for i in ids:
            x = R[i]
            if x["orig"]["axis"] != ax:
                continue
            if kind == "agree":
                n += 1
                k += x[v]["pred"] == x[ref]["pred"]
            elif x["orig"]["pred"] == x["orig"]["expected"] and x[ref]["pred"] == x[ref]["expected"]:
                n += 1
                k += x[v]["pred"] != x["orig"]["pred"]
        return k, n

    def pct(k, n):
        return 100 * k / n if n else float("nan")

    def split(R):
        out = {}
        for x in R.values():
            o = x["orig"]
            if o["axis"] != "lr" or o["pred"] != o["expected"] or x["hflip"]["pred"] != x["hflip"]["expected"]:
                continue
            w = o["pred"]
            for key in ("word=" + w, "letter=" + "AB"[o["choices"].index(w)]):
                c = out.setdefault(key, [0, 0])
                c[1] += 1
                c[0] += x["vit_w"]["pred"] != w
        return {k: (round(pct(*v), 1), v[1]) for k, v in sorted(out.items())}

    def n_ties(R):
        return sum(r["logp"][0] == r["logp"][1] for x in R.values() for r in x.values())

    B = a.B
    for m in sorted(a.orig):
        O, S = load(a.orig[m], "variant"), load(a.swap[m], "variant")
        ids = sorted(O)
        assert ids == sorted(S) and len(ids) == 650 and all(len(O[i]) == len(S[i]) == len(C.CONDITIONS) for i in ids)
        assert all(O[i][v]["choices"] == S[i][v]["choices"][::-1] for i in ids for v in O[i])
        print(f"== Swapped order {m}: ties original {n_ties(O)} / swapped {n_ties(S)}")
        samples = [[ids[j] for j in s] for s in resamples(len(ids), B)]
        over = []
        for q in KEY_QUANTITIES:
            ko, no = counts(O, ids, q)
            ks, ns = counts(S, ids, q)
            d = pct(ks, ns) - pct(ko, no)
            lo, hi = percentile_ci(sorted(pct(*counts(S, smp, q)) - pct(*counts(O, smp, q)) for smp in samples))
            print(f"  {q[0]:22s} original {pct(ko, no):5.1f} (n={no})  swapped {pct(ks, ns):5.1f} (n={ns})  diff {d:+5.1f} [{lo:+.1f}, {hi:+.1f}]"
                  f"  mean of both orders {(pct(ko, no) + pct(ks, ns)) / 2:.1f}")
            if abs(round(d, 1)) > MOVE_POINTS:
                over.append(q[0])
        print("  split by the original answer, original order", split(O))
        print("  split by the original answer, swapped       ", split(S))
        print("  more than 5 points:", over or "none")

    conds = list(COUNT_CONDS)
    for m in sorted(a.count):
        R = load(a.count[m], "cond")
        ids = sorted(R)
        assert len(ids) == 788 and all(len(R[i]) == 5 for i in ids)
        # the stored prediction must be the argmax of the stored scores (first one on ties)
        bad = sum(r["pred"] != max(range(len(r["lp"])), key=lambda j: (r["lp"][j], -j)) for x in R.values() for r in x.values())
        acc = {c: 100 * sum(R[i][c]["pred"] == R[i][c]["gold"] for i in ids) / len(ids) for c in conds}
        print(f"== Count {m} (n={len(ids)}, pred differs from the lp argmax: {bad}) accuracy " + "  ".join(f"{c} {x:.1f}" for c, x in acc.items()))
        samples = [[ids[j] for j in s] for s in resamples(len(ids), B)]
        ok = {c: {i: int(R[i][c]["pred"] == R[i][c]["gold"]) for i in ids} for c in conds}
        for c in conds[1:]:
            d = acc["orig"] - acc[c]
            lo, hi = percentile_ci(sorted(100 * sum(ok["orig"][i] - ok[c][i] for i in s) / len(s) for s in samples))
            chg = 100 * sum(R[i][c]["pred"] != R[i]["orig"]["pred"] for i in ids) / len(ids)
            print(f"  {c:6s} drop {d:+5.1f} [{lo:+.1f}, {hi:+.1f}]  answers changed {chg:.1f}%")


R = "results/controls/"
M = "results/main_table/qwen2.5-vl/"
_FILES = (["--orig", f"3b={M}qwen2.5-vl-3b-instruct.jsonl", f"7b={M}qwen2.5-vl-7b-instruct.jsonl",
           "--swap", f"3b={R}swap_3b.jsonl", f"7b={R}swap_7b.jsonl", "--count", f"3b={R}count_3b.jsonl", f"7b={R}count_7b.jsonl"])
# (output file, arguments of the analysis command that writes it, behind a table, figure or the main text)
OUTPUTS = [
    ("outputs/controls/summary.txt", ["summary", *_FILES, "--repro", f"3b={R}repro_3b.jsonl", f"7b={R}repro_7b.jsonl"], True),
    ("outputs/controls/cross_check.txt", ["cross-check", *_FILES], True),
]


def _kv(xs):
    return dict(x.split("=", 1) for x in xs or [])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.controls", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run-order", run_order), ("run-count", run_count)):
        r = sub.add_parser(name, help="GPU: " + fn.__doc__.split("\n")[0])
        r.add_argument("--model", required=True)
        r.add_argument("--parquet", required=True, help="CV-Bench test_2d.parquet")
        r.add_argument("--out", required=True)
        r.add_argument("--limit", type=int, default=0, help="first N questions only")
        r.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
        if name == "run-order":
            r.add_argument("--order", choices=["fwd", "swap"], required=True, help="fwd: options as in the main table")
    for name, fn in (("summary", summary), ("cross-check", cross_check)):
        s = sub.add_parser(name, help=fn.__doc__.split("\n")[0])
        for opt in ("--orig", "--swap", "--count"):
            s.add_argument(opt, nargs="*", help="model=predictions.jsonl ...")
        if name == "summary":
            s.add_argument("--repro", nargs="*", help="model=re-run in the original order")
            s.add_argument("--json", help="also write the numbers as JSON")
        s.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    a = ap.parse_args(argv)
    if a.cmd in ("run-order", "run-count"):
        {"run-order": run_order, "run-count": run_count}[a.cmd](a)
        return
    for opt in ("orig", "swap", "count", "repro"):
        if hasattr(a, opt):
            setattr(a, opt, _kv(getattr(a, opt)))
    {"summary": summary, "cross-check": cross_check}[a.cmd](a)


if __name__ == "__main__":
    main()
