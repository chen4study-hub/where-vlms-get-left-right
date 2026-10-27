"""Post hoc: the vertical content-only control on What'sUp, split by whether the real flip changes the answer (paper Section 4.5).

Input: the per-item predictions of the What'sUp content-only control (conditions orig, vflip, vflip_both_h, hflip,
hflip_both_w; one file per model). Flip-back rate: the answer under pix_both differs from the answer under pix
(up-down: pix = vflip, pix_both = vflip_both_h; left-right: pix = hflip, pix_both = hflip_both_w).
  1. share of items whose answer the real flip leaves unchanged (pix answer = orig answer);
  2. flip-back rates and D = FB_h - FB_v on the items whose answer the real flip changes (unpaired bootstrap,
     B resamples, seed 0);
  3. geometry-consistent items (orig and real flip both correct): share of pix_both answers equal to the orig
     answer, up-down minus left-right (unpaired bootstrap).

  python -m experiments.extra vertical_control_split summary --dir results/further_checks/vertical_content [--B 10000]
"""
import argparse
import json
import os
import random
from collections import defaultdict

INV = {"left": "right", "right": "left", "above": "below", "below": "above", "on": "under", "under": "on"}
AX = {"ud": ("vflip", "vflip_both_h"), "lr": ("hflip", "hflip_both_w")}


def load(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"per-item predictions not found: {p} (run the experiment first, see README)")
    d = defaultdict(dict)
    for line in open(p):
        r = json.loads(line)
        d[r["idx"]][r["cond"]] = r
    return d


def pct(x):
    return "  nan" if x != x else f"{100 * x:5.1f}"


def boot2(f1, i1, f2, i2, B, seed=0):
    rng = random.Random(seed)
    i1, i2 = sorted(i1), sorted(i2)
    vals = sorted(f1([i1[rng.randrange(len(i1))] for _ in i1]) - f2([i2[rng.randrange(len(i2))] for _ in i2])
                  for _ in range(B))
    return f1(i1) - f2(i2), (vals[int(0.025 * B)], vals[int(0.975 * B) - 1])


def summary(a):
    print(f"== What'sUp content-only control split by whether the real flip changes the answer (post hoc); unpaired bootstrap B={a.B} seed 0")
    for m in ("q3b", "q7b", "ivl2b"):
        d = load(f"{a.dir}/{m}_whatsup.jsonl")
        ids = {ax: sorted(i for i, v in d.items() if v["orig"]["axis"] == ax and AX[ax][1] in v) for ax in AX}

        def fb(ax):
            pix, both = AX[ax]
            return lambda I: sum(d[i][both]["pred"] != d[i][pix]["pred"] for i in I) / len(I)

        def same(ax):
            pix = AX[ax][0]
            return [i for i in ids[ax] if d[i][pix]["pred"] == d[i]["orig"]["pred"]]

        def disc(ax):
            pix = AX[ax][0]
            return [i for i in ids[ax] if d[i][pix]["pred"] != d[i]["orig"]["pred"]]

        su, sl = same("ud"), same("lr")
        du, dl = disc("ud"), disc("lr")
        D_all, ci_all = boot2(fb("lr"), ids["lr"], fb("ud"), ids["ud"], a.B)
        D_dis, ci_dis = boot2(fb("lr"), dl, fb("ud"), du, a.B)
        print(f"  {m:5s} real flip leaves the answer unchanged: up-down {len(su)}/{len(ids['ud'])} {pct(len(su) / len(ids['ud']))}; "
              f"left-right {len(sl)}/{len(ids['lr'])} {pct(len(sl) / len(ids['lr']))}")
        print(f"        all items: FB_v {pct(fb('ud')(ids['ud']))} FB_h {pct(fb('lr')(ids['lr']))} "
              f"D {pct(D_all)} [{pct(ci_all[0]).strip()}, {pct(ci_all[1]).strip()}]")
        print(f"        items whose answer the real flip changes: up-down n={len(du)} FB_v {pct(fb('ud')(du))}; left-right n={len(dl)} FB_h {pct(fb('lr')(dl))}; "
              f"D {pct(D_dis)} [{pct(ci_dis[0]).strip()}, {pct(ci_dis[1]).strip()}]")
        if su:
            print(f"        up-down items whose answer the real flip leaves unchanged n={len(su)}: FB_v {pct(fb('ud')(su))}")

        def gc(ax):
            pix = AX[ax][0]
            return sorted(i for i in ids[ax] if d[i]["orig"]["pred"] == d[i]["orig"]["gold"]
                          and d[i][pix]["pred"] == INV.get(d[i]["orig"]["gold"], d[i]["orig"]["gold"]))

        def back(ax):
            both = AX[ax][1]
            return lambda I: sum(d[i][both]["pred"] == d[i]["orig"]["pred"] for i in I) / len(I)

        gu, gl = gc("ud"), gc("lr")
        Dg, cig = boot2(back("ud"), gu, back("lr"), gl, a.B)
        print(f"        GC (orig and real flip both correct): up-down n={len(gu)} back to the orig answer {pct(back('ud')(gu))}; "
              f"left-right n={len(gl)} back to the orig answer {pct(back('lr')(gl))}; up-down − left-right {pct(Dg)} "
              f"[{pct(cig[0]).strip()}, {pct(cig[1]).strip()}]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.extra vertical_control_split", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="the split, from the per-item predictions")
    s.add_argument("--dir", required=True, help="directory with {q3b,q7b,ivl2b}_whatsup.jsonl")
    s.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(argv)
    summary(a)


OUTPUTS = [("outputs/further_checks/vertical_content_only_whatsup_split.txt",
            ["summary", "--dir", "results/further_checks/vertical_content"])]
