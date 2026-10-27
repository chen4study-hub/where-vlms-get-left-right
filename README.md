# Where Do Vision-Language Models Get Left and Right?

Code for the paper *Where Do Vision-Language Models Get Left and Right? Position-Code Counterfactuals in
Qwen2.5-VL, Qwen3-VL and InternVL3*. The experiments mirror only the position information that a vision-language model attaches to image
patches, on the ViT side or on the LLM side, without changing a single pixel, and compare the answers with those
for a real horizontal or vertical flip. Models: Qwen2.5-VL-3B / 7B (2D-RoPE in the ViT, M-RoPE in the LLM),
InternVL3-2B / 8B (learned absolute position embedding in the ViT, 1D RoPE in the LLM) and Qwen3-VL-4B.
Qwen2.5-VL-7B, Qwen3-VL-4B and InternVL3-2B are the headline models, one per series; Qwen3-VL-4B was run with the
ViT- and LLM-side mirrors on CV-Bench and What'sUp and with the same-axis permutation controls.

This repository holds the code and the summary output of every analysis (`outputs/`); the per-item predictions of
the runs are not included. Each experiment is run with `python -m experiments.<module> ...`: the models are run first
to produce the predictions (GPU; models and data in [DATA.md](DATA.md)), then the module's analysis commands
summarize them.

## Repository layout

```
vlmlr/               shared package
  positions.py         position-id arithmetic: M-RoPE and ViT 2D-RoPE mirrors, InternVL grid mirrors, token permutations
  hooks.py             hooks that apply the counterfactuals to a loaded model (whole model, chosen layers, token order)
  models.py            model loading, prompts, option-letter scoring (bf16 logits or fp32), greedy generation
  cvbench.py           CV-Bench 2D Relation items, pixel transforms, the 19 conditions of the main table
  whatsup.py           What'sUp items and relation bookkeeping
  depth.py             CV-Bench 3D Depth and EmbSpatial items, heuristic groups
  permutations.py      same-axis non-mirror permutations of the position code, removal of the ViT width code
  stats.py             Wilson intervals and the paired bootstrap (resampled items, seed 0)
  records.py           per-item prediction files (JSON lines, resumable)
  geometry.py          image resize rule of the Qwen2.5-VL processor and box helpers
  tables/              fixed CV-Bench validation splits and the shuffled merge-unit orders of the permutations (JSON)
experiments/         one module per experiment of the paper; experiments/extra/ for the smaller analyses
outputs/             the summary output of every analysis (text), one directory per experiment
```

Each module is run as `python -m experiments.<module> <command>`; `--help` lists the commands, and the module's
docstring gives the commands used for the paper. The smaller analyses live in `experiments/extra/`
(`python -m experiments.extra` lists them).

## Installation

Running the experiments needs Python 3.12, one GPU with 32 GB and
`pip install -r requirements.txt` (torch 2.5.1 + CUDA 12.4, transformers 4.57.6; the hooks are written against this
transformers version).

## Running the experiments

1. Models and data: download the benchmarks and models at the revisions in [DATA.md](DATA.md). Items and images built for this study are made by the modules' `items*` and `make-data` commands.
2. Run an experiment, e.g. `python -m experiments.main_table run --family qwen --model Qwen/Qwen2.5-VL-7B-Instruct
   --parquet <dir>/test_2d.parquet --out results/main_table/qwen2.5-vl/qwen2.5-vl-7b-instruct.jsonl`. Every run
   checks on its first item that its hooks reproduce the unmodified model where they should, writes one JSON line
   per item and condition (with option log-probabilities), and resumes an interrupted file; `--limit N` runs the
   first N items.
3. Summarize: the analysis commands of each module (listed by `--help`) compute the summaries in `outputs/` from
   the prediction files; each module's `OUTPUTS` lists the command and the prediction files behind every output.

## Experiments and the paper

| Paper | Module | Outputs (under `outputs/`) |
|---|---|---|
| Sec. 4.1, Table 2, Fig. 2a: ViT width mirror vs real flip on CV-Bench, four models; Appendix B: InternVL3-8B; Appendix C, Table C.1: ties | `main_table` | `main_table/` |
| Sec. 4.1, 4.2, 4.5, 4.6, Appendix B: same-axis non-mirror permutations of the ViT code (width and height), depth reference with a real vertical flip, ViT width code removed, InternVL3 token order | `controls_permutation` | `permutation_controls/` |
| Sec. 4.2, Table 3: LLM-side token order and position ids; InternVL3 tile order | `token_order`, `extra tiles` | `token_order/`, `extra/tiles.txt` |
| Sec. 4.1, 4.2, 4.4–4.7, Tables 2 and 4, Fig. 3a, Appendices B and C: Qwen3-VL-4B and its same-axis permutations, ViT block sets, InternVL3 default tiling and reversed options (Table C.3), vertical content-only control; post hoc: object order split, depth conflict items and shared source images, object boxes | `further_checks`, `extra vertical_control_split` | `further_checks/` |
| Sec. 4.3, Table 4, Fig. 2b: What'sUp; Appendix B: off-axis swaps, InternVL3-8B; Appendix C: ties | `whatsup` | `whatsup/` |
| Sec. 4.4, Fig. 3: single ViT blocks and dose curves | `blocks_dose` | `blocks_dose/` |
| Sec. 4.5, Fig. 4a: position / content / prior decomposition; Sec. 5.1: error types | `decomposition` | `decomposition/` |
| Sec. 4.6, Table 5, Fig. 4b: depth cue; Appendix D: EmbSpatial | `depth_shortcut` | `depth_shortcut/` |
| Sec. 4.7, Fig. 6, Appendix C, Table C.3: options in reverse order (two-order means in Table 2); counting under the mirrors | `controls` | `controls/` |
| Sec. 4.7, Fig. 5, Appendix E, Tables E.1 and E.2: public fine-tuned checkpoints and component swap | `public_checkpoints` | `public_checkpoints/` |
| Sec. 4.7: video input; Sec. 4.7, Appendix B: word reading, referring localization (boxes classified in `further_checks/box_reclassification.txt`) | `extra` | `extra/` |
| Post hoc analyses quoted in Sec. 3.4, 4.1–4.6, 5.1, Appendix B and Table C.2 (left prior, rotation angles, ties, depth differences, content-only direction count, errors that match the blank-image answer) | `posthoc_analyses` | `posthoc_analyses/` |

## Statistics

Intervals: 95 % Wilson intervals for proportions; paired bootstrap 95 % intervals for differences (items resampled
with replacement, `random.Random(0)`), 10,000 resamples, 2,000 in the analyses stated in the paper.

## License

The code is released under the MIT License ([LICENSE](LICENSE)); so are the outputs. Third-party datasets and models
keep their licenses; see [DATA.md](DATA.md).
