# Data, models and licenses

The repository redistributes **no third-party data, image, video or model weight**. Every dataset and model below
is public and is downloaded from its source at the revision we used (see Download below). Licenses were checked on 2026-10-26 against the Hugging Face cards, the project pages and the GitHub
repositories of each source.

## Datasets

| Dataset | Source and revision | Used by | License (as stated by the source) | Upstream image terms |
|---|---|---|---|---|
| CV-Bench 2D (Relation, Count) and 3D (Depth) | HF `nyu-visionx/CV-Bench` (`test_2d.parquet`, `test_3d.parquet`, `test_3d.jsonl`) @ `bc284db50d036958861cb60cdd7b77612052ce0d` | `main_table`, `blocks_dose`, `token_order`, `decomposition`, `depth_shortcut`, `controls`, `controls_permutation`, `further_checks`, `public_checkpoints`, `extra` | Apache-2.0 | 2D: COCO val2017 (Flickr terms), ADE20K val (non-commercial research); 3D: Omni3D (CC BY-NC 4.0) and its sources |
| What'sUp, Controlled Images A | HF `amitakamath2/whatsup_vlms` @ `373b38706ec0f20a2506f61d6e7cae92aa89da31` | `whatsup`, `controls_permutation`, `further_checks` | MIT | photographed by the authors |
| EmbSpatial-Bench | HF `Phineas476/EmbSpatial-Bench` (`embspatial_bench.json`) @ `e2733cacf0f20371518c687176e881d79644e61e` | `depth_shortcut` | CC BY 4.0 | Matterport3D (academic EULA), ScanNet (terms of use), AI2-THOR |
| RefCOCO (val) | HF `lmms-lab-encoder/RefCOCO`, LMMs-Lab packaging @ `a5dff0b3194715fda69b2d6a0d2aaafb41eaa407` | `extra box_readout`, `further_checks` | refer toolkit Apache-2.0; COCO annotations CC BY 4.0 | COCO train2014 (Flickr terms) |
| COCO val2017 instance annotations (`instances_val2017.json`) | cocodataset.org, 2017 Train/Val annotations | `further_checks object-boxes` (boxes of the two objects of the CV-Bench COCO questions) | CC BY 4.0 | COCO val2017 (Flickr terms) |

**Data created for this study** (rendered word images, the depth items with their heuristic groups) are re-created
from the public sources by the modules' `items*` and `make-data` commands.

**Attribution.** Please cite the original works of the datasets above, among them CV-Bench (Tong et al., 2024),
What'sUp (Kamath et al., 2023), EmbSpatial-Bench (Du et al., 2024) and RefCOCO (Yu et al., 2016; Kazemzadeh et al.,
2014).

## Download

The datasets (needs `pip install huggingface_hub`):

```bash
huggingface-cli download nyu-visionx/CV-Bench --repo-type dataset --revision bc284db50d036958861cb60cdd7b77612052ce0d --include test_2d.parquet test_3d.parquet test_3d.jsonl --local-dir datasets/nyu-visionx__CV-Bench
huggingface-cli download amitakamath2/whatsup_vlms --repo-type dataset --revision 373b38706ec0f20a2506f61d6e7cae92aa89da31 --local-dir datasets/amitakamath2__whatsup_vlms
huggingface-cli download Phineas476/EmbSpatial-Bench --repo-type dataset --revision e2733cacf0f20371518c687176e881d79644e61e --include embspatial_bench.json --local-dir datasets/Phineas476__EmbSpatial-Bench
huggingface-cli download lmms-lab-encoder/RefCOCO --repo-type dataset --revision a5dff0b3194715fda69b2d6a0d2aaafb41eaa407 --include "data/val-*" --local-dir datasets/lmms-lab-encoder__RefCOCO
```

The COCO val2017 instance annotations come from cocodataset.org. Each model below is downloaded with
`huggingface-cli download <model> --revision <revision>`.

## Models

| Model | Revision | License | Role |
|---|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | Qwen Research License (non-commercial) | core model |
| Qwen/Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | Apache-2.0 | core model; headline model |
| OpenGVLab/InternVL3-2B-hf | `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` | per model card | core model; headline model |
| OpenGVLab/InternVL3-8B-hf | `259a3b64a14623c0ec91a045cb43f7c5af5fa6af` | per model card | scale check |
| Qwen/Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 | headline model (ViT- and LLM-side mirrors, CV-Bench up-down mirrors, same-axis permutation controls) |
| hongxingli/SpatialLadder-3B | `0819c3adf8827a2ea6c0348d49a23503ecb1f428` | Apache-2.0 | public fine-tuned checkpoint, component swap |
| remyxai/SpaceQwen2.5-VL-3B-Instruct | `332ff3fbc40d0e62af64b1bd2e0a606b3cca028e` | Qwen Research License | public fine-tuned checkpoint |
| RUBBISHLIKE/SpaceR-3B | `079d963625710049514e93c99ca855bc8a43a19d` | no stated license; research evaluation only | public fine-tuned checkpoint |
| omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps | `b9a4c5aa915faddc7e129c3f61d630ceef73a911` | no stated license; research evaluation only | public fine-tuned checkpoint |
