# Active Status (2026-04-10)

## COMPLETED: Full Test Set Evaluation (2026-04-08)

### Eval Setup
- 2,186 patches from 405 test boards (512×512 crops, stride=384, 25% overlap)
- Eval JSON: `/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/eval_patches_full_test.json`
- Inference: 1024×1024 generation, downscaled to 512
- Script: `infer_full_eval.py` (split across 2 GPUs via `--start`/`--end`)
- Gallery: `eval_metrics/gallery_comparison.html` (50 sampled patches)

### Results Summary

| Metric | Real (Baseline) | v3_newdata ckpt14k (512-trained) | v3.1_1024 ckpt6k (1024-trained) |
|--------|----------------|----------------------------------|--------------------------------|
| **FID** | — | 37.50 | **33.50** |
| **KID** | — | 0.0223 | **0.0186** |
| **Mean IoU** | 0.9325 | 0.7908 | 0.7947 |
| **Class accuracy** | 0.9337 | **0.7804** | 0.7535 |
| **Precision** | 0.8467 | **0.8167** | 0.7045 |
| **Recall** | 0.9123 | **0.8986** | 0.8900 |
| **F1** | 0.8783 | **0.8557** | 0.7865 |
| **FP/image** | 4.36 | **5.32** | 9.85 |
| **Hallucinated/image** | 3.24 | **3.45** | 7.98 |
| **AP@0.5** | 0.655 | **0.303** | 0.253 |
| **AP@[.5:.95]** | 0.593 | **0.176** | 0.146 |

### Per-Class Recall (YOLO detection on generated images)

| Class | GT Count | Real Baseline | v3_newdata 14k | v3.1_1024 6k |
|-------|----------|--------------|----------------|--------------|
| RESISTOR | 18,119 | 0.84 | 0.73 | 0.71 |
| CAPACITOR | 20,698 | 0.88 | 0.78 | 0.75 |
| IC | 4,660 | 0.88 | 0.74 | 0.66 |
| CONNECTOR | 9,465 | 0.89 | 0.72 | 0.69 |
| DIODE | 2,538 | 0.75 | 0.21 | 0.16 |
| INDUCTOR | 963 | 0.54 | 0.17 | 0.14 |
| OSCILLATOR | 290 | 0.52 | 0.16 | 0.19 |
| TRANSISTOR | 421 | 0.56 | 0.08 | 0.05 |
| SWITCH | 562 | 0.48 | 0.06 | 0.11 |

### Top Confusions (v3_newdata ckpt14k)
- RESISTOR ↔ CAPACITOR: 5,283 mutual errors (tiny rectangles, hard to distinguish)
- DIODE → CAPACITOR: 923 errors
- SWITCH → CONNECTOR: 285 errors
- TRANSISTOR → IC: 165 errors

### Key Findings
1. **v3.1 (1024-trained) has better FID/KID** but **hallucinates 2x more** (7.98 vs 3.45 FP/image)
2. **v3_newdata (512-trained) wins on detection metrics** — better precision, fewer FPs, better class accuracy
3. **Rare classes (Switch, Transistor, Inductor, Diode) are poorly regenerated** — model generates generic shapes
4. **RESISTOR ↔ CAPACITOR confusion is inherent** — even YOLO on real images confuses them (2,004 errors)
5. **Most "hallucinations" are YOLO noise** — baseline has 3.24/image, generated adds only +0.21 (v3_newdata)
6. **COCO AP is low primarily due to class-conditional matching** — 22% class error turns TPs into FPs

### Output Locations
- v3_newdata ckpt14k: `runs/v3_newdata/eval_full/ckpt14k/` (fid_kid_metrics.json, detection_metrics.json)
- v3.1_1024 ckpt6k: `runs/v3.1_newdata_1024/eval_full/ckpt6k/` (fid_kid_metrics.json, detection_metrics.json)
- Real baseline: `eval_metrics/real_crops/detection_metrics_baseline.json`
- Real crops: `eval_metrics/real_crops/` (2186 images)

## COMPLETED: v3_newdata 30-sample evals (2026-04-07)
| Checkpoint | Normal | Dense |
|---|---|---|
| dir 6000 (ckpt6k) | `runs/v3_newdata/eval/ckpt6k_normal/gallery.html` ✓ | `runs/v3_newdata/eval/ckpt6k_dense/gallery.html` ✓ |
| dir 14000 (ckpt14k) | `runs/v3_newdata/eval/ckpt14k_normal/gallery.html` ✓ | `runs/v3_newdata/eval/ckpt14k_dense/gallery.html` ✓ |

## All v3_newdata checkpoints downloaded to local
- `runs/v3_newdata/ckpt/` — dirs 1000–14000 (111MB each)
- v3.1_1024 ckpt 6000 also downloaded: `runs/v3.1_newdata_1024/ckpt/6000/`
- Cluster source: `/projects/_ssd/xrssd/OminiControl/runs/20260406-081503/ckpt/`

## RUNNING: Job 55626 — OminiControl v3.1_newdata_1024 (FLUX)
- At step ~6440/8000, ~7h remaining
- From scratch, 8k steps, FLUX.1-dev backbone, **1024×1024 resolution**
- bs=2/gpu, accum=2, effective bs=16, omini env
- Output: `/projects/_ssd/xrssd/OminiControl/runs/v3.1_newdata_1024/`

## COMPLETED: Job 55597 — OminiControl v3_newdata (FLUX 512)
- Checkpoints saved: 1k–14k
- Output: `/projects/_ssd/xrssd/OminiControl/runs/20260406-081503/`
- wandb: OminiControl-PCB

## COMPLETED: Job 55263 — Qwen-Edit v2 + layout prompts
- **Cancelled at step ~9011 on 2026-04-06 08:33:32** (22h49m elapsed, manual cancel)
- Did NOT reach 10k target; last saved checkpoint = step 9000
- Training: resumed from step 4000 (job 53920 crashed at 4008 with NCCL timeout)
- Config: r=32, alpha=64, bs=4/gpu × 4 pro6000, accum=1, effective bs=16, flash_attn env
- Cluster path: `~/xrssd/qwen-image-finetune/`
- wandb: pcb_harmonize_qwen_edit_v2_layoutPrompt (resumed run f1h8vpld)
- Checkpoints:
  - v0/ (original run): checkpoint-0-1000, 0-2000, 1-3000, 1-4000
  - v1/ (resumed run): checkpoint-1-5000, 1-6000, 2-7000, 2-8000, 3-9000
  - backup_ckpts/: copy of v0 checkpoints

## Completed Evals

### Qwen V1 (no layout prompts) — infer_qwen_edit.py
| Checkpoint | Normal | Dense |
|---|---|---|
| Step 1000 | step1000_normal_v2new ✓ | - |
| Step 2000 | step2000_normal_v2new ✓ | - |
| Step 3000 | step3000_normal_v2new ✓ | step3000_dense_v2new ✓ |
| Step 4000 | step4000_normal_v2new ✓ | step4000_dense_v2new ✓ |
| Step 5000 | step5000_normal_v2new ✓ | step5000_dense_v2new ✓ |

### Qwen V2 (with layout prompts) — infer_qwen_edit_v2.py
| Checkpoint | Normal | Dense |
|---|---|---|
| Step 4000 (v0/) | step4000_normal_v2new ✓ | step4000_dense_v2new ✓ |
| Step 5000 (v1/) | step5000_normal_v2new ✓ | — (skipped) |
| Step 8000 (v1/) | step8000_normal_v2new ✓ | step8000_dense_v2new ✓ |
| Step 9000 (v1/) | PENDING — waiting for L40 GPUs | PENDING — waiting for L40 GPUs |

**Prompt format (verified correct):** natural language template by `board_color` + `\nLayout: IC (x,y,w,h); C (...); ...` — matches v2 training data exactly.

**Next inference run:** Step 9000 is the final v2 checkpoint (training cancelled there). Checkpoint exists at cluster path `/projects/_ssd/xrssd/qwen-image-finetune/runs/pcb_harmonize_qwen_edit_v2_layoutPrompt/v1/checkpoint-3-9000`. Not yet downloaded to L40. Run normal + dense inference when GPUs are free.

### FLUX baseline (OminiControl v2.1)
- v2.1 512×512: `OminiControl/runs/v2.1_pcb_harmonize/eval/smallcomp_512/`
- v2.1 1024×1024: `OminiControl/runs/v2.1_pcb_harmonize/eval/smallcomp_1024/`

## Inference Scripts
- `infer_qwen_edit.py` — v1 (no layout prompts), uses trainer.predict()
- `infer_qwen_edit_v2.py` — v2 (appends Layout line to prompt), uses trainer.predict()
- Both use ComponentBankV2_new with v2 annotations (color/resolution/orientation matching)

## Training Data Gallery
- `gallery_v2_training/gallery.html` — 30 samples from 3 boards (control | mask | target)

## COMPLETED: Training job 53281 (Qwen v1 baseline)
- 5k steps, checkpoints 1k-5k all downloaded locally
- wandb: pcb_harmonize_qwen_edit_v1

## A800 Cluster Migration Package
- Location: `/projects/xiyu004shared/xinrui_qwen/`
- **TODO:** Re-sync capped prompts + new cache

## Previous Bugs Fixed
1. DDP tensor shape mismatch — cap at 40 components
2. Dataset contamination — restored from backup
3. Config report_to — only accepts wandb/tensorboard/swanlab
4. dataset.py cache_exists patch — must revert after caching
5. HF_TOKEN crash — graceful skip
6. infer_range — uniform-value images
7. RGBA — palette/transparent images composited onto white background
8. NCCL watchdog timeout — increased to 30min, transient hardware issue
