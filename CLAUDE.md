# Project: PCB Image Harmonization

## Active Work
- Backbone migration from FLUX to **Qwen-Image-Edit-2511** (20B MMDiT)
- Training at 1024×1024 with LoRA on NTU cluster
- Check `STATUS.md` for current active state (job IDs, blockers, next actions)

## Cluster Rules
- SSH: `ssh xinrui004@10.97.216.128`
- **NEVER** use `--constraint=gpu` — always request specific GPU type: `--gpus=a6000:4` or `--gpus=pro6000:4`
- **NEVER** submit SLURM jobs without explicit user approval
- **ALWAYS** patch load_model.py to SDPA before GPU jobs: `sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' src/qflux/models/load_model.py`
- All data/envs/cache on `/projects/_ssd/xrssd/`, NEVER write to home (50GB limit)
- Use `rsync` from L40S for large transfers to cluster (~450MB/s vs ~20MB/s from HF)

## Code Locations
- **L40S (local):** `~/projects/OminiControl/` (data gen), `~/projects/Qwen-Image/qwen-image-finetune/` (training code)
- **Cluster:** `~/xrssd/qwen-image-finetune/` (training), `~/xrssd/data/ti_pcb/` (data)
- **Dataset:** `~/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize/` (34,281 samples)
- **Cache:** `~/xrssd/runs/pcb_harmonize_qwen_edit_v1/cache/`
- **Config:** `~/xrssd/qwen-image-finetune/configs/pcb_harmonize_qwen_edit_2511.yaml`

## Workflow Preferences
- Keep sessions focused — one task per session, update STATUS.md before ending
- Check memory files for project history and decisions
- Show scripts/plans before executing — user validates before pushing
- Use wandb for training monitoring (key in train.sh)
