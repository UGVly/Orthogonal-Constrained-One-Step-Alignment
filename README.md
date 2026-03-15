# Orthogonal-Constrained One-Step Alignment

A cleaned project layout for **test-time orthogonal noise optimization** on **SDXL Turbo**, now extended with a **latent-matched SFT** workflow for one-step generators.

Besides the original reward-driven test-time optimization backends, this repo now includes a practical version of the idea:

> direct one-step SFT on high-quality `(image, prompt)` pairs is often unstable, so first assign the most compatible latent noise to each target image, then fine-tune on those matched noises instead of random noises.

That gives you a cleaner supervision signal and usually behaves much better than naive random-noise SFT.

## What is new in this version

The project now has three connected pieces for the new idea:

1. **Single-sample assigned-noise inversion**  
   Given one target image and one prompt, optimize a patch-wise orthogonal noise transform and save the best matched latent noise.

2. **Assigned-noise dataset builder**  
   Given a `pairs.jsonl`, build a dataset where each sample contains:
   - prompt
   - target image
   - best matched input noise

3. **Latent-matched SFT trainer**  
   Fine-tune the one-step SDXL Turbo model on those matched noises, with optional **preserve/distillation loss** so the model does not forget its original random-noise behavior too aggressively.

There is also an **EM-style alternating loop**:

- E-step: reassign matched noises with the current model
- M-step: fine-tune on the updated matched-noise dataset

## Project layout

```text
src/ttt_reward_models/
  adapters.py
  cli_sdxl_reward.py
  cli_pickscore.py
  cli_imagereward.py
  cli_hpsv2.py
  cli_noise_theory.py
  cli_assign_sdxl_sft.py
  cli_build_assigned_noise_dataset.py
  cli_train_latent_matched_sft.py
  cli_em_latent_matched_sft.py
  data.py
  diagnostics.py
  downloaders.py
  paths.py
  pipeline.py
  rewards_clip.py
  rewards_pickscore.py
  rewards_imagereward.py
  rewards_hpsv2.py
  runners.py
  runners_sft.py
  utils.py
legacy/
legacy_reference/
scripts/
examples/
third_party_weights/
models/
```

## Install

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## Download models into the project-local `models/` folder

```bash
bash scripts/download_models.sh
```

That script downloads the main assets into a project-local layout like:

```text
models/
  sdxl-turbo/
  Hyper-SD15-1step/
  PickScore_v1/
  ImageReward/
  HPSv2/
```

## Reward-model assets inside the repo

Some reward assets are also stored under:

```text
third_party_weights/
  imagereward_modelscope/
  hpsv2/
```

## Original reward-driven one-step optimization

### CLIP / Aesthetic / Hybrid

```bash
python -m ttt_reward_models.cli_sdxl_reward \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --reward_type clip \
  --model_id ./models/sdxl-turbo \
  --patch_size 8
```

### PickScore

```bash
python -m ttt_reward_models.cli_pickscore \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --model_id ./models/sdxl-turbo \
  --patch_size 8
```

### ImageReward

```bash
python -m ttt_reward_models.cli_imagereward \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --model_id ./models/sdxl-turbo \
  --imagereward_auto_download \
  --patch_size 8
```

### HPSv2

```bash
python -m ttt_reward_models.cli_hpsv2 \
  --prompt "a cinematic portrait of a girl in soft light, highly detailed" \
  --model_id ./models/sdxl-turbo \
  --hps_version v2.1 \
  --hps_auto_download \
  --patch_size 8
```

## Noise-theory visualization

Every reward CLI and assigned-noise inversion run writes diagnostics such as:

```text
orthogonal_gaussian_init.png / .json
orthogonal_gaussian_final.png / .json
```

These compare the original standard Gaussian latent noise with the orthogonally transformed noise `Q @ z`, showing that the Gaussian statistics are preserved to a very high degree.

Standalone verification:

```bash
python -m ttt_reward_models.cli_noise_theory \
  --output_dir outputs/orthogonal_gaussian_theory \
  --channels 4 \
  --patch_size 2 \
  --num_samples 65536 \
  --batch_size 4096
```

## New workflow: latent-matched SFT

## Expected data format

Create a dataset folder like this:

```text
data/high_quality_pairs/
  pairs.jsonl
  images/
    0001.png
    0002.png
```

Where each line in `pairs.jsonl` looks like:

```json
{"prompt": "a cinematic portrait of a girl in soft light", "image": "images/0001.png"}
```

You can rename the fields with `--prompt_field` and `--image_field`.

## Step 1: assign the best noise for a single sample

```bash
python -m ttt_reward_models.cli_assign_sdxl_sft \
  --prompt "a cinematic portrait of a girl in soft light" \
  --target_image_path ./target.jpg \
  --model_id ./models/sdxl-turbo \
  --output_dir outputs/run_assign_noise_sft \
  --steps 40 \
  --lr 5e-4 \
  --patch_size 2 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1
```

This saves `best_input_noise.pt`, intermediate images, loss curves, and the orthogonal-Gaussian diagnostics.

## Step 2: build a matched-noise dataset from many samples

```bash
python -m ttt_reward_models.cli_build_assigned_noise_dataset \
  --data_root ./data/high_quality_pairs \
  --output_root ./outputs/assigned_noise_dataset \
  --model_id ./models/sdxl-turbo \
  --steps 40 \
  --lr 5e-4 \
  --patch_size 2 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1
```

This writes:

```text
outputs/assigned_noise_dataset/
  manifest.jsonl
  failures.jsonl
  summary.json
  samples/
    000000_xxx/
      best_input_noise.pt
      meta.json
      original_or_linked_target.png
      run_outputs/
```

## Step 3: fine-tune on matched noises instead of random noises

```bash
python -m ttt_reward_models.cli_train_latent_matched_sft \
  --manifest_path ./outputs/assigned_noise_dataset/manifest.jsonl \
  --model_id ./models/sdxl-turbo \
  --output_dir ./outputs/latent_matched_sft \
  --epochs 2 \
  --batch_size 1 \
  --lr 1e-5 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1 \
  --preserve_latent_weight 0.5 \
  --preserve_pixel_weight 0.05
```

The preserve losses are important in practice. They encourage the updated model to stay close to the base model on random-noise generations, which helps reduce catastrophic drift and keeps the model more usable off-distribution.

Trainer outputs:

```text
outputs/latent_matched_sft/
  checkpoint-best/
  checkpoint-final/
  checkpoint-epoch-001/
  train_history.png
  train_summary.json
```

The saved checkpoints are full diffusers pipelines, so you can reuse them as the next `--model_id`.

## Step 4: alternating E/M cycles

```bash
python -m ttt_reward_models.cli_em_latent_matched_sft \
  --data_root ./data/high_quality_pairs \
  --work_dir ./outputs/em_latent_matched_sft \
  --initial_model_id ./models/sdxl-turbo \
  --num_cycles 2 \
  --assign_steps 40 \
  --assign_lr 5e-4 \
  --train_epochs 1 \
  --train_batch_size 1 \
  --train_lr 1e-5 \
  --latent_loss_weight 1.0 \
  --pixel_l1_weight 0.1 \
  --preserve_latent_weight 0.5 \
  --preserve_pixel_weight 0.05
```

This implements the basic hard-EM style loop:

- assign better noises with the current model
- fine-tune on those noises
- repeat

## Practical notes

- The assigned-noise stage uses the **patch-wise orthogonal noise transform**, so the optimized noise stays tied to a Gaussian source in a structured way rather than becoming a totally unconstrained per-sample latent code.
- The trainer is intentionally conservative: it only trains the **UNet**, while keeping the VAE and text encoders frozen.
- The current clean implementation focuses on **SDXL Turbo**. Your uploaded `test_time_inference.zip` reference scripts are preserved under `legacy_reference/` for comparison, including the SD1.5 Hyper-SD variants and the older dataset-building code.
- The trainer here is meant as a stable baseline implementation of the idea, not the final word. Natural next steps would be LoRA-only tuning, EMA, distributed training, and explicit Gaussian-prior regularizers for more flexible per-sample latent parameterizations.

## Example helper scripts

```bash
bash scripts/run_assign_noise_sft_example.sh
bash scripts/run_build_assigned_dataset_example.sh
bash scripts/run_latent_matched_sft_example.sh
bash scripts/run_em_latent_matched_sft_example.sh
```
