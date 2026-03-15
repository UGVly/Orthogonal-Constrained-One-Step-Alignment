#!/usr/bin/env bash
set -euo pipefail

# Download all model assets into the project-local ./models directory.
# Default layout:
#   models/
#     sdxl-turbo/
#     Hyper-SD15-1step/
#     PickScore_v1/
#     ImageReward/
#     HPSv2/
#     CLIP-ViT-L-14/
#     Aesthetic/
#       sac+logos+ava1-l14-linearMSE.pth
#
# Usage:
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh --only sdxl-turbo
#   bash scripts/download_models.sh --only PickScore_v1 --only ImageReward
#   bash scripts/download_models.sh --only CLIP-ViT-L-14 --only Aesthetic
#   bash scripts/download_models.sh --hps-version v2.0
#   bash scripts/download_models.sh --skip-hyper-sd --skip-clip --skip-aesthetic

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

HPS_VERSION="v2.1"
ONLY_LIST=()
SKIP_SDXL_TURBO=0
SKIP_HYPER_SD=0
SKIP_PICKSCORE=0
SKIP_IMAGEREWARD=0
SKIP_HPSV2=0
SKIP_CLIP=0
SKIP_AESTHETIC=0

AESTHETIC_VARIANT="sac+logos+ava1-l14-linearMSE"

usage() {
  cat <<EOF
Download model assets into: $MODELS_DIR

Options:
  --only NAME           Only download the given model. Can be passed multiple times.
                        Supported names:
                          sdxl-turbo
                          Hyper-SD15-1step
                          PickScore_v1
                          ImageReward
                          HPSv2
                          CLIP-ViT-L-14
                          Aesthetic
  --hps-version VER     HPS version: v2.1 (default) or v2.0
  --aesthetic-variant   Aesthetic checkpoint variant:
                          sac+logos+ava1-l14-linearMSE (default)
                          ava+logos-l14-linearMSE
  --skip-sdxl-turbo     Skip SDXL-Turbo
  --skip-hyper-sd       Skip Hyper-SD15-1step
  --skip-pickscore      Skip PickScore_v1
  --skip-imagereward    Skip ImageReward
  --skip-hpsv2          Skip HPSv2
  --skip-clip           Skip CLIP ViT-L/14
  --skip-aesthetic      Skip Aesthetic predictor
  -h, --help            Show this help

Env:
  HF_TOKEN                Optional Hugging Face token for gated/rate-limited downloads
  MODELSCOPE_API_TOKEN    Optional ModelScope token
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY_LIST+=("$2")
      shift 2
      ;;
    --hps-version)
      HPS_VERSION="$2"
      shift 2
      ;;
    --aesthetic-variant)
      AESTHETIC_VARIANT="$2"
      shift 2
      ;;
    --skip-sdxl-turbo)
      SKIP_SDXL_TURBO=1
      shift
      ;;
    --skip-hyper-sd)
      SKIP_HYPER_SD=1
      shift
      ;;
    --skip-pickscore)
      SKIP_PICKSCORE=1
      shift
      ;;
    --skip-imagereward)
      SKIP_IMAGEREWARD=1
      shift
      ;;
    --skip-hpsv2)
      SKIP_HPSV2=1
      shift
      ;;
    --skip-clip)
      SKIP_CLIP=1
      shift
      ;;
    --skip-aesthetic)
      SKIP_AESTHETIC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$HPS_VERSION" != "v2.1" && "$HPS_VERSION" != "v2.0" ]]; then
  echo "[ERROR] --hps-version must be v2.1 or v2.0" >&2
  exit 1
fi

if [[ "$AESTHETIC_VARIANT" != "sac+logos+ava1-l14-linearMSE" && "$AESTHETIC_VARIANT" != "ava+logos-l14-linearMSE" ]]; then
  echo "[ERROR] --aesthetic-variant must be one of:" >&2
  echo "        sac+logos+ava1-l14-linearMSE" >&2
  echo "        ava+logos-l14-linearMSE" >&2
  exit 1
fi

mkdir -p "$MODELS_DIR"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    return 1
  fi
}

need_any_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    return 0
  fi
  echo "[ERROR] Need 'hf' or 'huggingface-cli' in PATH." >&2
  echo "        Install with: pip install -U 'huggingface_hub[cli]'" >&2
  return 1
}

need_any_downloader() {
  if command -v curl >/dev/null 2>&1; then
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    return 0
  fi
  echo "[ERROR] Need 'curl' or 'wget' in PATH." >&2
  return 1
}

hf_download_repo() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  mkdir -p "$local_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo_id" --local-dir "$local_dir" "$@"
  else
    huggingface-cli download "$repo_id" --local-dir "$local_dir" "$@"
  fi
}

ms_download_model() {
  local model_id="$1"
  local local_dir="$2"
  mkdir -p "$local_dir"
  need_cmd modelscope
  modelscope download --model "$model_id" --local_dir "$local_dir"
}

download_url() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

should_download() {
  local target="$1"
  if [[ ${#ONLY_LIST[@]} -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "${ONLY_LIST[@]}"; do
    if [[ "$item" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${HF_TOKEN:-}" ]]; then
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  fi
fi

if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
  mkdir -p "$HOME/.modelscope"
  printf '%s' "$MODELSCOPE_API_TOKEN" > "$HOME/.modelscope/api_token"
fi

# 1) SDXL-Turbo (prefer ModelScope)
if [[ $SKIP_SDXL_TURBO -eq 0 ]] && should_download "sdxl-turbo"; then
  log "Downloading sdxl-turbo -> $MODELS_DIR/sdxl-turbo"
  ms_download_model "AI-ModelScope/sdxl-turbo" "$MODELS_DIR/sdxl-turbo"
fi

# 2) Hyper-SD 1-step for SD1.5 (official HF file)
if [[ $SKIP_HYPER_SD -eq 0 ]] && should_download "Hyper-SD15-1step"; then
  log "Downloading Hyper-SD15-1step -> $MODELS_DIR/Hyper-SD15-1step"
  need_any_hf_cli
  hf_download_repo "ByteDance/Hyper-SD" "$MODELS_DIR/Hyper-SD15-1step" \
    --include "Hyper-SD15-1step-lora.safetensors" \
    --include "README.md" \
    --include "comfyui/Hyper-SD15-1step-unified-lora-workflow.json"
fi

# 3) PickScore v1 (full snapshot for local from_pretrained)
if [[ $SKIP_PICKSCORE -eq 0 ]] && should_download "PickScore_v1"; then
  log "Downloading PickScore_v1 -> $MODELS_DIR/PickScore_v1"
  need_any_hf_cli
  hf_download_repo "yuvalkirstain/PickScore_v1" "$MODELS_DIR/PickScore_v1"
fi

# 4) ImageReward (prefer ModelScope)
if [[ $SKIP_IMAGEREWARD -eq 0 ]] && should_download "ImageReward"; then
  log "Downloading ImageReward -> $MODELS_DIR/ImageReward"
  ms_download_model "ZhipuAI/ImageReward" "$MODELS_DIR/ImageReward"
fi

# 5) HPSv2 (official HF checkpoint)
if [[ $SKIP_HPSV2 -eq 0 ]] && should_download "HPSv2"; then
  log "Downloading HPSv2 ($HPS_VERSION) -> $MODELS_DIR/HPSv2"
  need_any_hf_cli
  HPS_FILE="HPS_v2.1_compressed.pt"
  if [[ "$HPS_VERSION" == "v2.0" ]]; then
    HPS_FILE="HPS_v2_compressed.pt"
  fi
  hf_download_repo "xswu/HPSv2" "$MODELS_DIR/HPSv2" --include "$HPS_FILE"
fi

# 6) CLIP ViT-L/14
if [[ $SKIP_CLIP -eq 0 ]] && should_download "CLIP-ViT-L-14"; then
  log "Downloading CLIP ViT-L/14 -> $MODELS_DIR/CLIP-ViT-L-14"
  need_any_hf_cli
  hf_download_repo "openai/clip-vit-large-patch14" "$MODELS_DIR/CLIP-ViT-L-14"
fi

# 7) Aesthetic predictor checkpoint
if [[ $SKIP_AESTHETIC -eq 0 ]] && should_download "Aesthetic"; then
  log "Downloading Aesthetic predictor ($AESTHETIC_VARIANT) -> $MODELS_DIR/Aesthetic"
  need_any_downloader
  AESTHETIC_FILE="${AESTHETIC_VARIANT}.pth"
  AESTHETIC_URL="https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/${AESTHETIC_FILE}"
  mkdir -p "$MODELS_DIR/Aesthetic"
  download_url "$AESTHETIC_URL" "$MODELS_DIR/Aesthetic/$AESTHETIC_FILE"
fi

cat <<EOF

Done.

Downloaded into:
  $MODELS_DIR

Expected local paths:
  $MODELS_DIR/sdxl-turbo
  $MODELS_DIR/Hyper-SD15-1step/Hyper-SD15-1step-lora.safetensors
  $MODELS_DIR/PickScore_v1
  $MODELS_DIR/ImageReward
  $MODELS_DIR/HPSv2
  $MODELS_DIR/CLIP-ViT-L-14
  $MODELS_DIR/Aesthetic/${AESTHETIC_VARIANT}.pth

Suggested runtime args:
  --model_id $MODELS_DIR/sdxl-turbo
  --pickscore_model_id $MODELS_DIR/PickScore_v1
  --imagereward_model_path $MODELS_DIR/ImageReward/ImageReward.pt
  --imagereward_med_config_path $MODELS_DIR/ImageReward/med_config.json
  --hps_checkpoint_path $MODELS_DIR/HPSv2/${HPS_FILE:-HPS_v2.1_compressed.pt}
  --clip_local_dir $MODELS_DIR/CLIP-ViT-L-14
  --aesthetic_ckpt $MODELS_DIR/Aesthetic/${AESTHETIC_VARIANT}.pth
EOF