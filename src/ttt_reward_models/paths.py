from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


# Canonical asset layout used by the shell downloader.
# We still keep backward-compatible fallbacks to the older third_party_weights/ tree.


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_models_root() -> Path:
    return get_project_root() / "models"


def get_default_weights_root() -> Path:
    return get_project_root() / "third_party_weights"


def get_default_clip_root() -> Path:
    return get_models_root() / "CLIP-ViT-L-14"


def get_default_aesthetic_root() -> Path:
    return get_models_root() / "Aesthetic"


def get_default_aesthetic_ckpt(variant: str = "sac+logos+ava1-l14-linearMSE") -> Path:
    return get_default_aesthetic_root() / f"{variant}.pth"


def get_default_pickscore_root() -> Path:
    return get_models_root() / "PickScore_v1"


def get_default_imagereward_root() -> Path:
    return get_models_root() / "ImageReward"


def get_legacy_imagereward_root() -> Path:
    return get_default_weights_root() / "imagereward_modelscope"


def get_default_hpsv2_root() -> Path:
    return get_models_root() / "HPSv2"


def get_legacy_hpsv2_root() -> Path:
    return get_default_weights_root() / "hpsv2"


def _unique_existing_or_all(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    existing: List[Path] = []
    all_paths: List[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        all_paths.append(path)
        if path.exists():
            existing.append(path)
    return existing if existing else all_paths


def get_imagereward_roots(root: Optional[str | Path] = None) -> List[Path]:
    if root is not None:
        return [Path(root)]
    return _unique_existing_or_all([
        get_default_imagereward_root(),
        get_legacy_imagereward_root(),
    ])


def get_hpsv2_roots(root: Optional[str | Path] = None) -> List[Path]:
    if root is not None:
        return [Path(root)]
    return _unique_existing_or_all([
        get_default_hpsv2_root(),
        get_legacy_hpsv2_root(),
    ])
