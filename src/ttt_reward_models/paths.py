from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_weights_root() -> Path:
    return get_project_root() / "third_party_weights"


def get_default_imagereward_root() -> Path:
    return get_default_weights_root() / "imagereward_modelscope"


def get_default_hpsv2_root() -> Path:
    return get_default_weights_root() / "hpsv2"
