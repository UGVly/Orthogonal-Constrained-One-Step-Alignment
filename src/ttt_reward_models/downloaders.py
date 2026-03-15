import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import hf_hub_download

from .paths import get_default_hpsv2_root, get_default_imagereward_root


class DownloadError(RuntimeError):
    pass


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _search_one(root: Path, name: str) -> Optional[Path]:
    matches = list(root.rglob(name))
    if not matches:
        return None
    return matches[0]


def find_imagereward_assets(root: Optional[str | Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    base = Path(root) if root is not None else get_default_imagereward_root()
    ckpt = _search_one(base, 'ImageReward.pt')
    med = _search_one(base, 'med_config.json')
    return ckpt, med


def ensure_imagereward_assets(
    root: Optional[str | Path] = None,
    *,
    use_modelscope: bool = True,
) -> Tuple[Path, Path]:
    base = _ensure_dir(Path(root) if root is not None else get_default_imagereward_root())
    ckpt, med = find_imagereward_assets(base)
    if ckpt is not None and med is not None:
        return ckpt, med

    errors: list[str] = []

    if use_modelscope:
        try:
            from modelscope import snapshot_download

            snapshot_dir = snapshot_download('ZhipuAI/ImageReward', local_dir=str(base))
            snap_root = Path(snapshot_dir)
            ckpt = _search_one(snap_root, 'ImageReward.pt') or _search_one(base, 'ImageReward.pt')
            med = _search_one(snap_root, 'med_config.json') or _search_one(base, 'med_config.json')
            if ckpt is not None and med is not None:
                return ckpt, med
            errors.append('ModelScope snapshot completed but ImageReward.pt / med_config.json were not found in the downloaded files.')
        except Exception as exc:
            errors.append(f'ModelScope download failed: {exc}')

    # Fallback to the official Hugging Face files, still stored inside the project directory.
    try:
        ckpt_path = Path(
            hf_hub_download(
                repo_id='THUDM/ImageReward',
                filename='ImageReward.pt',
                local_dir=str(base),
            )
        )
        med_path = Path(
            hf_hub_download(
                repo_id='THUDM/ImageReward',
                filename='med_config.json',
                local_dir=str(base),
            )
        )
        return ckpt_path, med_path
    except Exception as exc:
        errors.append(f'Hugging Face fallback failed: {exc}')

    raise DownloadError('Unable to prepare ImageReward assets. ' + ' | '.join(errors))


_HPS_FILE_MAP = {
    'v2.0': 'HPS_v2_compressed.pt',
    'v2.1': 'HPS_v2.1_compressed.pt',
}


def ensure_hpsv2_checkpoint(
    root: Optional[str | Path] = None,
    *,
    hps_version: str = 'v2.1',
) -> Path:
    if hps_version not in _HPS_FILE_MAP:
        raise ValueError(f'Unsupported HPS version: {hps_version}')

    base = _ensure_dir(Path(root) if root is not None else get_default_hpsv2_root())
    filename = _HPS_FILE_MAP[hps_version]
    local_path = _search_one(base, filename)
    if local_path is not None:
        return local_path

    path = Path(
        hf_hub_download(
            repo_id='xswu/HPSv2',
            filename=filename,
            local_dir=str(base),
        )
    )
    return path


def stage_hps_root_env(root: Optional[str | Path] = None) -> Path:
    base = _ensure_dir(Path(root) if root is not None else get_default_hpsv2_root())
    os.environ['HPS_ROOT'] = str(base)
    return base
