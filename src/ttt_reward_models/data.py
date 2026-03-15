import json
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_DEFAULT_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def load_target_image(image_path: str, height: int, width: int, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    image = tfm(image).unsqueeze(0).to(device)
    return image



def iter_pairs(
    pairs_jsonl: Path,
    image_field: str = 'image',
    prompt_field: str = 'prompt',
    start_index: int = 0,
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, object]]:
    yielded = 0
    with pairs_jsonl.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_index:
                continue
            if max_samples is not None and yielded >= max_samples:
                break
            item = json.loads(line)
            if image_field not in item:
                raise KeyError(f"Field '{image_field}' not found in line {line_idx}: {list(item.keys())}")
            if prompt_field not in item:
                raise KeyError(f"Field '{prompt_field}' not found in line {line_idx}: {list(item.keys())}")
            item['_line_idx'] = line_idx
            yielded += 1
            yield item



def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def copy_or_link_image(src: Path, dst: Path, link_image: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_image:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


class AssignedNoiseDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        height: int,
        width: int,
        image_field: str = 'target_image_path',
        prompt_field: str = 'prompt',
        noise_field: str = 'best_input_noise_path',
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.height = height
        self.width = width
        self.image_field = image_field
        self.prompt_field = prompt_field
        self.noise_field = noise_field
        self.records: List[Dict[str, object]] = []
        with self.manifest_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.prompt_field not in rec:
                    raise KeyError(f"Missing '{self.prompt_field}' in manifest record")
                if self.image_field not in rec:
                    raise KeyError(f"Missing '{self.image_field}' in manifest record")
                if self.noise_field not in rec:
                    raise KeyError(f"Missing '{self.noise_field}' in manifest record")
                self.records.append(rec)
        if not self.records:
            raise ValueError(f'No records found in manifest: {self.manifest_path}')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        target_image_path = str(rec[self.image_field])
        noise_path = str(rec[self.noise_field])
        prompt = str(rec[self.prompt_field])
        image = load_target_image(target_image_path, self.height, self.width, device='cpu').squeeze(0)
        noise = torch.load(noise_path, map_location='cpu').float()
        if noise.ndim == 4 and noise.shape[0] == 1:
            noise = noise.squeeze(0)
        item: Dict[str, object] = {
            'prompt': prompt,
            'image': image,
            'matched_noise': noise,
            'target_image_path': target_image_path,
            'best_input_noise_path': noise_path,
        }
        for key, value in rec.items():
            if key not in item:
                item[key] = value
        return item



def collate_assigned_noise(batch: List[Dict[str, object]]) -> Dict[str, object]:
    prompts = [str(x['prompt']) for x in batch]
    images = torch.stack([x['image'] for x in batch], dim=0)
    noises = torch.stack([x['matched_noise'] for x in batch], dim=0)
    out: Dict[str, object] = {
        'prompts': prompts,
        'images': images,
        'matched_noises': noises,
    }
    for key in batch[0].keys():
        if key in {'prompt', 'image', 'matched_noise'}:
            continue
        out[key] = [x[key] for x in batch]
    return out
