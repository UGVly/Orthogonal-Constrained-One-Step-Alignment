from __future__ import annotations

import json
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CIFAR10_CLASSES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

_CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CIFAR10_ARCHIVE = 'cifar-10-python.tar.gz'
_CIFAR10_EXTRACTED = 'cifar-10-batches-py'


def _download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def ensure_cifar10_files(root: str) -> Path:
    data_dir = Path(root) / 'cifar10_raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / _CIFAR10_ARCHIVE
    extracted_dir = data_dir / _CIFAR10_EXTRACTED
    if not archive_path.exists():
        _download_url(_CIFAR10_URL, archive_path)
    if not extracted_dir.exists():
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
    return extracted_dir


def _load_batch(batch_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(batch_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data']
    labels = data[b'labels']
    images = np.asarray(images, dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(labels, dtype=np.int64)
    tensor = torch.from_numpy(images).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor, torch.from_numpy(labels).long()


class CIFAR10TargetDataset(Dataset):
    def __init__(self, root: str = './data', train: bool = True, limit: Optional[int] = None):
        base_dir = ensure_cifar10_files(root)
        if train:
            image_list = []
            label_list = []
            for idx in range(1, 6):
                imgs, labels = _load_batch(base_dir / f'data_batch_{idx}')
                image_list.append(imgs)
                label_list.append(labels)
            images = torch.cat(image_list, dim=0)
            labels = torch.cat(label_list, dim=0)
        else:
            images, labels = _load_batch(base_dir / 'test_batch')
        if limit is not None:
            images = images[:limit]
            labels = labels[:limit]
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        label = int(self.labels[index].item())
        return {
            'image': self.images[index].clone(),
            'label': label,
            'prompt': CIFAR10_CLASSES[label],
            'source_index': index,
        }


class AssignedCIFAR10Dataset(Dataset):
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        self.items: List[Dict[str, object]] = []
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        if not self.items:
            raise ValueError(f'No assigned samples found in {manifest_path}')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        image = load_image_tensor(item['target_image_path'])
        noise = torch.load(item['noise_path'], map_location='cpu').float()
        if noise.ndim == 2 and noise.shape[0] == 1:
            noise = noise[0]
        return {
            'image': image,
            'label': int(item['label']),
            'noise': noise,
            'prompt': item.get('prompt', CIFAR10_CLASSES[int(item['label'])]),
            'source_index': int(item.get('source_index', index)),
        }


def assigned_cifar_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([x['image'] for x in batch], dim=0)
    labels = torch.tensor([int(x['label']) for x in batch], dtype=torch.long)
    noises = torch.stack([x['noise'] for x in batch], dim=0)
    prompts = [str(x['prompt']) for x in batch]
    source_indices = [int(x['source_index']) for x in batch]
    return {
        'image': images,
        'label': labels,
        'noise': noises,
        'prompt': prompts,
        'source_index': source_indices,
    }


def load_image_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert('RGB').resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f'Expected single image or CHW tensor, got shape={tuple(t.shape)}')
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f'Expected CHW tensor, got shape={tuple(t.shape)}')
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] > 3:
        t = t[:3]
    t = ((t + 1.0) * 0.5).clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype('uint8')
    Image.fromarray(arr).save(path)
