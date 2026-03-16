from __future__ import annotations

import gzip
import json
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DIGIT_TO_TEXT = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
}

_MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]
_MNIST_FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
}


def _download_url(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def ensure_mnist_files(root: str) -> Path:
    raw_dir = Path(root) / 'MNIST' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in _MNIST_FILES.values() if not (raw_dir / name).exists()]
    for filename in missing:
        last_err = None
        for base in _MNIST_MIRRORS:
            try:
                _download_url(base + filename, raw_dir / filename)
                last_err = None
                break
            except Exception as err:  # noqa: BLE001
                last_err = err
        if last_err is not None:
            raise RuntimeError(f'Failed to download {filename}: {last_err}')
    return raw_dir


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid image file magic for {path}: {magic}')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    tensor = torch.from_numpy(data).float().unsqueeze(1) / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid label file magic for {path}: {magic}')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num)
    return torch.from_numpy(data).long()


class MNISTTargetDataset(Dataset):
    def __init__(self, root: str = './data', train: bool = True, limit: Optional[int] = None):
        raw_dir = ensure_mnist_files(root)
        if train:
            images = _read_images(raw_dir / _MNIST_FILES['train_images'])
            labels = _read_labels(raw_dir / _MNIST_FILES['train_labels'])
        else:
            images = _read_images(raw_dir / _MNIST_FILES['test_images'])
            labels = _read_labels(raw_dir / _MNIST_FILES['test_labels'])
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
            'prompt': DIGIT_TO_TEXT[label],
            'source_index': index,
        }


class AssignedMNISTDataset(Dataset):
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
            'prompt': item.get('prompt', DIGIT_TO_TEXT[int(item['label'])]),
            'source_index': int(item.get('source_index', index)),
        }


def assigned_mnist_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
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
    image = Image.open(path).convert('L')
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
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
        arr = ((t[0] + 1.0) * 0.5).clamp(0, 1).numpy()
        img = Image.fromarray((arr * 255.0).round().clip(0, 255).astype('uint8'), mode='L')
    else:
        t = ((t + 1.0) * 0.5).clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype('uint8')
        img = Image.fromarray(arr)
    img.save(path)
