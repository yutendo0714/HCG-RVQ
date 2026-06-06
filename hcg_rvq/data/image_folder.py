from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    """Recursive image folder dataset with optional random crop."""

    def __init__(
        self,
        roots: list[str],
        patch_size: int | None = None,
        training: bool = True,
        max_images: int | None = None,
        start_index: int = 0,
        return_path: bool = False,
    ) -> None:
        self.roots = [Path(root) for root in roots]
        self.patch_size = patch_size
        self.training = training
        self.return_path = return_path
        paths: list[Path] = []
        for root in self.roots:
            if root.exists():
                paths.extend(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
        self.paths = sorted(paths)
        if start_index:
            self.paths = self.paths[int(start_index):]
        if max_images is not None:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise FileNotFoundError(f"no images found under: {roots}")

    def __len__(self) -> int:
        return len(self.paths)

    def _crop(self, image: Image.Image) -> Image.Image:
        if self.patch_size is None:
            return image
        w, h = image.size
        size = self.patch_size
        if min(w, h) < size:
            scale = size / min(w, h)
            image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            w, h = image.size
        if self.training:
            left = random.randint(0, w - size)
            top = random.randint(0, h - size)
        else:
            left = max((w - size) // 2, 0)
            top = max((h - size) // 2, 0)
        return image.crop((left, top, left + size, top + size))

    def __getitem__(self, index: int) -> torch.Tensor | dict[str, torch.Tensor | str]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = self._crop(image)
            tensor = TF.to_tensor(image)
        if self.return_path:
            return {"image": tensor, "path": str(self.paths[index])}
        return tensor

