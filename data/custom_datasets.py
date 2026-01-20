### adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py

import torchvision
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional, Union


def pil_loader(path: Union[str, Path]) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def default_loader(path: Union[str, Path]) -> Any:
    return pil_loader(path)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class DistillImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize_transform: Optional[Callable] = None,
        distill_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.distill_transform = distill_transform
        self.normalize_transform = normalize_transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        distill_sample = self.distill_transform(sample)
        if self.normalize_transform is not None:
            sample = self.normalize_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, distill_sample, target
    