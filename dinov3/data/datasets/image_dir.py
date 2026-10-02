import os
from typing import Callable, Optional

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageDir(ExtendedVisionDataset):
    """Unlabeled images in a directory, including nested folders."""

    def __init__(
        self,
        *,
        root: str,
        extra: Optional[str] = None,
        split: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        del extra, split
        paths = []
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS:
                    paths.append(os.path.join(dirpath, name))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No images found under {root}")
        self._paths = paths

    def get_image_data(self, index: int) -> bytes:
        with open(self._paths[index], "rb") as handle:
            return handle.read()

    def get_target(self, index: int) -> int:
        return 0

    def __len__(self) -> int:
        return len(self._paths)
