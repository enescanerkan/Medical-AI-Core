from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    filename: str
    class_name: str
    split: str


class LabelPolicy:
    def resolve_class(self, class_folder_name: str) -> str:
        raise NotImplementedError


class FolderLabelPolicy(LabelPolicy):
    """Maps folder name directly to class label (CC or MLO)."""

    def __init__(self, allowed_classes: Iterable[str] = ("CC", "MLO")) -> None:
        self.allowed_classes = {name.upper() for name in allowed_classes}

    def resolve_class(self, class_folder_name: str) -> str:
        label = class_folder_name.upper()
        if label not in self.allowed_classes:
            raise ValueError(f"Unsupported class folder: {class_folder_name}")
        return label


class DatasetIndexer:
    """Indexes `data/yolo-data/{CC,MLO}/images/{train,val,test}` into image records."""

    def __init__(self, source_root: Path, label_policy: LabelPolicy) -> None:
        self.source_root = source_root
        self.label_policy = label_policy

    def collect(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []

        for class_dir in sorted(self.source_root.iterdir()):
            if not class_dir.is_dir():
                continue

            try:
                class_name = self.label_policy.resolve_class(class_dir.name)
            except ValueError:
                continue

            images_root = class_dir / "images"
            if not images_root.exists():
                continue

            for split_dir in sorted(images_root.iterdir()):
                if not split_dir.is_dir() or split_dir.name.lower() not in VALID_SPLITS:
                    continue

                split = split_dir.name.lower()
                for image_path in sorted(split_dir.iterdir()):
                    if image_path.suffix.lower() not in VALID_EXTENSIONS:
                        continue

                    records.append(
                        ImageRecord(
                            image_path=image_path,
                            filename=image_path.name,
                            class_name=class_name,
                            split=split,
                        )
                    )

        if not records:
            raise RuntimeError(
                f"No images found under expected structure: {self.source_root}/{{CC,MLO}}/images/{{train,val,test}}"
            )

        return records

