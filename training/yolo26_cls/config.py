from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Yolo26Config:
    """Configuration for building and training the CC/MLO classification dataset."""

    source_root: Path
    output_root: Path
    model_name: str = "yolo26m-cls.pt"
    epochs: int = 30
    early_stopping_patience: int = 8
    image_size: int = 640
    batch_size: int = 32
    workers: int = 4
    device: str = "cpu"
    seed: int = 42
    copy_mode: str = "hardlink"  # hardlink | copy
    save_confusion_matrix: bool = True

    @property
    def prepared_data_dir(self) -> Path:
        return self.output_root / "prepared"

    @property
    def manifest_csv(self) -> Path:
        return self.output_root / "labels.csv"
