from __future__ import annotations

import csv
import importlib
import shutil
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Yolo26Config
from .dataset import DatasetIndexer, FolderLabelPolicy, ImageRecord


class UltralyticsTrainer:
    """Wrapper to keep ultralytics dependency isolated from data prep."""

    def __init__(self, config: Yolo26Config) -> None:
        self.config = config

    def train_and_validate(self, data_dir: Path, run_name: str) -> dict:
        yolo_module = importlib.import_module("ultralytics")
        model = yolo_module.YOLO(self.config.model_name)
        requested_device, effective_device, fallback_reason = self._resolve_device(self.config.device)

        if fallback_reason:
            print(f"[device] {fallback_reason} Falling back to '{effective_device}'.")

        train_results = model.train(
            task="classify",
            data=str(data_dir),
            epochs=self.config.epochs,
            patience=self.config.early_stopping_patience,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            workers=self.config.workers,
            device=effective_device,
            seed=self.config.seed,
            project=str(self.config.output_root / "runs"),
            name=run_name,
            exist_ok=True,
        )
        val_results = model.val(task="classify", data=str(data_dir), split="test")

        run_dir = Path(str(train_results.save_dir))
        completed_epochs = self._read_completed_epochs(run_dir)
        best_weights = run_dir / "weights" / "best.pt"

        return {
            "run_dir": str(run_dir),
            "best_weights": str(best_weights),
            "requested_device": requested_device,
            "effective_device": effective_device,
            "completed_epochs": completed_epochs,
            "stopped_early": completed_epochs < self.config.epochs,
            "val_top1": getattr(val_results, "top1", None),
            "val_top5": getattr(val_results, "top5", None),
        }

    @staticmethod
    def _read_completed_epochs(run_dir: Path) -> int:
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            return 0

        with results_csv.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)

        return max(0, line_count - 1)

    @staticmethod
    def _resolve_device(requested: str) -> tuple[str, str, str | None]:
        torch_module = importlib.import_module("torch")

        normalized = (requested or "auto").strip().lower()
        cuda_available = bool(torch_module.cuda.is_available())

        if normalized == "auto":
            if cuda_available:
                return requested, "0", None
            return requested, "cpu", "CUDA is not available on this machine."

        if normalized == "cpu":
            return requested, "cpu", None

        if not cuda_available:
            return requested, "cpu", "Requested CUDA device, but CUDA is not available."

        return requested, requested, None


class ConfusionMatrixEvaluator:
    def __init__(self, config: Yolo26Config) -> None:
        self.config = config

    def evaluate(self, weights_path: Path, data_dir: Path, output_path: Path) -> dict:
        yolo_module = importlib.import_module("ultralytics")
        model = yolo_module.YOLO(str(weights_path))

        test_root = data_dir / "test"
        if not test_root.exists():
            raise RuntimeError(f"Test split not found: {test_root}")

        class_names = self._resolve_class_names(model, test_root)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

        for class_name in class_names:
            class_dir = test_root / class_name
            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue

                results = model.predict(source=str(image_path), verbose=False)
                if not results:
                    continue

                probs = getattr(results[0], "probs", None)
                if probs is None:
                    continue

                pred_idx = int(probs.top1)
                true_idx = class_to_idx[class_name]
                matrix[true_idx, pred_idx] += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._plot_matrix(matrix, class_names, output_path)

        total = int(matrix.sum())
        correct = int(np.trace(matrix))
        accuracy = (correct / total) if total else 0.0

        return {
            "confusion_matrix_path": str(output_path),
            "test_samples": total,
            "test_accuracy": round(accuracy, 4),
        }

    @staticmethod
    def _resolve_class_names(model: object, test_root: Path) -> list[str]:
        names = getattr(model, "names", None)

        if isinstance(names, dict):
            return [names[idx] for idx in sorted(names)]
        if isinstance(names, list):
            return names

        return sorted([entry.name for entry in test_root.iterdir() if entry.is_dir()])

    @staticmethod
    def _plot_matrix(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Test Confusion Matrix",
        )

        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                ax.text(
                    j,
                    i,
                    f"{value}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


class ClassificationDatasetBuilder:
    def __init__(self, config: Yolo26Config) -> None:
        self.config = config

    def build(self, records: list[ImageRecord]) -> Path:
        data_dir = self.config.prepared_data_dir
        if data_dir.exists():
            shutil.rmtree(data_dir)

        for record in records:
            class_dir = data_dir / record.split / record.class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            target_path = class_dir / record.filename
            self._materialize_image(record.image_path, target_path)

        self._write_manifest(records)
        return data_dir

    def _materialize_image(self, source: Path, target: Path) -> None:
        if self.config.copy_mode == "copy":
            shutil.copy2(source, target)
            return

        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)

    def _write_manifest(self, records: list[ImageRecord]) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        with self.config.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "class", "split", "source_path"])
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "filename": record.filename,
                        "class": record.class_name,
                        "split": record.split,
                        "source_path": str(record.image_path),
                    }
                )


class Yolo26ClassificationPipeline:
    def __init__(self, config: Yolo26Config) -> None:
        self.config = config
        self.indexer = DatasetIndexer(config.source_root, FolderLabelPolicy())
        self.builder = ClassificationDatasetBuilder(config)
        self.trainer = UltralyticsTrainer(config)
        self.evaluator = ConfusionMatrixEvaluator(config)

    def prepare(self) -> dict:
        records = self.indexer.collect()
        prepared_dir = self.builder.build(records)

        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        class_counts: dict[str, int] = {"CC": 0, "MLO": 0}
        for record in records:
            split_counts[record.split] += 1
            class_counts[record.class_name] += 1

        return {
            "config": asdict(self.config),
            "prepared_data_dir": str(prepared_dir),
            "manifest_csv": str(self.config.manifest_csv),
            "total_images": len(records),
            "split_counts": split_counts,
            "class_counts": class_counts,
        }

    def train(self, run_name: str = "yolo26_cc_mlo") -> dict:
        prepared_dir = self.config.prepared_data_dir
        if not prepared_dir.exists():
            raise RuntimeError("Prepared dataset not found. Run prepare() first.")

        train_report = self.trainer.train_and_validate(prepared_dir, run_name=run_name)

        if not self.config.save_confusion_matrix:
            return train_report

        cm_path = Path(train_report["run_dir"]) / "confusion_matrix_test.png"
        eval_report = self.evaluator.evaluate(
            weights_path=Path(train_report["best_weights"]),
            data_dir=prepared_dir,
            output_path=cm_path,
        )

        return {**train_report, **eval_report}
