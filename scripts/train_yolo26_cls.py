from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.yolo26_cls.config import Yolo26Config
from training.yolo26_cls.pipeline import Yolo26ClassificationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26-m classifier for CC vs MLO")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data") / "yolo-data",
        help="Root that contains CC/ and MLO/ folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset") / "yolo26_cls",
        help="Output root for prepared dataset, manifest, and runs",
    )
    parser.add_argument("--model", type=str, default="yolo26m-cls.pt", help="Ultralytics model name/path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, 0, 0,1",
    )
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--run-name", type=str, default="yolo26_cc_mlo")
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_true",
        help="Skip confusion matrix generation after training",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only build dataset and labels.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Yolo26Config(
        source_root=args.source_root,
        output_root=args.output_root,
        model_name=args.model,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        image_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        device=args.device,
        copy_mode=args.copy_mode,
        save_confusion_matrix=not args.no_confusion_matrix,
    )

    pipeline = Yolo26ClassificationPipeline(config)
    prep_info = pipeline.prepare()
    print("[prepare] Dataset is ready")
    print(json.dumps(prep_info, indent=2, default=str))

    if args.prepare_only:
        return

    train_info = pipeline.train(run_name=args.run_name)
    print("[train] Training finished")
    print(json.dumps(train_info, indent=2, default=str))


if __name__ == "__main__":
    main()
