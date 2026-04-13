# Medical-AI-Core

This repository currently includes two main workflows for mammography imaging:

1. DICOM -> preprocessing -> standard classification dataset (`dataset/processed`)
2. YOLO classification -> a single model that predicts imaging view (`CC` vs `MLO`)

## Current Data Layout

- `data/raw/`: raw DICOM inputs
- `data/raw_test/`: optional separate DICOM test pool
- `data/yolo-data/CC/images/{train,val,test}`: CC view images
- `data/yolo-data/MLO/images/{train,val,test}`: MLO view images
- `dataset/yolo26_cls/`: generated workspace for YOLO classification

## YOLO26-m CC/MLO Classification Pipeline

This pipeline does not use detection label files (`labels/*.txt`).
Class labels are inferred from folder names only:

- `CC` -> class `CC`
- `MLO` -> class `MLO`

### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare Dataset (CSV + Classification Structure)

```bash
python scripts/train_yolo26_cls.py --prepare-only
```

Outputs:

- `dataset/yolo26_cls/labels.csv` (`filename`, `class`, `split`, `source_path`)
- `dataset/yolo26_cls/prepared/{train,val,test}/{CC,MLO}/*.png`

Note: The default mode is `hardlink`, so files are linked without full duplication when supported by the filesystem. Use `--copy-mode copy` if you need physical copies.

### 3) Start Training

```bash
python scripts/train_yolo26_cls.py --model yolo26m-cls.pt --epochs 30 --imgsz 640 --batch 32 --device auto
```

Device behavior:

- `--device auto`: uses CUDA `0` if available, otherwise falls back to CPU.
- `--device 0`: requests CUDA `0`; if CUDA is unavailable, the script falls back to CPU with a warning.
- `--device cpu`: forces CPU execution.

Use early stopping with `--patience`:

```bash
python scripts/train_yolo26_cls.py --model yolo26m-cls.pt --epochs 50 --patience 8 --device auto
```

After training, a test confusion matrix is generated automatically:

- `dataset/yolo26_cls/runs/<run_name>/confusion_matrix_test.png`

Disable confusion matrix generation if required:

```bash
python scripts/train_yolo26_cls.py --no-confusion-matrix
```

### 4) Custom Run Name

```bash
python scripts/train_yolo26_cls.py --run-name yolo26_cc_mlo_v1
```

## Data Versioning and Git Policy

Large medical datasets and generated training artifacts are intentionally excluded from Git.

Ignored by default:

- Raw DICOM datasets under `data/raw/` and `data/raw_test/`
- External YOLO source data under `data/yolo-data/`
- Generated images and run outputs under `dataset/processed/` and `dataset/yolo26_cls/runs/`
- Model weight artifacts such as `*.pt`, `*.pth`, `*.onnx`, and `*.engine`

Recommended workflow:

1. Keep only code, configs, and documentation in the repository.
2. Recreate prepared datasets locally with `--prepare-only`.
3. Keep trained weights and run artifacts in local or remote artifact storage, not in Git.

## Design Notes (OOP/SOLID)

- `training/yolo26_cls/dataset.py`: data indexing and label policy abstraction
- `training/yolo26_cls/pipeline.py`: orchestration layer for preparation, training, and evaluation
- `training/yolo26_cls/config.py`: centralized configuration boundary
- `scripts/train_yolo26_cls.py`: CLI entry point and runtime wiring

This separation keeps responsibilities clear and supports maintainable, testable evolution of the pipeline.
