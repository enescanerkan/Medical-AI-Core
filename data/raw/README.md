# Raw DICOM Drop Folder

This folder is for local use.

- Place your raw `.dcm` / DICOM files here.
- This folder is not added to Git; it remains only on your local computer.
- The processing step reads the files located here using `create_dataset.py` and saves them as PNGs under `dataset/processed/`.

## Recommended workflow

1. Copy raw DICOM files here.
2. Run `python create_dataset.py`.
3. Check the generated PNGs under `dataset/processed/`.
4. Prepare `dataset/labels.csv` for training/evaluation.