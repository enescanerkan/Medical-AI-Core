# Raw Test DICOM Folder

This folder is used exclusively as the source for the **test split**.

* Place the **raw DICOM files** to be used for testing here.
* When `create_dataset.py` is executed, the images generated from this folder are recorded in the `dataset/labels.csv` file with the attribute `split=test`.
* **No augmentation** is applied to the data in this folder.