# dicom_processor.py

import os
from data.dicom_handler import DicomManager
from config import ProjectConfig

def main():
    cfg = ProjectConfig()

    raw_dir = os.path.join(cfg.dataset_path, "raw_dicom")
    processed_dir = os.path.join(cfg.dataset_path, "processed")

    os.makedirs(processed_dir, exist_ok=True)

    dicom_files = [f for f in os.listdir(raw_dir) if f.endswith(".dicom") or f.endswith(".dcm")]

    print(f"Found {len(dicom_files)} DICOM files. Starting processing...\n")

    for i, file_name in enumerate(dicom_files):
        raw_path = os.path.join(raw_dir, file_name)

        if i == 0:
            DicomManager.print_all_raw_metadata(raw_path)

        # 1. Read the metadata and write
        DicomManager.print_dicom_metadata(raw_path)

        # 2. Convert Png and save processed data
        new_name = file_name.replace(".dicom", ".png").replace(".dcm", ".png")
        output_path = os.path.join(processed_dir, new_name)
        DicomManager.convert_dicom_to_png(raw_path, output_path)

    print("\nAll files have been successfully processed and saved to the 'processed' folder!")

if __name__ == "__main__":
    main()