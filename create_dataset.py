import os
import cv2
import pydicom
import numpy as np
from preprocessing.smart_cropper import SmartCropper
from config import ProjectConfig


class DatasetCreator:
    """
    End-to-end pipeline to generate the deep learning dataset.
    Reads raw DICOM files, converts them to 8-bit grayscale, applies
    laterality-aware cropping/padding, and saves model-ready PNGs.
    """

    def __init__(self, dicom_dir: str, output_dir: str):
        """
        Initializes the DatasetCreator.

        Args:
            dicom_dir (str): Path to the directory containing raw DICOM files.
            output_dir (str): Path to the directory where processed PNGs will be saved.
        """
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir
        self.cropper = SmartCropper(target_size=(224, 224))

    def convert_to_8bit(self, dicom_data: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Extracts the pixel array from a DICOM file and standardizes it to 8-bit format.
        Handles MONOCHROME1 (inverted) and MONOCHROME2 color spaces.

        Args:
            dicom_data (pydicom.dataset.FileDataset): The parsed DICOM object.

        Returns:
            np.ndarray: The 8-bit grayscale image array.
        """
        pixels = dicom_data.pixel_array.astype(float)

        # Invert colors if the image is MONOCHROME1
        if 'PhotometricInterpretation' in dicom_data and dicom_data.PhotometricInterpretation == "MONOCHROME1":
            pixels = np.amax(pixels) - pixels

        # Normalize to 0-255 range (Min-Max Scaling)
        pixels = pixels - np.min(pixels)
        if np.max(pixels) != 0:
            pixels = pixels / np.max(pixels)

        return (pixels * 255).astype(np.uint8)

    def create(self) -> None:
        """
        Executes the dataset creation process for all DICOM files in the input directory.
        """
        if not os.path.isdir(self.dicom_dir):
            print(f"Input directory not found: {self.dicom_dir}. Put your raw DICOM files there first.")
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # DEĞİŞEN KISIM BURASI: Klasördeki her şeyi al (.dcm şartını kaldırdık)
        dicom_files = [
            f for f in os.listdir(self.dicom_dir)
            if os.path.isfile(os.path.join(self.dicom_dir, f))
        ]

        if not dicom_files:
            print(f"No files found in {self.dicom_dir}. Add raw DICOM files and run again.")
            return

        print(f"Starting dataset creation... Processing {len(dicom_files)} files.")

        for dcm_name in dicom_files:
            dcm_path = os.path.join(self.dicom_dir, dcm_name)

            # Uzantısı yoksa bile .png olarak kaydetmek için isimlendirme:
            png_name = f"{os.path.splitext(dcm_name)[0]}.png"
            out_path = os.path.join(self.output_dir, png_name)

            try:
                # 1. Read DICOM from disk
                dicom_data = pydicom.dcmread(dcm_path)

                # 2. Convert to 8-bit in memory
                image_8bit = self.convert_to_8bit(dicom_data)

                # 3. Apply smart cropping and padding in memory
                processed_image = self.cropper.process(image_8bit)

                # 4. Save the final model-ready image to disk
                cv2.imwrite(out_path, processed_image)
                print(f"Successfully processed: {dcm_name} -> {png_name}")

            except Exception as e:
                # DICOM olmayan bir dosya (örn: .DS_Store) varsa buraya düşer, kod çökmez
                print(f"Skipped {dcm_name}: {str(e)}")

        print(f"\nDataset creation complete! Model-ready images saved to: {self.output_dir}")


if __name__ == "__main__":
    cfg = ProjectConfig()
    INPUT_DICOM_DIR = cfg.raw_data_path
    OUTPUT_PNG_DIR = cfg.processed_data_path

    creator = DatasetCreator(dicom_dir=INPUT_DICOM_DIR, output_dir=OUTPUT_PNG_DIR)
    creator.create()