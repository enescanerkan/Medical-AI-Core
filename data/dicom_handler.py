# dicom_handler.py
"""
Reads DICOM metadata and converts 16-bit medical pixels to standard formats.
"""

import os
import pydicom
import numpy as np
import cv2
from mpmath.libmp import normalize


class DicomManager:
    """
    Utility class for reading and converting DICOM files to standard formats.
    """

    @staticmethod
    def print_dicom_metadata(dicom_path: str):
        """
        Reads a DICOM file and prints crucial medical metadata.
        """
        if not os.path.exists(dicom_path):
            print(f"Dicom file {dicom_path} not found.")
            return

        try:
            # Read the DICOM file
            dcm = pydicom.dcmread(dicom_path)

            print(f"\n--- Metadata for {os.path.basename(dicom_path)} ---\n")
            # Some DICOM files might not have all tags.
            # We use getattr() to safely fetch data without crashing if a tag is missing.
            print(f"Patient ID: {getattr(dcm, 'PatientID', 'Unknown')}")
            print(f"Patient Age: {getattr(dcm, 'PatientAge', 'Unknown')}")
            print(f"Patient Sex: {getattr(dcm, 'PatientSex', 'Unknown')}")
            print(f"Modality (Device): {getattr(dcm, 'Modality', 'Unknown')} (e.g., MG=Mammogram, CT=Scan)")
            print(f"Body Part Examined: {getattr(dcm, 'BodyPartExamined', 'Unknown')}")
            print(f"Image Dimensions: {dcm.Rows} x {dcm.Columns}")
            print("-" * 40)

        except Exception as e:
            print(f"Failed to read DICOM metadata: {e}")

    @staticmethod
    def print_all_raw_metadata(dicom_path: str):
        """
        Prints the entire, raw DICOM dataset (all tags and values) to the terminal.

        Args:
            dicom_path (str): The path to the DICOM file.
        """
        try:
            dcm = pydicom.dcmread(dicom_path)
            print(f"\n========== FULL RAW DICOM DATA FOR {os.path.basename(dicom_path)} ==========\n")
            print(dcm)
            print("\n==================================================================================\n")
        except Exception as e:
            print(f"Failed to read DICOM metadata: {e}")

    @staticmethod
    def convert_dicom_to_png(dicom_path: str, output_path: str):
        """
        Converts 16-bit DICOM pixel array to an 8-bit PNG image using Min-Max Normalization.
        Automatically inverts MONOCHROME1 images so all backgrounds are black.
        """
        try:
            dcm = pydicom.dcmread(dicom_path)
            pixel_array = dcm.pixel_array

            # IMPORTANT CONCEPT: Handle Photometric Interpretation (Inversion)
            # TODO: Bunu düzelt ; Eğer cihaz 'MONOCHROME1' kaydettiyse, matrisi tersine çeviriyoruz (Negatif alıyoruz).
            # Böylece tüm resimler standart 'MONOCHROME2' (Siyah arka plan) mantığına döner.
            photo_interp = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
            if photo_interp == 'MONOCHROME1':
                # Matrisi ters çevir: (Maksimum Değer - Mevcut Değer)
                pixel_array = np.max(pixel_array) - pixel_array

            # 16-bit to 8-bit Conversion (Windowing/Normalization)
            # Matrix to float
            pixel_array = pixel_array.astype(np.float32)

            # Min-Max : (X - Min) / (Max - Min)
            img_min = np.min(pixel_array)
            img_max = np.max(pixel_array)

            if img_max > img_min:
                normalized_img = (pixel_array - img_min) / (img_max - img_min)
            else:
                normalized_img = np.zeros_like(pixel_array)

            # 0-1 to 0-255 and 8-bit (uint8)
            img_8bit = (normalized_img * 255).astype(np.uint8)

            cv2.imwrite(output_path, img_8bit)
            # print(f"Successfully saved as PNG: {output_path}")

        except Exception as e:
            print(f"Conversion failed for {dicom_path}: {e}")