import os
import csv
import argparse
import random
import cv2
import pydicom
import numpy as np
from pydicom.misc import is_dicom
from preprocessing.augmentations import MedicalDataAugmenter
from preprocessing.cv_transforms import AdvancedImageProcessor
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

    def create(self) -> list:
        """
        Executes the dataset creation process for all DICOM files in the input directory.
        """
        if not os.path.isdir(self.dicom_dir):
            print(f"Input directory not found: {self.dicom_dir}. Put your raw DICOM files there first.")
            return []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # DEĞİŞEN KISIM BURASI: Klasördeki her şeyi al (.dcm şartını kaldırdık)
        dicom_files = [
            f for f in os.listdir(self.dicom_dir)
            if os.path.isfile(os.path.join(self.dicom_dir, f))
        ]

        if not dicom_files:
            print(f"No files found in {self.dicom_dir}. Add raw DICOM files and run again.")
            return []

        print(f"Starting dataset creation... Scanning {len(dicom_files)} files.")

        processed_count = 0
        skipped_non_dicom_count = 0
        failed_count = 0
        processed_records = []

        for dcm_name in dicom_files:
            dcm_path = os.path.join(self.dicom_dir, dcm_name)

            if not is_dicom(dcm_path):
                skipped_non_dicom_count += 1
                print(f"Ignored non-DICOM file: {dcm_name}")
                continue

            # Uzantısı yoksa bile .png olarak kaydetmek için isimlendirme:
            png_name = f"{os.path.splitext(dcm_name)[0]}.png"
            out_path = os.path.join(self.output_dir, png_name)

            try:
                # 1. Read DICOM from disk
                dicom_data = pydicom.dcmread(dcm_path)

                # 2. Convert to 8-bit in memory
                image_8bit = self.convert_to_8bit(dicom_data)

                # 3. Apply smart cropping and padding in memory
                processed_image, laterality = AdvancedImageProcessor.smart_crop_and_pad(
                    image_8bit,
                    target_size=(224, 224),
                )

                # 4. Save the final model-ready image to disk
                cv2.imwrite(out_path, processed_image)
                print(f"Successfully processed: {dcm_name} -> {png_name}")
                processed_count += 1
                processed_records.append({"filename": png_name, "laterality": laterality})

            except Exception as e:
                failed_count += 1
                print(f"Skipped {dcm_name}: {str(e)}")

        print("\nDataset creation summary:")
        print(f"- Processed DICOM files : {processed_count}")
        print(f"- Ignored non-DICOM     : {skipped_non_dicom_count}")
        print(f"- Failed during process : {failed_count}")

        if processed_count == 0:
            print(
                f"No valid DICOM files were processed from {self.dicom_dir}. "
                "Put your raw `.dcm` files in this folder and run again."
            )
        else:
            print(f"Model-ready images saved to: {self.output_dir}")

        return processed_records

    @staticmethod
    def _laterality_to_label(laterality: str, default_label: int = 0) -> int:
        if laterality == "L":
            return 0
        if laterality == "R":
            return 1
        return default_label

    @staticmethod
    def _infer_laterality_from_png(image_path: str) -> str:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Unknown"

        contours = AdvancedImageProcessor.find_contours(image, threshold_val=15)
        if not contours:
            return "Unknown"

        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        img_h, img_w = image.shape
        if moments["m00"] == 0:
            center_x = img_w // 2
        else:
            center_x = int(moments["m10"] / moments["m00"])

        return "L" if center_x < (img_w // 2) else "R"

    @staticmethod
    def _split_counts(total_count: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple:
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("train/val/test ratios must sum to a positive value")

        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

        n_train = int(round(total_count * train_ratio))
        n_val = int(round(total_count * val_ratio))
        n_test = total_count - n_train - n_val

        # Keep counts non-negative and guarantee sum exactly equals total_count.
        if n_test < 0:
            n_test = 0
            n_val = total_count - n_train
        if n_val < 0:
            n_val = 0
            n_train = total_count

        while n_train + n_val + n_test < total_count:
            n_train += 1
        while n_train + n_val + n_test > total_count:
            if n_train > 0:
                n_train -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_test -= 1

        if total_count >= 3:
            # Ensure each split has at least one sample when possible.
            split_counts = {"train": n_train, "val": n_val, "test": n_test}
            for key in ["train", "val", "test"]:
                if split_counts[key] == 0:
                    donor = max(split_counts, key=split_counts.get)
                    if split_counts[donor] > 1:
                        split_counts[donor] -= 1
                        split_counts[key] += 1
            n_train, n_val, n_test = split_counts["train"], split_counts["val"], split_counts["test"]

        return n_train, n_val, n_test

    @staticmethod
    def generate_labels_csv(
        img_dir: str,
        labels_csv_path: str,
        overwrite: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        default_label: int = 0,
    ) -> None:
        """
        Generates labels.csv from processed images.
        - label: inferred from laterality (L->0, R->1), fallback to default_label
        - split: deterministic shuffled train/val/test split
        """
        if not os.path.isdir(img_dir):
            print(f"Labels generation skipped: processed directory not found at {img_dir}")
            return

        if os.path.exists(labels_csv_path) and not overwrite:
            print(
                f"Labels generation skipped: {labels_csv_path} already exists. "
                "Use --overwrite-labels if you want to recreate it."
            )
            return

        image_files = sorted(
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f))
            and f.lower().endswith(".png")
            and "_aug_" not in f
        )

        if not image_files:
            print(f"Labels generation skipped: no base PNG files found in {img_dir}")
            return

        rng = random.Random(seed)
        shuffled_files = image_files[:]
        rng.shuffle(shuffled_files)

        n_train, n_val, n_test = DatasetCreator._split_counts(
            len(shuffled_files),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        split_lookup = {}
        for file_name in shuffled_files[:n_train]:
            split_lookup[file_name] = "train"
        for file_name in shuffled_files[n_train:n_train + n_val]:
            split_lookup[file_name] = "val"
        for file_name in shuffled_files[n_train + n_val:]:
            split_lookup[file_name] = "test"

        unknown_count = 0
        rows = []
        for file_name in image_files:
            image_path = os.path.join(img_dir, file_name)
            laterality = DatasetCreator._infer_laterality_from_png(image_path)
            if laterality not in {"L", "R"}:
                unknown_count += 1
            label = DatasetCreator._laterality_to_label(laterality, default_label=default_label)
            rows.append([file_name, label, split_lookup[file_name]])

        os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
        with open(labels_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filename", "label", "split"])
            writer.writerows(rows)

        print(f"Labels CSV created: {labels_csv_path}")
        print(f"- Total base images: {len(rows)}")
        print(f"- Split counts      : train={n_train}, val={n_val}, test={n_test}")
        print(f"- Unknown laterality: {unknown_count}")

    @staticmethod
    def generate_placeholder_labels(
        img_dir: str,
        labels_csv_path: str,
        default_label: int = 0,
        default_split: str = "train",
        overwrite: bool = False,
    ) -> None:
        """
        Creates a starter labels CSV from processed PNG files.
        This is a bootstrap helper; user should update labels/splits before real training.
        """
        if not os.path.isdir(img_dir):
            print(f"Labels generation skipped: processed directory not found at {img_dir}")
            return

        if os.path.exists(labels_csv_path) and not overwrite:
            print(
                f"Labels generation skipped: {labels_csv_path} already exists. "
                "Use --overwrite-labels if you want to recreate it."
            )
            return

        image_files = sorted(
            f for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(".png")
        )

        if not image_files:
            print(f"Labels generation skipped: no PNG files found in {img_dir}")
            return

        os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
        with open(labels_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filename", "label", "split"])
            for file_name in image_files:
                writer.writerow([file_name, default_label, default_split])

        print(f"Starter labels CSV created: {labels_csv_path}")
        print(
            "Important: labels were generated with placeholder values. "
            "Please update `label` and `split` columns before training."
        )


if __name__ == "__main__":
    cfg = ProjectConfig()
    parser = argparse.ArgumentParser(description="Create processed PNG dataset from raw DICOM files.")
    parser.add_argument("--input-dir", type=str, default=cfg.raw_data_path, help="Raw DICOM input directory.")
    parser.add_argument("--output-dir", type=str, default=cfg.processed_data_path, help="Processed PNG output directory.")
    parser.add_argument(
        "--generate-labels",
        action="store_true",
        help="Generate dataset/labels.csv from processed PNG files with train/val/test split.",
    )
    parser.add_argument("--labels-path", type=str, default=cfg.labels_csv_path, help="Path for generated labels CSV.")
    parser.add_argument("--overwrite-labels", action="store_true", help="Overwrite labels CSV if it already exists.")
    parser.add_argument("--default-label", type=int, default=0, help="Fallback label if laterality cannot be inferred.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for labels generation.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for labels generation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for labels generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val/test split.",
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="After labels generation, augment only train split and append rows to labels.csv.",
    )
    args = parser.parse_args()

    creator = DatasetCreator(dicom_dir=args.input_dir, output_dir=args.output_dir)
    creator.create()

    if args.generate_labels:
        DatasetCreator.generate_labels_csv(
            img_dir=args.output_dir,
            labels_csv_path=args.labels_path,
            overwrite=args.overwrite_labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            default_label=args.default_label,
        )

    if args.augment_train:
        MedicalDataAugmenter.generate_offline_augmentations(csv_path=args.labels_path, img_dir=args.output_dir)

