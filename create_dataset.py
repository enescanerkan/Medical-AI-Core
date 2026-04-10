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

    def _create_from_dir(self, input_dir: str, source_tag: str) -> list:
        """
        Processes one DICOM source directory and returns processed file records.
        """
        if not os.path.isdir(input_dir):
            print(f"Input directory not found: {input_dir}. Skipping source '{source_tag}'.")
            return []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # DEĞİŞEN KISIM BURASI: Klasördeki her şeyi al (.dcm şartını kaldırdık)
        dicom_files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]

        if not dicom_files:
            print(f"No files found in {input_dir}. Skipping source '{source_tag}'.")
            return []

        print(f"Starting dataset creation for '{source_tag}'... Scanning {len(dicom_files)} files.")

        processed_count = 0
        skipped_non_dicom_count = 0
        failed_count = 0
        processed_records = []

        for dcm_name in dicom_files:
            dcm_path = os.path.join(input_dir, dcm_name)

            if not is_dicom(dcm_path):
                skipped_non_dicom_count += 1
                print(f"Ignored non-DICOM file: {dcm_name}")
                continue

            # Use deterministic names per source so reruns overwrite instead of duplicating files.
            base_png_name = f"{os.path.splitext(dcm_name)[0]}.png"
            png_name = f"{source_tag}_{base_png_name}"
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
                processed_records.append({"filename": png_name, "laterality": laterality, "source": source_tag})

            except Exception as e:
                failed_count += 1
                print(f"Skipped {dcm_name}: {str(e)}")

        print("\nDataset creation summary:")
        print(f"- Processed DICOM files : {processed_count}")
        print(f"- Ignored non-DICOM     : {skipped_non_dicom_count}")
        print(f"- Failed during process : {failed_count}")

        if processed_count == 0:
            print(
                f"No valid DICOM files were processed from {input_dir}. "
                "Put your raw `.dcm` files in this folder and run again."
            )
        else:
            print(f"Model-ready images for '{source_tag}' saved to: {self.output_dir}")

        return processed_records

    def create(self) -> list:
        """
        Backward-compatible entrypoint: processes `self.dicom_dir` as train/val source.
        """
        return self._create_from_dir(self.dicom_dir, source_tag="trainval")

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
    def _split_train_val_counts(total_count: int, train_ratio: float, val_ratio: float) -> tuple:
        ratio_sum = train_ratio + val_ratio
        if ratio_sum <= 0:
            raise ValueError("train/val ratios must sum to a positive value")

        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum

        n_train = int(round(total_count * train_ratio))
        n_val = total_count - n_train

        while n_train + n_val < total_count:
            n_train += 1
        while n_train + n_val > total_count:
            if n_train > 0:
                n_train -= 1
            else:
                n_val -= 1

        if total_count >= 2:
            # Ensure both splits have at least one sample when possible.
            split_counts = {"train": n_train, "val": n_val}
            for key in ["train", "val"]:
                if split_counts[key] == 0:
                    donor = max(split_counts, key=split_counts.get)
                    if split_counts[donor] > 1:
                        split_counts[donor] -= 1
                        split_counts[key] += 1
            n_train, n_val = split_counts["train"], split_counts["val"]

        return n_train, n_val

    @staticmethod
    def generate_labels_csv(
        img_dir: str,
        labels_csv_path: str,
        trainval_records: list,
        test_records: list,
        overwrite: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        seed: int = 42,
        default_label: int = 0,
    ) -> None:
        """
        Generates labels.csv from processed images.
        - label: inferred from laterality (L->0, R->1), fallback to default_label
        - split: train/val from raw trainval source, test from raw_test source
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

        trainval_files = sorted([row["filename"] for row in trainval_records])
        test_files = sorted([row["filename"] for row in test_records])

        if not trainval_files and not test_files:
            print("Labels generation skipped: no processed base files found from raw/raw_test sources.")
            return

        rng = random.Random(seed)
        shuffled_trainval = trainval_files[:]
        rng.shuffle(shuffled_trainval)

        n_train, n_val = DatasetCreator._split_train_val_counts(
            len(shuffled_trainval),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        split_lookup = {}
        for file_name in shuffled_trainval[:n_train]:
            split_lookup[file_name] = "train"
        for file_name in shuffled_trainval[n_train:n_train + n_val]:
            split_lookup[file_name] = "val"
        for file_name in test_files:
            split_lookup[file_name] = "test"

        unknown_count = 0
        rows = []
        all_base_files = sorted(trainval_files + test_files)
        for file_name in all_base_files:
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
        print(f"- Split counts      : train={n_train}, val={n_val}, test={len(test_files)}")
        print(f"- Unknown laterality: {unknown_count}")


if __name__ == "__main__":
    cfg = ProjectConfig()
    parser = argparse.ArgumentParser(description="Create processed PNG dataset from raw DICOM files.")
    parser.add_argument("--trainval-dir", type=str, default=cfg.raw_data_path, help="Raw DICOM directory for train/val source.")
    parser.add_argument("--test-dir", type=str, default=cfg.raw_test_data_path, help="Raw DICOM directory for test-only source.")
    parser.add_argument("--output-dir", type=str, default=cfg.processed_data_path, help="Processed PNG output directory.")
    parser.add_argument(
        "--generate-labels",
        action="store_true",
        help="Generate one labels.csv where --trainval-dir becomes train/val and --test-dir becomes test.",
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
        default=0.2,
        help="Validation split ratio for labels generation.",
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

    creator = DatasetCreator(dicom_dir=args.trainval_dir, output_dir=args.output_dir)
    trainval_records = creator._create_from_dir(args.trainval_dir, source_tag="trainval")
    # Feature flag control
    if not cfg.process_test_images:
        print("Test image processing is disabled by feature flag.")
        test_records = []
    else:
        test_records = creator._create_from_dir(args.test_dir, source_tag="test")

    if args.generate_labels:
        DatasetCreator.generate_labels_csv(
            img_dir=args.output_dir,
            labels_csv_path=args.labels_path,
            trainval_records=trainval_records,
            test_records=test_records,
            overwrite=args.overwrite_labels,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            default_label=args.default_label,
        )

    if args.augment_train:
        MedicalDataAugmenter.generate_offline_augmentations(csv_path=args.labels_path, img_dir=args.output_dir)

