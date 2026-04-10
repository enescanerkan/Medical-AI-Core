"""
Defines data augmentation pipelines and offline generation for medical images.
"""

import os
import pandas as pd
from PIL import Image
from torchvision import transforms  # PEP-8 Fix: Removed the uppercase 'T' alias

class MedicalDataAugmenter:
    """
    Handles both online preprocessing transforms and offline data augmentation
    specifically tailored for mammography (MLO and CC views).
    """

    def __init__(self, target_size: tuple = (224, 224)):
        """
        Initializes the standard preprocessing pipelines.
        """
        self.standard_transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _flip_binary_label(label):
        """For laterality tasks: horizontal flip swaps L/R labels encoded as 0/1."""
        try:
            value = int(label)
            if value in (0, 1):
                return 1 - value
        except Exception:
            pass
        return label

    @staticmethod
    def generate_offline_augmentations(csv_path: str, img_dir: str, flip_swaps_label: bool = True) -> None:
        """
        Reads labels.csv, safely augments ONLY the 'train' split images using
        horizontal flips and safe rotations (-15 to +15 degrees), saves them to disk,
        and updates the CSV.

        If flip_swaps_label=True, horizontally flipped images get label 1-label
        for binary 0/1 tasks (e.g., left/right laterality).
        """
        df = pd.read_csv(csv_path)

        train_df = df[df['split'] == 'train']
        new_rows = []
        existing_filenames = set(df['filename'].astype(str).tolist())

        # Safe medical rotation angles
        rotation_angles = [-15, -10, -5, 5, 10, 15]

        for index, row in train_df.iterrows():
            img_name = row['filename']

            # Skip already-augmented images so reruns do not explode dataset size.
            if "_aug_" in str(img_name):
                continue

            label = row['label']
            flipped_label = MedicalDataAugmenter._flip_binary_label(label) if flip_swaps_label else label
            split = row['split']

            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            original_img = Image.open(img_path)
            base_name = os.path.splitext(img_name)[0]

            # 1. Horizontal Flip (Fixed for modern Pillow versions >= 10.0)
            hf_img = original_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hf_name = f"{base_name}_aug_hf.png"
            if hf_name not in existing_filenames:
                hf_img.save(os.path.join(img_dir, hf_name))
                new_rows.append({"filename": hf_name, "label": flipped_label, "split": split})
                existing_filenames.add(hf_name)

            # 2. Rotations on original image
            for angle in rotation_angles:
                rot_img = original_img.rotate(angle, fillcolor=0)
                rot_name = f"{base_name}_aug_r{angle}.png"
                if rot_name not in existing_filenames:
                    rot_img.save(os.path.join(img_dir, rot_name))
                    new_rows.append({"filename": rot_name, "label": label, "split": split})
                    existing_filenames.add(rot_name)

            # 3. Rotations on horizontally flipped image
            for angle in rotation_angles:
                rot_hf_img = hf_img.rotate(angle, fillcolor=0)
                rot_hf_name = f"{base_name}_aug_hf_r{angle}.png"
                if rot_hf_name not in existing_filenames:
                    rot_hf_img.save(os.path.join(img_dir, rot_hf_name))
                    new_rows.append({"filename": rot_hf_name, "label": flipped_label, "split": split})
                    existing_filenames.add(rot_hf_name)

        if new_rows:
            augmented_df = pd.DataFrame(new_rows)
            final_df = pd.concat([df, augmented_df], ignore_index=True)
            final_df.to_csv(csv_path, index=False)
            print(f"Success! Generated {len(new_rows)} new augmented images.")
            print(f"New CSV size: {len(final_df)} rows.")
        else:
            print("No 'train' data found to augment.")

if __name__ == "__main__":
    # Standard paths configuration (PEP-8 Fix: lowercase local variables)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    csv_file = os.path.join(project_root, "dataset", "labels.csv")
    image_dir = os.path.join(project_root, "dataset", "processed")

    MedicalDataAugmenter.generate_offline_augmentations(csv_path=csv_file, img_dir=image_dir)