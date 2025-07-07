import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import Dataset, DatasetDict

# Paths
ROOT_DIR = "data/mimic-cxr-jpg/files"
METADATA_PATH = "data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv"
SPLIT_PATH = "data/mimic-cxr-jpg/mimic-cxr-2.0.0-split.csv"
OUTPUT_PATH = "processed/mimic_cxr_hf_subset"

# Limit to 100 images for testing
LIMIT = 100

# Load metadata
metadata = pd.read_csv(METADATA_PATH)
split = pd.read_csv(SPLIT_PATH)

# Merge data
df = metadata.merge(split, on=['dicom_id', 'subject_id', 'study_id'], how="inner")


# Limit to 100 total examples, ensuring all splits are represented
df_train = df[df["split"] == "train"].head(34)
df_val = df[df["split"] == "validate"].head(33)
df_test = df[df["split"] == "test"].head(33)

df = pd.concat([df_train, df_val, df_test])


print(f"[Info] Processing {len(df)} images...")

data = []


for _, row in tqdm(df.iterrows(), total=len(df)):
    subject_id = row["subject_id"]
    study_id = row["study_id"]
    dicom_id = row["dicom_id"]
    split_group = row["split"]

    # Build image path
    image_path = os.path.join(ROOT_DIR, f"p{str(subject_id)[:2]}", f"s{study_id}", f"{dicom_id}.jpg")

    if not os.path.exists(image_path):
        print(f"[Warning] Image not found: {image_path}")
        continue

    # Load and preprocess image
    image = Image.open(image_path).convert("L")  # Grayscale
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)

    data.append({
        "image": image,
        "split": split_group
    })

# Build DatasetDict
train_data = [d for d in data if d["split"] == "train"]
val_data = [d for d in data if d["split"] == "validate"]
test_data = [d for d in data if d["split"] == "test"]

ds_dict = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data)
})

print(f"[Info] Saving DatasetDict to {OUTPUT_PATH}")
ds_dict.save_to_disk(OUTPUT_PATH)
print("[Info] Done.")