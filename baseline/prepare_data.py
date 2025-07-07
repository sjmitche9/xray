import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from PIL import Image
import multiprocessing

REPORTS_CSV = "data/indiana_reports.csv"
PROJECTIONS_CSV = "data/indiana_projections.csv"
IMAGE_DIR = "data/images"
MAX_LENGTH = 256
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_PATH = "processed/iu_xray_hf_split"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):

    try:
        # Load and normalize image
        img = Image.open(example["image_path"]).convert("L").resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0  # normalize

        # Tokenize text
        tokens = tokenizer(
            example["report"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Make labels by replacing pad token IDs with -100 for loss masking
        labels = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in tokens["input_ids"]
        ]

        return {
            "image": img,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels,
        }

    except Exception as e:
        print(f"⚠️ Error in preprocessing {example.get('image_path', '')}: {e}")
        return {
            "image": np.zeros((256, 256), dtype=np.float32),
            "input_ids": [0] * MAX_LENGTH,
            "attention_mask": [0] * MAX_LENGTH,
            "labels": [-100] * MAX_LENGTH,
        }


def main():

    print("📄 Loading CSV files...")
    reports = pd.read_csv(REPORTS_CSV)
    projections = pd.read_csv(PROJECTIONS_CSV)

    # If 'filename' column is missing in projections, try to infer from uid or filepath
    if "filename" not in projections.columns:
        # Attempt to use uid with '.dcm' or '.png' extension as fallback
        projections["filename"] = projections["uid"].astype(str)

    print("🔗 Merging CSV files on 'uid'...")
    merged = pd.merge(projections, reports, on="uid")

    print("📄 Building dataset records...")

    records = []
    for _, row in merged.iterrows():

        report_text = f"{row.get('impression', '')} {row.get('findings', '')}".strip()
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        records.append({
            "uid": str(row["uid"]),
            "report": report_text,
            "image_path": img_path
        })

    print(f"✅ Created {len(records)} dataset records")
    dataset = Dataset.from_list(records)

    num_proc = min(14, multiprocessing.cpu_count())
    print(f"⚙️ Preprocessing dataset using {num_proc} parallel processes...")

    processed_dataset = dataset.map(preprocess)

    print("🔀 Splitting into train/validation/test sets...")
    train_test = processed_dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })


    for split in ["train", "validation", "test"]:
        dataset_dict[split].set_format(
            type="torch",
            columns=["image", "input_ids", "attention_mask", "labels"]
        )


    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"💾 Saving processed dataset to {OUTPUT_PATH}...")
    dataset_dict.save_to_disk(OUTPUT_PATH)

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()