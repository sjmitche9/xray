import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from PIL import Image
import multiprocessing

REPORTS_CSV = "data/indiana_reports.csv"
PROJECTIONS_CSV = "data/indiana_projections.csv"
IMAGE_DIR = "data/images"  # Folder containing images
MAX_LENGTH = 256
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
OUTPUT_PATH = "processed/iu_xray_hf"


def preprocess(example):
    try:
        if not hasattr(preprocess, "tokenizer"):
            preprocess.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        img = Image.open(example["image_path"]).convert("L").resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0

        tokens = preprocess.tokenizer(
            example["report"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        return {
            "image": img,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
    except Exception as e:
        print(f"⚠️ Skipping {example['image_id']} due to error: {e}")
        return {
            "image": np.zeros((256, 256), dtype=np.float32),
            "input_ids": [0] * 256,
            "attention_mask": [0] * 256
        }


def main():
    global tokenizer  # Needed because preprocess accesses it

    print("📄 Loading CSV files...")
    reports = pd.read_csv(REPORTS_CSV)
    projections = pd.read_csv(PROJECTIONS_CSV)

    projections["filepath"] = projections["uid"]

    print("🔗 Merging CSV files...")
    merged = pd.merge(projections, reports, on="uid")

    print("📄 Building records for dataset...")
    records = []
    for _, row in merged.iterrows():
        report_text = f"{row.get('impression', '')} {row.get('findings', '')}".strip()
        records.append({
            "uid": str(row["uid"]),
            "uid": row["uid"],
            "report": report_text,
            "image_path": os.path.join(IMAGE_DIR, row["filename"])
        })

    print(f"✅ Created {len(records)} records")
    dataset = Dataset.from_list(records)

    print(f"🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    num_proc = 14
    print(f"⚙️ Preprocessing dataset using {num_proc} processes...")

    processed_dataset = dataset.map(
        preprocess,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Preprocessing IU X-ray dataset"
    )

    processed_dataset.set_format(
        type="torch",
        columns=["image", "input_ids", "attention_mask"]
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"💾 Saving processed dataset to {OUTPUT_PATH}...")
    processed_dataset.save_to_disk(OUTPUT_PATH)

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
