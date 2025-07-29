# preprocess_dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from datasets import Dataset, Features, Array3D, Value, Sequence
from transformers import AutoTokenizer
import yaml
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

def process_row(row, root_dir, image_size, tokenizer):
    subject_id = row.subject_id
    study_id = row.study_id
    dicom_id = row.dicom_id
    split_group = row["split"]
    report = row["report"]

    orientation = str(row.get("ViewPosition", "")).strip().lower()
    if orientation:
        report += f", view: {orientation}"

    image_path = os.path.join(
        root_dir,
        f"p{str(subject_id)[:2]}",
        f"p{subject_id}",
        f"s{study_id}",
        f"{dicom_id}.jpg"
    )

    if not os.path.exists(image_path):
        return None

    try:
        image = Image.open(image_path).convert("L")
        image = image.resize((image_size, image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        tokenized = tokenizer(report, truncation=True, padding="max_length", max_length=128)

        return {
            "image": image,
            "split": split_group,
            "report": report,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
    except:
        return None


def process_train_in_batches(split_df, root_dir, image_size, tokenizer, num_cpus, output_path, batch_size=10000):
    print(f"[info] processing train set in batches with {len(split_df)} rows...")
    rows = [row for _, row in split_df.iterrows()]
    features = Features({
        "image": Array3D(shape=(1, image_size, image_size), dtype="float32"),
        "report": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "split": Value("string")
    })

    for i in range(0, len(rows), batch_size):
        chunk_rows = rows[i:i + batch_size]
        chunk_path = os.path.join(output_path, f"train_chunk_{i // batch_size}")
        print(f"[info] checking: {chunk_path}")
        if os.path.exists(chunk_path):
            print(f"[info] skipping existing chunk: {chunk_path}")
            continue

        with mp.Pool(processes=num_cpus) as pool:
            func = partial(process_row, root_dir=root_dir, image_size=image_size, tokenizer=tokenizer)
            mapped = list(tqdm(pool.imap(func, chunk_rows), total=len(chunk_rows), desc=f"[mp] train_chunk_{i // batch_size}"))
            results = [res for res in mapped if res is not None]

        ds_chunk = Dataset.from_list(results, features=features)
        ds_chunk.save_to_disk(chunk_path)


def process_split(split_name, split_df, root_dir, image_size, tokenizer, num_cpus):
    print(f"[info] processing {split_name} set with {len(split_df)} rows...")
    with mp.Pool(processes=num_cpus) as pool:
        func = partial(process_row, root_dir=root_dir, image_size=image_size, tokenizer=tokenizer)
        mapped = list(tqdm(pool.imap(func, [row for _, row in split_df.iterrows()]), total=len(split_df), desc=f"[mp] {split_name}"))
    return [res for res in mapped if res is not None]


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    data_cfg = config["DATASET"]
    tokenizer_name = config["MODEL"]["TOKENIZER_NAME"]

    root_dir = data_cfg["ROOT_DIR"]
    metadata_path = data_cfg["METADATA_PATH"]
    split_path = data_cfg["SPLIT_PATH"]
    output_path = data_cfg["OUTPUT_PATH"]
    image_size = data_cfg["IMAGE_SIZE"]
    num_cpus = data_cfg.get("NUM_CPUS", 4)
    limit_samples = data_cfg.get("LIMIT_SAMPLES", None)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    split = pd.read_csv(split_path)

    # Load labels
    chexpert = pd.read_csv("data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv")
    negbio = pd.read_csv("data/mimic-cxr-jpg/mimic-cxr-2.0.0-negbio.csv")
    label_names = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
        "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumonia",
        "Pneumothorax", "Pleural Other", "Support Devices", "No Finding"
    ]
    labels = pd.merge(chexpert, negbio, on=["subject_id", "study_id"], suffixes=("_chex", "_neg"))

    def agree(row):
        agreed = []
        for label in label_names:
            v1 = row.get(f"{label}_chex")
            v2 = row.get(f"{label}_neg")
            if v1 == 1.0 and v2 == 1.0:
                agreed.append(label.lower())
        return ", ".join(agreed) if agreed else "no finding"

    labels["report"] = labels.apply(agree, axis=1)

    # Merge all
    df = pd.merge(metadata, split, on=["subject_id", "study_id", "dicom_id"], how="inner")
    df = pd.merge(df, labels[["subject_id", "study_id", "report"]], on=["subject_id", "study_id"], how="inner")

    if limit_samples is not None:
        df = df.groupby("split").apply(lambda x: x.head(limit_samples)).reset_index(drop=True)

    print(f"[info] processing {len(df)} total rows...")

    splits = {name: group for name, group in df.groupby("split")}
    process_train_in_batches(splits.get("train", pd.DataFrame()), root_dir, image_size, tokenizer, num_cpus, output_path)

    features = Features({
        "image": Array3D(shape=(1, image_size, image_size), dtype="float32"),
        "report": Value("string"),
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "split": Value("string")
    })

    val_data = process_split("validate", splits.get("validate", pd.DataFrame()), root_dir, image_size, tokenizer, num_cpus)
    test_data = process_split("test", splits.get("test", pd.DataFrame()), root_dir, image_size, tokenizer, num_cpus)

    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")
    Dataset.from_list(val_data, features=features).save_to_disk(val_path)
    Dataset.from_list(test_data, features=features).save_to_disk(test_path)

    print("[info] saved all dataset splits separately to disk")
    print("[info] done.")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()