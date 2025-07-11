import os
import torch
import yaml
from collections import defaultdict
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from multiprocessing import Pool, set_start_method, cpu_count
from functools import partial

# --- Load config ---
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_path = config["DATASET"]["OUTPUT_PATH"]
latent_output_path = os.path.join(output_path, "latent")
num_workers = config["DATASET"].get("NUM_CPUS", cpu_count())

# Globals for worker processes
tokenizer = None
text_model = None
vae = None
device = None


def init_worker():
    global tokenizer, text_model, vae, device
    import torch
    from transformers import AutoTokenizer, AutoModel
    from models.vae import VAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device).eval()

    vae = VAE().eval()
    vae.load_state_dict(torch.load(config["MODEL"]["VAE_CHECKPOINT"], map_location=device))
    vae.to(device)


def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()


def encode_image(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, logvar = vae.encoder(image_tensor)
        z = vae.reparameterize(mu, logvar)
    return z.squeeze(0).cpu()


def process_example(example):
    try:
        new_example = {k: v for k, v in example.items()}
        new_example["z_target"] = encode_image(example["image"])
        new_example["text_embedding"] = encode_text(example.get("report", ""))
        return new_example
    except Exception as e:
        print(f"[warn] failed processing example: {e}")
        return None


def process_dataset(dataset, desc, workers=num_workers):
    dataset.set_format(type="torch", columns=["image"], output_all_columns=True)
    examples = list(dataset)

    with Pool(workers, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_example, examples), total=len(examples), desc=desc))

    return [r for r in results if r is not None]


def save_train_chunks():
    chunk_id = 0
    train_chunk_count = 0
    chunk_size = 10000
    buffer = []

    while True:
        chunk_path = os.path.join(output_path, f"train_chunk_{chunk_id}")
        if not os.path.exists(chunk_path):
            break

        dataset = load_from_disk(chunk_path)
        chunk = process_dataset(dataset, f"Encoding train_chunk_{chunk_id}")

        buffer.extend(chunk)
        while len(buffer) >= chunk_size:
            this_chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]

            output_path_chunk = os.path.join(latent_output_path, f"latent_train_chunk_{train_chunk_count}")
            os.makedirs(output_path_chunk, exist_ok=True)
            Dataset.from_list(this_chunk).save_to_disk(output_path_chunk)
            print(f"[info] saved {output_path_chunk}")
            train_chunk_count += 1

        chunk_id += 1

    if buffer:
        output_path_chunk = os.path.join(latent_output_path, f"latent_train_chunk_{train_chunk_count}")
        os.makedirs(output_path_chunk, exist_ok=True)
        Dataset.from_list(buffer).save_to_disk(output_path_chunk)
        print(f"[info] saved {output_path_chunk}")


def save_val_test():
    result = {}
    for split in ["val", "test"]:
        split_path = os.path.join(output_path, split)
        if not os.path.exists(split_path):
            continue
        dataset = load_from_disk(split_path)
        encoded = process_dataset(dataset, f"Encoding {split}")
        result[split] = Dataset.from_list(encoded)

    if result:
        os.makedirs(latent_output_path, exist_ok=True)
        DatasetDict(result).save_to_disk(latent_output_path)
        print(f"[info] saved latent val/test dataset to {latent_output_path}")


if __name__ == "__main__":
    set_start_method("spawn")
    save_train_chunks()
    save_val_test()
