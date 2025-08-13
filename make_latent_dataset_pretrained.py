# make_latent_dataset_pretrained.py
import os
import yaml
import torch
from tqdm import tqdm
from datasets import load_from_disk, Dataset, Features, Array3D, Value
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

with open("config/config.yaml") as f:
	config = yaml.safe_load(f)

# --- Config knobs ---
STORE_SCALED = bool(config.get("DATASET", {}).get("STORE_SCALED_LATENTS", False))
BATCH_SIZE   = int(config.get("DATASET", {}).get("ENCODE_BATCH_SIZE", 16))
NUM_WORKERS  = int(config.get("DATASET", {}).get("ENCODE_NUM_WORKERS", 4))

# Paths
input_root  = config["DATASET"]["OUTPUT_PATH"]
output_root = config["DATASET"]["LATENT_OUTPUT_PATH"]
os.makedirs(output_root, exist_ok=True)

LATENT_SHAPE = (
	int(config["MODEL"]["LATENT_DIM"]),
	int(config["MODEL"]["LATENT_H"]),
	int(config["MODEL"]["LATENT_W"]),
)


device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
ckpt_path = config.get("MODEL", {}).get("VAE_CHECKPOINT", None)

if ckpt_path and os.path.exists(ckpt_path):
	vae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
	vae_source = f"checkpoint:{ckpt_path}"
else:
	vae_source = "pretrained:stabilityai/sd-vae-ft-ema"

vae = vae.to(device).eval()

# Prefer explicit config scale; otherwise VAE config; fallback to SD default
SCALE = float(config.get("MODEL", {}).get("LATENT_SCALE",
			 getattr(vae.config, "scaling_factor", 0.18215)))


def encode_chunk(dataset):
	"""
	Encode a dataset split/chunk into latents using posterior MEAN (deterministic).
	If STORE_SCALED is True, multiply by SCALE before saving.
	"""
	loader = DataLoader(
		dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)
	latents = []

	for batch in tqdm(loader, desc="Encoding batch", leave=False):
		imgs = batch["image"].to(device, non_blocking=True)

		# VAE expects 3-channel in [-1, 1]; expand grayscale to RGB
		if imgs.ndim == 4 and imgs.shape[1] == 1:
			imgs = imgs.repeat(1, 3, 1, 1)
		imgs = imgs.float()
		imgs = imgs * 2.0 - 1.0  # map [0,1] -> [-1,1]

		with torch.no_grad():
			dist = vae.encode(imgs).latent_dist     # posterior q(z|x)
			z = dist.mean                           # <-- deterministic!
			if STORE_SCALED:
				z = z * SCALE
			z = z.detach().cpu().float()

		# package examples
		for zi, report in zip(z, batch["report"]):
			if tuple(zi.shape) != LATENT_SHAPE:
				raise ValueError(
					f"❌ Latent has wrong shape: {tuple(zi.shape)} != {LATENT_SHAPE}"
				)
			latents.append({"z_target": zi.numpy().astype("float32"), "report": report})

	return latents

def save_meta(dirpath, split_desc):
	meta = {
		"split": split_desc,
		"store_scaled": STORE_SCALED,
		"scaling_factor": SCALE,
		"latent_shape": LATENT_SHAPE,
		"posterior": "mean",       # we use deterministic posterior mean
		"vae_source": vae_source,  # checkpoint or pretrained string
	}
	with open(os.path.join(dirpath, "meta.yaml"), "w") as f:
		yaml.safe_dump(meta, f, sort_keys=False)

def encode_and_save(dataset_path, output_path, split_desc):
	if os.path.exists(os.path.join(output_path, "dataset_info.json")):
		print(f"[skip] {split_desc} already complete")
		return

	print(f"[info] Processing {split_desc}")
	dataset = load_from_disk(dataset_path)
	dataset.set_format("torch", columns=["image", "report"])

	latents = encode_chunk(dataset)

	os.makedirs(output_path, exist_ok=True)
	features = Features({
		"z_target": Array3D(shape=LATENT_SHAPE, dtype="float32"),
		"report": Value("string"),
	})

	Dataset.from_list(latents, features=features).save_to_disk(output_path)
	save_meta(output_path, split_desc)
	print(f"[done] Saved latent {split_desc} to {output_path}")

def main():
	# --- Train Chunks ---
	chunk_id = 0
	while True:
		chunk_path = os.path.join(input_root, f"train_chunk_{chunk_id}")
		if not os.path.exists(chunk_path):
			break
		out_path = os.path.join(output_root, f"latent_train_chunk_{chunk_id}")
		encode_and_save(chunk_path, out_path, f"train_chunk_{chunk_id}")
		chunk_id += 1
		# break # use this to save one chunk and then skip to val/test

	# --- Val/Test ---
	for split in ["val", "test"]:
		split_path = os.path.join(input_root, split)
		if os.path.exists(split_path):
			out_path = os.path.join(output_root, f"latent_{split}")
			encode_and_save(split_path, out_path, split)

if __name__ == "__main__":
	main()