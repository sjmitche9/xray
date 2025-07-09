# 🧠 Report-to-Image Diffusion Model for Medical Imaging

This project explores the use of **conditional diffusion models** to generate medical images (e.g., chest X-rays) from **radiology reports**. The goal is to simulate realistic medical images based on textual clinical descriptions — with potential applications in medical education, dataset augmentation, and interpretability.

---

# diffusion_text_to_image: Starter scaffold for text-to-image radiology diffusion model

# Directory structure suggestion
# diffusion_text_to_image/
# ├── config.yaml
# ├── train.py
# ├── generate.py
# ├── preprocess_dataset.py
# ├── models/
# │   ├── text_encoder.py
# │   ├── vae.py
# │   └── unet.py
# ├── scheduler/
# │   └── ddpm_scheduler.py
# ├── utils/
# │   ├── dataset.py
# │   └── wandb_logging.py