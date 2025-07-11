# 🧠 Report-to-Image Diffusion Model for Medical Imaging

This project explores the use of **conditional diffusion models** to generate medical images (e.g., chest X-rays) from **radiology reports**. The goal is to simulate realistic medical images based on textual clinical descriptions — with potential applications in medical education, dataset augmentation, and interpretability.

---

# diffusion_text_to_image: Starter scaffold for text-to-image radiology diffusion model

# Directory structure suggestion
.
├── config/
│   └── config.yaml
├── models/
│   ├── diffusion.py
│   ├── text_encoder.py
│   ├── text_to_latent.py
│   ├── unet.py
│   └── vae.py
├── sampling/
│   ├── sample_utils.py
│   └── sampler.py
├── scheduler/
│   └── ddpm_scheduler.py
├── generate.py
├── make_latent_dataset.py
├── preprocess_dataset.py
├── train_diffusion_model.py
├── train_text_to_latent.py
├── train_vae_model.py
├── README.md
├── requirements.txt
└── .gitignore