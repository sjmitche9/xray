# 🧠 Text-to-X-ray: Generating Chest X-rays from Radiology Reports with Diffusion Models

This project explores a novel direction in multimodal learning: generating realistic chest X-ray images directly from free-text radiology reports. By combining latent-space modeling with modern diffusion techniques, we build a pipeline that can reconstruct plausible medical images conditioned on clinical descriptions.

---

## 🚀 Project Goals

- Generate chest X-ray images from radiology reports using latent diffusion.
- Leverage pretrained medical language models (Bio_ClinicalBERT) and a variational autoencoder.
- Evaluate the feasibility of text-to-image generation for clinical education, simulation, and dataset augmentation.

---

## 📁 Project Structure

```
.
├── config/
│   └── config.yaml                     # Centralized training and path configuration
│
├── models/
│   ├── vae.py                          # Variational Autoencoder for image compression
│   ├── text_to_latent.py              # MLP to map BERT embeddings to image latents
│   ├── text_encoder.py                # Frozen Bio_ClinicalBERT wrapper
│   ├── unet.py                        # Conditional U-Net for full image diffusion
│   ├── diffusion.py                   # End-to-end diffusion model wrapper
│   └── latent_unet.py                 # Lightweight UNet for simplified latent training
│
├── scheduler/
│   └── ddpm_scheduler.py              # Beta schedule for DDPM diffusion
│
├── train_vae_model.py                 # Trains VAE on X-ray images
├── train_text_to_latent.py           # Trains mapping from report embeddings to image latents
├── train_diffusion_model.py          # Trains diffusion model to generate latents
│
├── preprocess_dataset.py             # Converts raw image/report pairs into disk chunks
├── make_latent_dataset.py            # Precomputes latents and embeddings for fast training
├── generate.py                        # Generates samples from a report prompt
│
├── requirements.txt                   # Dependencies
├── README.md                          # You're here
```

---

## 🧪 Workflow Overview

1. **Preprocess the Dataset**

2. **Train the Variational Autoencoder (VAE)**

3. **Generate Latent Dataset**

4. **Train Text-to-Latent Model**

5. **Train Latent Diffusion Model**

6. **Generate Image from Report**

---

## 📌 Notes

- All configuration is handled through `config/config.yaml`.
- Chunked training allows scaling to large datasets on low-memory machines.
- Training and evaluation logs are managed via [Weights & Biases](https://wandb.ai/).

---

## 📚 Citation & Credits

- MIMIC-CXR: Johnson A, Lungren M, Peng Y, Lu Z, Mark R, Berkowitz S, Horng S. MIMIC-CXR-JPG - chest radiographs with structured labels (version 2.1. 0). PhysioNet. 2024. RRID:SCR_007345. Available from: https://doi.org/10.13026/jsn5-t979
- Bio_ClinicalBERT: Alsentzer et al.
- DDPM: Ho et al. 2020

---

## 🛠️ Future Work

- CLIPScore or radiology-specific generation metrics
- Mixed precision and multi-GPU support
- Zero-shot transfer from synthetic to real domains