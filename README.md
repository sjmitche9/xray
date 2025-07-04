# 🧠 Report-to-Image Diffusion Model for Medical Imaging

This project explores the use of **conditional diffusion models** to generate medical images (e.g., chest X-rays) from **radiology reports**. The goal is to simulate realistic medical images based on textual clinical descriptions — with potential applications in medical education, dataset augmentation, and interpretability.

---

## 📂 Dataset

### ✅ Phase 1: IU X-ray (Start Here)
- 7,470 chest X-ray images (PNG)
- Paired with corresponding radiology reports
- Source: [Open-I (NLM)](https://openi.nlm.nih.gov/)
- Report–image mapping from: [MedKLIP IU-Xray JSON](https://github.com/Alibaba-MIIL/MedKLIP/tree/main/data/iu_xray)

### 🔜 Phase 2: MIMIC-CXR (Once Approved)
- 377,000+ chest X-rays
- Rich, real-world reports and labels
- Requires credentialing via PhysioNet (MIT)

---

## 🏗️ Project Structure

project/
│
├── data/
│ ├── images/ # Chest X-ray PNGs
│ └── iu_xray.json # Report/image mappings and text sections
│
├── src/
│ ├── preprocessing/ # Text cleaning, DICOM parsing, etc.
│ ├── model/ # Diffusion model definition
│ ├── training/ # Fine-tuning loop with LoRA/SD
│ └── inference/ # Report → image pipeline
│
├── scripts/ # Downloaders, evaluation, experiments
└── README.md