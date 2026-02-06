# 🕵️‍♂️ Universal Forgery Detector: A Hybrid Approach to Digital Forensics

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Sumit0098073e5/Universal-Deepfake-Detector)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Maintained-success)]()

> **"Detecting the undetectable."**

A State-of-the-Art (SOTA) Hybrid Deep Learning system designed to identify **AI-Generated Images (Generative AI)**, **Deepfakes**, and **Traditional Image Manipulation** (Photoshop/Splicing). Unlike standard detectors that fail on modern Diffusion models, this system leverages a **Dual-Stream Architecture** combining visual semantics with forensic noise analysis.

---

## 🚀 Live Demo
**Don't just read about it. Test it.**
We have deployed a production-ready version of the model on Hugging Face Spaces (running on CPU/GPU).
### [👉 Launch the Universal Forgery Detector App](https://huggingface.co/spaces/Sumit0098073e5/Universal-Deepfake-Detector)

---

## 📖 Table of Contents
1. [🌟 Project Motivation](#-project-motivation)
2. [📉 The Research Gap](#-the-research-gap)
3. [🧠 System Architecture](#-system-architecture)
    - [Stream A: Visual Semantics](#stream-a-visual-semantics-swin-transformer)
    - [Stream B: Forensic Noise](#stream-b-forensic-noise-bayarconv)
    - [The Fusion Head](#the-fusion-head)
4. [📂 Dataset Curation](#-dataset-curation)
5. [📜 The Training Saga](#-the-training-saga)
6. [⚔️ Engineering Challenges & Solutions](#%EF%B8%8F-engineering-challenges--solutions)
7. [📊 Performance Metrics](#-performance-metrics)
8. [🛠️ Installation & Setup](#%EF%B8%8F-installation--setup)
9. [🔮 Future Roadmap](#-future-roadmap)
10. [🤝 Acknowledgments](#-acknowledgments)

---

## 🌟 Project Motivation

In the last 24 months, the digital landscape has shifted dramatically. Tools like **Midjourney v6**, **DALL-E 3**, and **Stable Diffusion XL** allow anyone to generate hyper-realistic imagery in seconds. While this creative revolution is exciting, it poses a catastrophic risk to information integrity.

* **The Threat:** Malicious actors are already using GenAI to fabricate evidence, create non-consensual deepfakes, and manufacture "fake news" that is visually indistinguishable from reality.
* **The Failure of Legacy Tech:** Traditional forensic tools (ELA, noise analysis) were built for Photoshop splicing. They look for "jagged edges" or "compression artifacts." Modern Diffusion models don't leave these traces; they generate statistically "perfect" noise distributions.

**Our Mission:** To build a robust, universal detector that does not rely on a single artifact type. We aim to detect the *process* of generation itself, whether it's a pixel-level splice or a diffusion-based synthesis.

---

## 📉 The Research Gap

During our literature review and initial testing, we identified a critical failure mode in existing open-source detectors, which we call the **"Generation Gap"**.

| Feature | Legacy Detectors (ResNet/ELA) | Modern GenAI (Midjourney/DALL-E) | **Our Hybrid Approach** |
| :--- | :--- | :--- | :--- |
| **Artifact Focus** | Sharp edges, splicing boundaries | Smooth textures, logic errors | **Both** |
| **Noise Analysis** | Looks for JPEG grid inconsistencies | Generates clean Gaussian noise | **Constrained Conv Layers** |
| **Resolution** | Resizes images to 224x224 (Loses detail) | High-Res (1024x1024+) | **Patch-Based Scanning** |

**The Core Insight:** A standard CNN sees a "cat" and thinks "Real." A Forensic Filter sees "noise" and thinks "Real." Only by combining **Semantic Understanding** (Swin) with **Forensic Residuals** (BayarConv) can we catch modern fakes.

---

## 🧠 System Architecture

Our model utilizes a **Dual-Stream Hybrid Network**. This allows the system to "see" the image in two different dimensions simultaneously.

### 🔹 Stream A: Visual Semantics (Swin Transformer)
* **Backbone:** `swin_tiny_patch4_window7_224` (Pretrained on ImageNet).
* **Why Swin?** Unlike CNNs (which look at local pixels), Transformers use **Self-Attention**. This allows the model to understand global context.
* **What it detects:**
    * *Semantic Inconsistencies:* A shadow falling in the wrong direction.
    * *Physiological Errors:* The classic "AI hands" problem, asymmetrical eyes, or merging textures.
    * *Texture Smoothing:* The "plastic skin" look common in Midjourney.

### 🔹 Stream B: Forensic Noise (BayarConv + EfficientNet)
* **Input Layer:** **Bayar Constraint Layer (BayarConv2d)**.
    * This is a custom convolution layer where the center weight is forced to be -1 and the sum of all weights is forced to be 0.
    * *Math:* $w_{0,0} = -1, \sum w_{i,j} = 0$.
    * *Effect:* This acts as a high-pass filter that **destroys image content** (colors, shapes) and preserves only the **prediction error (noise residuals)**.
* **Backbone:** `efficientnet_b0`.
* **What it detects:**
    * *Invisible Fingerprints:* Unique frequency patterns left by GAN upsamplers.
    * *Grid Artifacts:* Checkerboard patterns from Transposed Convolutions.

### 🔹 The Fusion Head
The feature vectors from Stream A (1000 dims) and Stream B (1000 dims) are concatenated into a dense vector (2000 dims).
* **Layers:** Linear -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear.
* **Output:** Softmax Probability distribution over 3 classes: `['Real', 'Edited', 'AI']`.

---

## 📂 Dataset Curation

We curated a specialized dataset to ensure "Universal" coverage.

| Dataset Source | Class | Count | Purpose |
| :--- | :--- | :--- | :--- |
| **COCO 2017** | Real 📸 | 4,000 | Ground truth for natural scenes and objects. |
| **CASIA 2.0 (Tp)** | Edited 🎨 | 4,000 | Gold standard for splicing, copy-move, and Photoshop. |
| **ArtiFact** | AI 🤖 | 1,500 | Extensive collection of GANs, Glide, and early Diffusion. |
| **Midjourney v6** | AI 🤖 | 1,500 | Manually scraped dataset of high-fidelity v6 prompts. |
| **Gemini/DALL-E 3** | AI 🤖 | 1,532 | Custom generated images to cover the newest models. |

**Total:** ~12,500 Images (Balanced Split).

---

## 📜 The Training Saga

Building this model was not a straight line. We encountered and overcame significant "Catastrophic Forgetting" issues.

###  Phase 1: The Foundation
We trained the baseline architecture on COCO and CASIA. The model became excellent at detecting Photoshop edits (95% Accuracy) but completely failed on Midjourney images, classifying them as "Real" because they lacked the sharp edges of spliced photos.

### Phase 2: The "Sniper" Fine-Tuning
We attempted to fine-tune purely on Diffusion data.
* **Strategy:** Freeze the Forensic Stream (Stream B) and train only the Visual Stream (Stream A).
* **Result:** Accuracy on AI images spiked to 99%.
* **The Crash:** Accuracy on Photoshop edits plummeted to 42%. The model had "overwritten" its understanding of traditional forgery.

### Phase 3: The Grand Unification (Replay Training)
We implemented a **Replay Strategy**. We combined all datasets (Real, Edited, Old AI, New AI) into a single balanced loader.
* **Hyperparameters:** Learning Rate `1e-5` (Very Low), Weight Decay `1e-4`, AdamW Optimizer.
* **Outcome:** The model converged with high accuracy across ALL three categories, proving the Hybrid architecture can hold conflicting patterns in memory.

---

## ⚔️ Engineering Challenges & Solutions

### 1. The "Resizing" Trap
**Problem:** DALL-E 3 generates 1024x1024 images. Resizing them to 224x224 (standard Model input) blurs out the tiny artifacts (hair blending, textile patterns) that reveal them as AI.
**Solution:** **Patch-Based Voting ("The Magnifying Glass")**.
Instead of analyzing one resized image, we implement a sliding window strategy during inference:
1.  Extract 5 patches: Center, Top-Left, Top-Right, Bot-Left, Bot-Right.
2.  Run inference on all 5 patches + the global resized view.
3.  **Logic:** If *any* single patch is predicted as AI with >90% confidence, the entire image is flagged. This improved DALL-E 3 detection from **25% -> 80%**.

### 2. Streamlit Model Drift
**Problem:** During deployment, we noticed the model's predictions becoming erratic after multiple uses.
**Root Cause:** The `BayarConv2d` layer was enforcing its weight constraints *in-place* during the forward pass. This meant the weights were being divided by their sum every time a user uploaded an image, eventually degrading to zero.
**Solution:** Modified the inference class to treat weights as read-only, fixing the drift.

---

## 📊 Performance Metrics

**Test Set Evaluation (150 Unseen Images):**

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Real** | 0.96 | 1.00 | 0.98 |
| **Edited** | 1.00 | 0.96 | 0.98 |
| **AI** | 0.98 | 0.98 | 0.98 |
| **Overall** | **0.98** | **0.98** | **0.98** |

**Real-World Stress Test:**
* **Midjourney v6:** 98% Detection Rate.
* **DALL-E 3:** 85% Detection Rate (via Patch Scanning).
* **Photoshop:** 96% Detection Rate.

---

## 🛠️ Installation & Setup

Want to run this locally? Follow these steps.

### 1. Clone the Repository
```bash
git clone https://github.com/Sumit0098073e5/universal-forgery-detector.git
cd universal-forgery-detector
```

### 2. Download Model Weights (⚠️ Crucial)
The `.pth` file is too large for GitHub LFS. You must download it from our Hugging Face repository.

📥 **Download Link:** [universal_forgery_FINAL.pth](https://huggingface.co/spaces/Sumit0098073e5/Universal-Deepfake-Detector/blob/main/universal_forgery_FINAL.pth)

**Action:** Place this file in the root directory of the project.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Linux Users:** You may need to install `libgl1`:
```bash
sudo apt-get install libgl1-mesa-glx
```

### 4. Run the App
```bash
streamlit run app.py
```
The app will launch in your browser at `http://localhost:8501`.

🔮 Future Roadmap
Video Support: Implementing frame-by-frame analysis to detect Deepfake videos (FaceSwaps).

Frequency Analysis Visualization: Adding a Discrete Cosine Transform (DCT) view to the UI to show users the "invisible" noise.

Browser Extension: A lightweight JavaScript version to scan images directly on Twitter/Instagram.

🤝 Acknowledgments
Timm Library: For the pristine PyTorch Image Models.

Albumentations: For robust data augmentation pipelines.

Streamlit: For enabling rapid full-stack prototyping.

Research Papers: Inspired by "The constrained convolution layer for camera source identification" (Bayar et al.).

Built with ❤️ and ☕ by Sumit Kumar
