# 📖 Optical Character Recognition (OCR) - Comparative Study

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-green)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TrOCR-orange)
![LLM](https://img.shields.io/badge/Vision%20Language%20Model-Moondream%20%7C%20Qwen2.5--VL-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## 📌 Project Overview

This project presents a comprehensive implementation and comparative analysis of **Optical Character Recognition (OCR)** using three different approaches:

- **Machine Learning**
- **Deep Learning**
- **Large Language Model (LLM) / Vision-Language Model (VLM)**

The objective is to evaluate the evolution of OCR technologies by comparing traditional Machine Learning techniques with modern Transformer-based Deep Learning models and Vision-Language Models for handwritten and printed text recognition.

The project demonstrates how OCR systems have evolved from feature-based classification to intelligent multimodal models capable of understanding both images and natural language.

---

# 🚀 Project Workflow

```
                    Input Image
                         │
                         ▼
                Image Preprocessing
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
     ▼                   ▼                   ▼
Machine Learning     Deep Learning      LLM/VLM OCR
     │                   │                   │
     ▼                   ▼                   ▼
     SVM              TrOCR Model      Moondream/Qwen2.5-VL
     │                   │                   │
     └───────────────Prediction──────────────┘
                         │
                         ▼
             Performance Evaluation
                         │
                         ▼
      Accuracy • CER • Precision • Recall
      F1 Score • Confusion Matrix
```

---

# 🔹 Method 1: Machine Learning OCR

The first implementation uses a traditional Machine Learning pipeline for handwritten digit recognition.

### Dataset

- MNIST Dataset
- 70,000 handwritten digit images
- 60,000 Training Images
- 10,000 Testing Images

### Image Preprocessing

- Grayscale Conversion
- Gaussian Blur
- Noise Removal
- Binary Thresholding
- Image Resizing
- Pixel Normalization
- Character Centering

### Algorithm Used

- Support Vector Machine (SVM)

### Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 🔹 Method 2: Deep Learning OCR

The second implementation uses Microsoft's **TrOCR (Transformer OCR)** architecture for handwritten text recognition.

Unlike traditional OCR, TrOCR automatically extracts visual features using Transformer networks.

### Architecture

- Vision Transformer (ViT)
- Transformer Decoder
- Transfer Learning

### Techniques Used

- Patch Embedding
- Positional Encoding
- Multi-Head Self Attention
- Cross Attention
- Auto-Regressive Decoding
- Greedy Search Decoding

### Evaluation

- Character Accuracy
- Character Error Rate (CER)
- Sequence Matching
- Character-Level Confusion Matrix

---

# 🔹 Method 3: Large Language Model OCR

The final implementation utilizes a pretrained Vision-Language Model executed locally using Ollama.

Instead of training a model, OCR is performed through **Zero-Shot Inference** using prompt engineering.

### Vision Language Models

- Moondream
- Qwen2.5-VL

### Techniques Used

- Vision-Language Modeling
- Zero-Shot Learning
- Prompt Engineering
- Multimodal Learning
- Auto-Regressive Text Generation
- Local Inference using Ollama

### Advantages

- No model training required
- Minimal preprocessing
- Handles handwritten and printed text
- Offline execution
- High flexibility

---

# 🛠 Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Transformer Models | Hugging Face Transformers |
| Vision Language Models | Moondream, Qwen2.5-VL |
| Local LLM Runtime | Ollama |
| Image Processing | Pillow (PIL), OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib, Seaborn |

---

# 📚 Libraries Used

## Machine Learning

- scikit-learn
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Seaborn

---

## Deep Learning

- torch
- transformers
- Pillow
- difflib
- matplotlib
- seaborn
- numpy

---

## LLM OCR

- ollama
- Pillow
- io (BytesIO)
- os
- matplotlib
- seaborn
- scikit-learn

---

# 📊 Evaluation Metrics

The OCR models were evaluated using multiple performance metrics.

### Machine Learning

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Deep Learning

- Character Accuracy
- Character Error Rate (CER)
- Sequence Matching
- Character-Level Confusion Matrix

### LLM

- Character Accuracy
- Confusion Matrix
- Heatmap Visualization
- Prediction Comparison

---


# 🎯 Features

- Three complete OCR implementations
- Traditional Machine Learning OCR
- Transformer-based OCR
- Vision-Language Model OCR
- Local LLM inference with Ollama
- Character-level evaluation
- Confusion Matrix visualization
- CER and Accuracy calculation
- Comparative analysis of all methods

---

# 📈 Key Highlights

✅ Traditional OCR using SVM

✅ Transformer-based OCR using TrOCR

✅ Vision-Language Model OCR using Moondream/Qwen2.5-VL

✅ Zero-Shot OCR

✅ Character-Level Performance Analysis

✅ Heatmap Visualization

✅ Local AI Inference with Ollama

---

# 🔮 Future Improvements

- Fine-tune Vision-Language Models on custom handwritten datasets
- Add multilingual OCR support
- Improve recognition for degraded and historical documents
- Document Layout Analysis
- Intelligent Document Processing (IDP)
- Named Entity Recognition (NER)
- Key Information Extraction (KIE)
- Real-time OCR using webcam/video streams

---

# 📖 References

- MNIST Dataset
- IAM Handwriting Database
- TrOCR (Microsoft Research)
- Vision Transformer (ViT)
- RoBERTa
- Moondream Vision Language Model
- Qwen2.5-VL
- Ollama
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- OpenCV
- Pillow

---

# 👨‍💻 Authors

**Optical Character Recognition (OCR) Comparative Study**

Developed as part of a **Bachelor of Technology (B.Tech)** project to explore and compare Machine Learning, Deep Learning, and Large Language Model-based OCR systems for handwritten and printed text recognition.
