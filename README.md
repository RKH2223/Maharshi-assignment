@'
# 🧠 AI Technical Assignment – Computer Vision & OCR (Offline)

## 📌 Overview
This project implements a fully offline Computer Vision and OCR system designed for industrial and hardware-constrained environments.  
The solution avoids cloud APIs and works entirely offline, making it suitable for edge or laptop deployment.

The system includes:
1. Human & Animal Detection (Video-based)
2. Offline OCR for Industrial / Stenciled Text (Image-based)

---

## 📂 Project Structure

```text
project/
├── datasets/
│   ├── part_a/
│   └── part_b/
├── models/
│   ├── detection_model.pth
│   └── classification_model.pth
├── test_videos/
├── outputs/
│   ├── annotated_videos/
│   └── ocr_results/
├── main.py
├── streamlit_app.py
├── requirements.txt
└── README.md


## 📄 Assignment Deliverable: Single Python Script Explanation

This project includes a single consolidated Python script (main.py) that documents and implements the complete AI pipeline.

The script contains:
- Well-commented, step-by-step explanations
- Dataset and model selection justification
- Explanation of training, inference, and OCR pipelines
- Challenges faced and possible improvements

The goal is to ensure the solution is readable, reproducible, and suitable for real-world industrial deployment.

---

## 📊 Dataset Sources & Download Instructions

### Part A: Human & Animal Detection Dataset

Dataset Source:
Open Images Dataset V7  
https://storage.googleapis.com/openimages/web/index.html

Selected Classes:
- Person
- Dog
- Cat
- Horse
- Elephant
- Bear

Download Commands Used:

pip install openimages

oid_v7_download --classes Person Dog Cat Horse Elephant Bear --type_data train validation --limit 5000 --dest datasets/part_a/

Annotations are used in Pascal VOC (XML) format.

---

### Part B: Industrial OCR Dataset

The OCR dataset consists of industrial-style images containing:
- Painted or stenciled text
- Faded markings
- Low contrast
- Surface damage

The entire OCR pipeline works fully offline.

---

## 🧠 Model Selection & Justification

Detection:
- Faster R-CNN (ResNet-50 backbone)
- Selected for high localization accuracy and robustness

Classification:
- ResNet-50 for Human vs Animal classification

OCR:
- EasyOCR
- Selected for offline capability and robustness on degraded text

---

## 🏋️ Training Pipeline

- Transfer learning with pretrained backbones
- Dataset reduction for limited GPU memory
- Batch size tuning to avoid CUDA OOM errors
- Backbone freezing during fine-tuning
- Metrics logging using Weights & Biases (wandb)

---

## 🎥 Inference Pipeline – Part A

1. Videos placed in test_videos/
2. Faster R-CNN detects objects
3. ResNet-50 classifies Human vs Animal
4. Annotated videos saved to outputs/annotated_videos/

---

## 📝 OCR Pipeline – Part B

1. OpenCV preprocessing (grayscale, denoising, CLAHE, thresholding)
2. EasyOCR text detection
3. Bounding box extraction
4. Outputs generated:
   - Structured JSON
   - Annotated image

outputs/ocr_results/
├── sample.json
└── sample_annotated.jpg

---

## ⚠️ Challenges Faced

- Limited GPU memory on laptop
- CUDA out-of-memory errors
- OCR accuracy on faded industrial text
- Annotation inconsistencies

---

## 🛠️ Possible Improvements

- Use lighter detection models for edge devices
- Train a custom OCR model for industrial fonts
- Apply stronger data augmentation
- Optimize inference using ONNX or TensorRT
- Deploy on embedded platforms

---

## 🧑‍💻 Author
Ravi Kanani

---

## 📜 License
Educational and evaluation use only.
"@ | Out-File -Encoding UTF8 README.md
