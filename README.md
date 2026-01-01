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
