🧠 AI Technical Assignment – Computer Vision & OCR (Offline)
📌 Overview

This project implements a fully offline Computer Vision and OCR system designed for industrial and hardware-constrained environments.
The solution avoids cloud APIs and works entirely offline, making it suitable for edge or laptop deployment.

The system includes:

Human & Animal Detection (Video-based)

Offline OCR for Industrial / Stenciled Text (Image-based)

📂 Project Structure
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

📄 Assignment Deliverable – Single Python Script

This project uses a single consolidated Python script (main.py) that contains both explanation and implementation.

The script includes:

Well-commented step-by-step explanations

Dataset and model selection justification

Training, inference, and OCR pipeline explanation

Challenges faced and possible improvements

The goal is to keep the solution readable, reproducible, and suitable for real-world industrial deployment.

📊 Dataset Sources & Download Instructions
Part A – Human & Animal Detection

Dataset Source:
Open Images Dataset V7
https://storage.googleapis.com/openimages/web/index.html

Selected Classes:

Person

Dog

Cat

Horse

Elephant

Bear

Download Commands Used:

pip install openimages

oid_v7_download \
--classes Person Dog Cat Horse Elephant Bear \
--type_data train validation \
--limit 5000 \
--dest datasets/part_a/


Annotations are used in Pascal VOC (XML) format.

Part B – Industrial OCR Dataset

The OCR dataset consists of industrial images containing:

Painted or stenciled text

Faded markings

Low contrast text

Surface damage

The entire OCR pipeline works fully offline.

🧠 Model Selection & Justification

Detection Model:

Faster R-CNN with ResNet-50 backbone

Chosen for accurate localization and robustness

Classification Model:

ResNet-50 for Human vs Animal classification

OCR Model:

EasyOCR

Selected for offline operation and robustness on degraded text

🏋️ Training Pipeline

Transfer learning with pretrained backbones

Dataset size reduction for limited GPU memory

Batch size tuning to prevent CUDA out-of-memory errors

Backbone freezing during fine-tuning

Metrics logging using Weights & Biases (wandb)

🎥 Inference Pipeline – Part A

Videos placed in test_videos/

Faster R-CNN detects humans and animals

ResNet-50 classifies detected objects

Annotated videos saved to outputs/annotated_videos/

📝 OCR Pipeline – Part B

OpenCV preprocessing (grayscale, denoising, CLAHE, thresholding)

EasyOCR text detection

Bounding box extraction

Outputs generated:

Structured JSON

Annotated image with bounding boxes

outputs/ocr_results/
├── sample.json
└── sample_annotated.jpg

⚠️ Challenges Faced

Limited GPU memory on laptop hardware

CUDA out-of-memory issues during training

OCR accuracy drop on faded industrial text

Annotation inconsistencies in open datasets

🛠️ Possible Improvements

Use lighter detection models for edge devices

Train a custom OCR model for industrial fonts

Apply stronger data augmentation

Optimize inference using ONNX or TensorRT

Deploy on embedded or edge AI platforms

🧑‍💻 Author

Ravi Kanani

📜 License

Educational and evaluation use only.

✅ What to do now

Paste this into README.md

Save the file

Run:

git add README.md
git commit -m "Add complete project README"
git push