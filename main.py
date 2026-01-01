"""
MAIN INFERENCE SCRIPT
=====================
Uses trained models:
- Faster R-CNN (Human / Animal detection)
- ResNet50 (Human vs Animal classification)
- EasyOCR (Industrial / Stenciled Text OCR)

Author: Ravi Kanani
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import easyocr
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DETECTION_MODEL_PATH = "models/detection_model.pth"
CLASSIFICATION_MODEL_PATH = "models/classification_model.pth"
CONF_THRESHOLD = 0.5


# ============================================================
# PART A: HUMAN–ANIMAL DETECTION SYSTEM
# ============================================================

class HumanAnimalDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")

        self.detector = self._load_detection_model()
        self.classifier = self._load_classification_model()

    def _load_detection_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes=3  # background + human + animal
        )
        model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=self.device))
        model.to(self.device).eval()
        print("✅ Detection model loaded")
        return model

    def _load_classification_model(self):
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Human / Animal
        model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=self.device))
        model.to(self.device).eval()
        print("✅ Classification model loaded")
        return model

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        print(f"\n🎥 Processing video: {video_path.name}")

        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Frames"):
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = F.to_tensor(rgb).to(self.device)
                detections = self.detector([img_tensor])[0]

                for box, score in zip(detections["boxes"], detections["scores"]):
                    if score < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    crop = rgb[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_pil = Image.fromarray(crop).resize((224, 224))
                    crop_tensor = F.to_tensor(crop_pil).unsqueeze(0).to(self.device)
                    output = self.classifier(crop_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = ["Human", "Animal"][pred]

                    color = (0, 255, 0) if label == "Human" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {score:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                writer.write(frame)

        cap.release()
        writer.release()
        print(f"✅ Saved video to {output_path}")


# ============================================================
# PART B: INDUSTRIAL OCR SYSTEM (RESTORED)
# ============================================================

class IndustrialOCRSystem:
    def __init__(self, languages=['en']):
        print("📝 Initializing Offline OCR System...")
        self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return binary

    def extract_text(self, image_path, output_json_path=None, output_image_path=None):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        preprocessed = self.preprocess_image(image)
        results = self.reader.readtext(preprocessed, detail=1)

        output = {
            "image_path": str(image_path),
            "detections": []
        }

        annotated = image.copy()

        for bbox, text, confidence in results:
            # Convert bbox to int
            bbox_int = [[int(x), int(y)] for x, y in bbox]

            # Save JSON data
            output["detections"].append({
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox_int
            })

            # 🔹 DRAW BOX (GREEN)
            pts = np.array(bbox_int, dtype=np.int32)
            cv2.polylines(
                annotated,
                [pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # 🔹 DRAW TEXT (RED)
            x, y = pts[0]
            cv2.putText(
                annotated,
                text,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ✅ Save JSON
        if output_json_path:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(output, f, indent=2)

        # ✅ Save annotated image (ROBUST)
        if output_image_path:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(output_image_path), annotated)

            print("🖼️ Saving annotated image to:", output_image_path)
            print("✅ cv2.imwrite success:", success)



        return output



# ============================================================
# MAIN (OPTIONAL CLI USE)
# ============================================================

def main():
    print("=" * 70)
    print("HUMAN–ANIMAL DETECTION & INDUSTRIAL OCR (INFERENCE)")
    print("=" * 70)

    detector = HumanAnimalDetector()

    for video in Path("test_videos").glob("*.mp4"):
        output = Path("outputs/annotated_videos") / f"{video.stem}_output.mp4"
        detector.process_video(video, output)

    ocr = IndustrialOCRSystem()
    image_dir = Path("datasets/part_b/test")
    print("📂 OCR image directory:", image_dir.resolve())
    print("📸 Images found:", list(image_dir.glob("*.*")))

    for img in image_dir.glob("*.jpg"):
        ocr.extract_text(
            image_path=img,
            output_json_path=Path("outputs/ocr_results") / f"{img.stem}.json",
            output_image_path=Path("outputs/ocr_results") / f"{img.stem}_annotated.jpg"
        )

    for img in image_dir.glob("*.png"):
        ocr.extract_text(
            image_path=img,
            output_json_path=Path("outputs/ocr_results") / f"{img.stem}.json",
            output_image_path=Path("outputs/ocr_results") / f"{img.stem}_annotated.jpg"
        )

    for img in image_dir.glob("*.jpeg"):
        ocr.extract_text(
            image_path=img,
            output_json_path=Path("outputs/ocr_results") / f"{img.stem}.json",
            output_image_path=Path("outputs/ocr_results") / f"{img.stem}_annotated.jpg"
        )


    print("\n🎉 Inference completed!")


if __name__ == "__main__":
    main()
