import streamlit as st
from pathlib import Path
from PIL import Image
import json
import imageio.v2 as imageio

# Import systems from main.py
from main import HumanAnimalDetector, IndustrialOCRSystem

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="CV & OCR System", layout="wide")
st.title("🤖 Computer Vision & OCR System (Browser Compatible)")

tab1, tab2 = st.tabs(
    ["Part A: Human–Animal Detection", "Part B: Industrial OCR"]
)

# ============================================================
# HELPER: CONVERT VIDEO TO BROWSER FORMAT
# ============================================================

def convert_to_browser_video(input_video, output_video):
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()["fps"]

    frames = []
    for frame in reader:
        frames.append(frame)

    imageio.mimsave(
        output_video,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p"
    )

# ============================================================
# PART A: VIDEO DETECTION
# ============================================================

with tab1:
    st.header("🎯 Human & Animal Detection (Inference)")
    col1, col2 = st.columns(2)

    # ---------- LEFT: INPUT ----------
    with col1:
        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi"]
        )

        if uploaded_video:
            test_dir = Path("test_videos")
            test_dir.mkdir(exist_ok=True)

            video_path = test_dir / uploaded_video.name
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            st.subheader("📥 Input Video")
            st.video(str(video_path))

            if st.button("🚀 Run Detection"):
                with st.spinner("Running detection..."):
                    detector = HumanAnimalDetector()

                    out_dir = Path("outputs/annotated_videos")
                    out_dir.mkdir(parents=True, exist_ok=True)

                    raw_output = out_dir / f"{video_path.stem}_output.mp4"
                    browser_output = out_dir / f"{video_path.stem}_output_browser.mp4"

                    detector.process_video(video_path, raw_output)

                    st.info("🔄 Converting video for browser playback...")
                    convert_to_browser_video(raw_output, browser_output)

                st.success("✅ Detection & conversion completed")

    # ---------- RIGHT: OUTPUT ----------
    with col2:
        st.subheader("📺 Output Video (Browser Version)")

        if uploaded_video:
            browser_video = (
                Path("outputs/annotated_videos")
                / f"{Path(uploaded_video.name).stem}_output_browser.mp4"
            )

            if browser_video.exists():
                st.video(str(browser_video))
            else:
                st.info("Run detection to generate browser-playable output.")
        else:
            st.info("Upload a video to view output.")

# ============================================================
# PART B: INDUSTRIAL OCR
# ============================================================

with tab2:
    st.header("📝 Industrial OCR (Offline)")
    col1, col2 = st.columns(2)

    # ---------- LEFT ----------
    with col1:
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "png"]
        )

        if uploaded_image:
            img = Image.open(uploaded_image)
            st.subheader("📥 Input Image")
            st.image(img, use_column_width=True)

            img_dir = Path("datasets/part_b/test")
            img_dir.mkdir(parents=True, exist_ok=True)

            img_path = img_dir / uploaded_image.name
            img.save(img_path)

            if st.button("🔍 Extract Text"):
                with st.spinner("Running OCR..."):
                    ocr = IndustrialOCRSystem()

                    output_dir = Path("outputs/ocr_results")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    result = ocr.extract_text(
                        image_path=img_path,
                        output_json_path=output_dir / f"{img_path.stem}.json",
                        output_image_path=output_dir / f"{img_path.stem}_annotated.jpg"
                    )

                st.success("✅ OCR completed")
                st.image(str(output_dir / f"{img_path.stem}_annotated.jpg"))


    # ---------- RIGHT ----------
    with col2:
        st.subheader("📄 OCR Output")

        results_dir = Path("outputs/ocr_results")
        json_files = list(results_dir.glob("*.json"))

        if json_files:
            selected = st.selectbox(
                "Select OCR result",
                json_files
            )

            with open(selected, "r") as f:
                data = json.load(f)

            for det in data.get("detections", []):
                st.write(
                    f"**{det['text']}** — confidence: `{det['confidence']:.2f}`"
                )
        else:
            st.info("No OCR results yet.")
