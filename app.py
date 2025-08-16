import os
import io
from datetime import datetime
import streamlit as st
from PIL import Image
from gtts import gTTS
import numpy as np
import onnxruntime as ort
from transformers import BlipProcessor

# ---------------------------
# Directories
# ---------------------------
os.makedirs("assets", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------
# Load ONNX BLIP model
# ---------------------------
@st.cache_resource
def load_blip_onnx(onnx_path="models/blip_base.onnx"):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    session = ort.InferenceSession(onnx_path)
    return processor, session

processor, blip_session = load_blip_onnx()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🦾 Assistive Image/Text AI Tool (ONNX)")
st.caption("Image-to-Text and Text-to-Image generation with optional speech output.")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["🖼️ Describe Image", "🎨 Generate Image"])
tab_describe, tab_generate = tabs

# ---------------------------
# Helper functions
# ---------------------------
def tts_to_file(text, lang="en"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"assets/speech_{ts}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path

def pil_download_bytes(img: Image.Image, form="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=form)
    buf.seek(0)
    return buf

# ---------------------------
# TAB 1: Image Captioning (BLIP)
# ---------------------------
with tab_describe:
    st.subheader("Describe an Image")
    uploaded = st.file_uploader("Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])
    prompt_hint = st.text_input("Optional guidance", value="")
    do_tts = st.checkbox("Speak the description (gTTS)", value=True)
    lang = st.text_input("TTS language (ISO code)", value="en", max_chars=5)
    max_len = st.slider("Max words (approx.)", 5, 60, 30)

    if st.button("Describe Image"):
        if not uploaded:
            st.warning("⚠️ Please upload an image first.")
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input image", use_column_width=True)

            # Preprocess
            inputs = processor(images=image, text=prompt_hint if prompt_hint else None, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items()}

            # ONNX inference
            outputs = blip_session.run(None, ort_inputs)
            logits = outputs[0]

            # Decode
            caption = processor.decode(logits[0], skip_special_tokens=True)

            st.success("Description:")
            st.write(f"**{caption}**")

            if do_tts:
                try:
                    mp3_path = tts_to_file(caption, lang=lang)
                    with open(mp3_path, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                    st.download_button(
                        "⬇️ Download audio (MP3)",
                        data=open(mp3_path, "rb"),
                        file_name=os.path.basename(mp3_path)
                    )
                except Exception as e:
                    st.info(f"TTS skipped: {e}")

# ---------------------------
# TAB 2: Text-to-Image (Stable Diffusion placeholder)
# ---------------------------
with tab_generate:
    st.subheader("Generate an Image from Text (ONNX placeholder)")
    prompt = st.text_area("Prompt", value="A cozy reading nook by a window, photorealistic")
    steps = st.slider("Inference steps", 5, 50, 25)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)
    seed_opt = st.checkbox("Set seed for reproducibility", value=False)
    seed_val = st.number_input("Seed", value=42, step=1) if seed_opt else None
    gen_tts = st.checkbox("Speak prompt (gTTS)", value=False)
    lang2 = st.text_input("TTS language for prompt", value="en", max_chars=5)

    colA, colB = st.columns([1, 1])
    with colA:
        width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    with colB:
        height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)

    if st.button("Generate Image"):
        st.info("⚠️ Text-to-Image ONNX model not yet exported. Placeholder only.")
        if gen_tts:
            try:
                mp3p = tts_to_file(f"Generated image for prompt: {prompt}", lang=lang2)
                with open(mp3p, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                st.download_button(
                    "⬇️ Download prompt audio (MP3)",
                    data=open(mp3p, "rb"),
                    file_name=os.path.basename(mp3p)
                )
            except Exception as e:
                st.info(f"TTS skipped: {e}")

# ---------------------------
# Footer
# ---------------------------
st.caption("Models: BLIP (ONNX). Stable Diffusion ONNX placeholder. gTTS for speech.")
