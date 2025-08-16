# app.py
import os
import io
from datetime import datetime
from PIL import Image
import streamlit as st
from gtts import gTTS
from models_utils import load_blip, load_sd

# Disable file watcher issues
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# ---------------------------
# Load models at startup
# ---------------------------
@st.cache_resource
def init_models():
    blip_pipe, blip_device = load_blip()
    sd_pipe, sd_device = load_sd()
    return blip_pipe, blip_device, sd_pipe, sd_device

with st.spinner("Loading models…"):
    blip_pipe, blip_device, sd_pipe, sd_device = init_models()

st.success("✅ Models loaded!")

# ---------------------------
# Helper functions
# ---------------------------
def tts_to_file(text, lang="en"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"assets/speech_{ts}.mp3"
    os.makedirs("assets", exist_ok=True)
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path

def pil_download_bytes(img: Image.Image, form="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=form)
    buf.seek(0)
    return buf

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🦾 Image-to-Text & Text-to-Image Assistive Tool")
st.caption("Describe any image in natural language or generate images from text — with optional speech output.")

tabs = ["🖼️ Describe Image", "🎨 Generate Image"]
tab_objects = st.tabs(tabs)

# ---------------------------
# TAB 1: Image Captioning
# ---------------------------
with tab_objects[0]:
    st.subheader("Describe an Image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    prompt_hint = st.text_input("(Optional) Add guidance", value="")
    do_tts = st.checkbox("Speak the description (gTTS)", value=True)
    lang = st.text_input("TTS language (ISO code, e.g., 'en')", value="en", max_chars=5)
    max_len = st.slider("Max words (approx.)", 5, 60, 30)

    if st.button("Describe"):
        if not uploaded:
            st.warning("⚠️ Please upload an image first.")
        else:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input image", use_column_width=True)

            # BLIP inference
            caption = blip_pipe(image)[0]["generated_text"]
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
# TAB 2: Text-to-Image
# ---------------------------
with tab_objects[1]:
    st.subheader("Generate an Image from Text")
    prompt = st.text_area("Prompt", value="a cozy reading nook by a window, photorealistic")
    steps = st.slider("Inference steps", 5, 50, 25)
    guidance = st.slider("Guidance scale (CFG)", 1.0, 15.0, 7.5)
    seed_opt = st.checkbox("Set seed for reproducibility", value=False)
    seed_val = st.number_input("Seed", value=42, step=1) if seed_opt else None
    gen_tts = st.checkbox("Speak back the prompt (gTTS)", value=False)
    lang2 = st.text_input("TTS language for prompt", value="en", max_chars=5)

    colA, colB = st.columns([1,1])
    with colA:
        width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    with colB:
        height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        else:
            generator = None
            if seed_opt:
                generator = torch.Generator(device="cpu").manual_seed(int(seed_val))

            try:
                image = sd_pipe(
                    prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=generator
                ).images[0]

                st.image(image, caption="Generated image", use_column_width=True)
                png_bytes = pil_download_bytes(image, "PNG")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("⬇️ Download image (PNG)", data=png_bytes, file_name=f"generated_{ts}.png")

                if gen_tts:
                    mp3p = tts_to_file(f"Generated image for prompt: {prompt}", lang=lang2)
                    with open(mp3p, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                    st.download_button(
                        "⬇️ Download prompt audio (MP3)",
                        data=open(mp3p, "rb"),
                        file_name=os.path.basename(mp3p)
                    )

            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

st.caption("Models used: BLIP (Salesforce/blip-image-captioning-base), Stable Diffusion v1-5 (runwayml/stable-diffusion-v1-5)")
