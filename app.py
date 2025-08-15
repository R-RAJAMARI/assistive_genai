import os
import io
from datetime import datetime
import torch
import streamlit as st
from PIL import Image
from gtts import gTTS

# ---------------------------
# Import model-loading functions
# ---------------------------
from models_utils import load_blip, load_sd


# ---------------------------
# Preload models at startup (cached)
# ---------------------------
@st.cache_resource
def init_models():
    processor, blip, blip_device = load_blip()
    pipe, sd_device = load_sd()
    return processor, blip, blip_device, pipe, sd_device


with st.spinner("Loading models… this may take a while (~30–90s first time)"):
    processor, blip, blip_device, pipe, sd_device = init_models()
st.success("✅ Models loaded and ready!")


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

tab1, tab2 = st.tabs(["🖼️ Describe Image", "🎨 Generate Image"])


# ---------------------------
# TAB 1: Image Captioning
# ---------------------------
with tab1:
    st.subheader("Describe an Image")

    colL, colR = st.columns([1, 1])
    with colL:
        uploaded = st.file_uploader("Upload an image (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp"])
        prompt_hint = st.text_input("(Optional) Add guidance (e.g., 'describe details and colors')", value="")
        do_tts = st.checkbox("Speak the description (gTTS)", value=True)
        lang = st.text_input("TTS language (ISO code, e.g., 'en', 'hi')", value="en", max_chars=5)
        max_len = st.slider("Max words (approx.)", 5, 60, 30, help="Controls caption length")

        if st.button("Describe"):
            if not uploaded:
                st.warning("⚠️ Please upload an image first.")
            else:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Input image", use_column_width=True)

                inputs = processor(images=image, text=prompt_hint if prompt_hint else None, return_tensors="pt").to(blip_device)
                gen_kwargs = {"max_new_tokens": max_len, "num_beams": 5}

                with torch.inference_mode():
                    out = blip.generate(**inputs, **gen_kwargs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                st.success("Description:")
                st.write(f"**{caption}**")

                if do_tts:
                    try:
                        mp3_path = tts_to_file(caption, lang=lang)
                        with open(mp3_path, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        st.download_button("⬇️ Download audio (MP3)",
                                           data=open(mp3_path, "rb"),
                                           file_name=os.path.basename(mp3_path))
                    except Exception as e:
                        st.info(f"TTS skipped: {e}")

    with colR:
        st.markdown("### 💡 Tips for Interviews")
        st.markdown("- Try with random photos to show robustness. \n"
                    "- Use the guidance box to demonstrate controllability. \n"
                    "- Toggle TTS to highlight accessibility focus.")


# ---------------------------
# TAB 2: Text-to-Image Generation
# ---------------------------
with tab2:
    st.subheader("Generate an Image from Text")

    prompt = st.text_area("Prompt", value="a cozy reading nook by a window, soft morning light, photorealistic")
    steps = st.slider("Inference steps", 5, 50, 25)
    guidance = st.slider("Guidance scale (CFG)", 1.0, 15.0, 7.5)
    seed_opt = st.checkbox("Set seed for reproducibility", value=False)
    seed_val = st.number_input("Seed", value=42, step=1) if seed_opt else None
    gen_tts = st.checkbox("Speak back the prompt (gTTS)", value=False)
    lang2 = st.text_input("TTS language for prompt (ISO code)", value="en", max_chars=5)

    colA, colB = st.columns([1, 1])
    with colA:
        width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    with colB:
        height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt.")
        else:
            if seed_opt:
                generator = torch.Generator(device=sd_device).manual_seed(int(seed_val))
            else:
                generator = None

            try:
                image = pipe(
                    prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance),
                    width=int(width),
                    height=int(height),
                    generator=generator
                ).images[0]

                st.image(image, caption="Generated image", use_column_width=True)

                # Download button
                png_bytes = pil_download_bytes(image, "PNG")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("⬇️ Download image (PNG)", data=png_bytes, file_name=f"generated_{ts}.png")

                if gen_tts:
                    try:
                        mp3p = tts_to_file(f"Generated image for prompt: {prompt}", lang=lang2)
                        with open(mp3p, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        st.download_button("⬇️ Download prompt audio (MP3)",
                                           data=open(mp3p, "rb"),
                                           file_name=os.path.basename(mp3p))
                    except Exception as e:
                        st.info(f"TTS skipped: {e}")

            except torch.cuda.OutOfMemoryError:
                st.error("❌ CUDA out of memory. Try smaller dimensions, fewer steps, or use a GPU with more VRAM.")
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")


# ---------------------------
# Footer
# ---------------------------
st.caption("Models used: BLIP (Salesforce/blip-image-captioning-large), Stable Diffusion v1-5 (runwayml/stable-diffusion-v1-5).")
