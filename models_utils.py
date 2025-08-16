from transformers import pipeline

# ---------------------------
# BLIP Image Captioning via Hugging Face Inference API
# ---------------------------
def load_blip():
    blip_pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=-1  # CPU only, safe for Streamlit Cloud
    )
    return blip_pipe, "cpu"  # keep return signature similar

# ---------------------------
# Stable Diffusion Text-to-Image via Hugging Face Inference API
# ---------------------------
def load_sd():
    sd_pipe = pipeline(
        "text-to-image",
        model="runwayml/stable-diffusion-v1-5",
        device=-1  # CPU only
    )
    return sd_pipe, "cpu"
