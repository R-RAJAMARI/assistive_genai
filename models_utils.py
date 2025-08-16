# models_utils.py
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# ---------------------------
# BLIP: Image captioning pipeline
# ---------------------------
def load_blip():
    """
    Returns:
        blip_pipe: HuggingFace pipeline for image-to-text
        device: "cpu"
    """
    blip_pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=-1  # CPU only
    )
    return blip_pipe, "cpu"

# ---------------------------
# Stable Diffusion: Text-to-image pipeline
# ---------------------------
def load_sd():
    """
    Returns:
        sd_pipe: Diffusers Stable Diffusion pipeline (CPU)
        device: "cpu"
    """
    device = "cpu"
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    sd_pipe = sd_pipe.to(device)
    return sd_pipe, device
