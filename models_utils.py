import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# ---------------------------
# Create folders if they don't exist
# ---------------------------
os.makedirs("models/blip", exist_ok=True)
os.makedirs("models/sd", exist_ok=True)

# ---------------------------
# Load BLIP from local folder
# ---------------------------

def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


# ---------------------------
# Load Stable Diffusion from local folder
# ---------------------------
from diffusers import StableDiffusionPipeline
import torch

def load_sd():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    return pipe, device
