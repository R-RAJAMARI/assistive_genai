import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# Create model folders
os.makedirs("models/blip", exist_ok=True)
os.makedirs("models/sd", exist_ok=True)

print("✅ Model folders created (if not already exist)")
import torch
device = torch.device("cpu")
# Download BLIP-base
print("⏳ Downloading BLIP-base model...")
BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir="./models/blip"
)
BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir="./models/blip"
)
print("✅ BLIP-base downloaded to ./models/blip")

# Download Stable Diffusion v1-4
print("⏳ Downloading Stable Diffusion v1-4 model...")
StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # or another public model
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    cache_dir="./models/sd"
)

print("✅ Stable Diffusion downloaded to ./models/sd")
print("🎉 All models downloaded successfully!")
