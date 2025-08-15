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
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def load_blip():
    device = "cpu"
    processor = BlipProcessor.from_pretrained(
        r".\models\blip\models--Salesforce--blip-image-captioning-base\snapshots\82a37760796d32b1411fe092ab5d4e227313294b",  
        local_files_only=True
    )

    model = BlipForConditionalGeneration.from_pretrained(
        r".\models\blip\models--Salesforce--blip-image-captioning-base\snapshots\82a37760796d32b1411fe092ab5d4e227313294b",    # local folder
        local_files_only=True
    ).to(device)

    return processor, model, device

# ---------------------------
# Load Stable Diffusion from local folder
# ---------------------------
def load_sd():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        r".\models\sd\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        torch_dtype=dtype,
        safety_checker=None,
        local_files_only=True,
        device_map=None 
    )
    if device == "cuda":
        pipe = pipe.to("cuda")
    return pipe, device
