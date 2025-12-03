import asyncio
import aiohttp
import os
import csv
import logging
from PIL import Image
import torch
import clip

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# ------------------------------
# Async image downloader
# ------------------------------
async def download_image(session, url, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    path = os.path.join(save_dir, filename)
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            content = await resp.read()
            with open(path, "wb") as f:
                f.write(content)
        logging.info(f"Downloaded {url}")
        return path
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return None

async def download_all_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# ------------------------------
# CLIP similarity
# ------------------------------
def compute_clip_similarity(image_path, caption, model, preprocess, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).item()
    except Exception as e:
        logging.error(f"CLIP error on {image_path}: {e}")
        return None

# ------------------------------
# Main async runner
# ------------------------------
async def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Read CSV
    rows = []
