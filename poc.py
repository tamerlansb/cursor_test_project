import csv
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import clip

def download_image(url, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    return path

def compute_clip_similarity(image_path, caption, model, preprocess, device):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize([caption]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        return similarity
    except Exception as e:
        print(f"CLIP error on {image_path}: {e}")
        return None

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    rows = []
    with open("data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["image_url"]
            caption = row["caption"]

            path = download_image(url)
            if path:
                sim = compute_clip_similarity(path, caption, model, preprocess, device)
            else:
                sim = None

            row["image_path"] = path
            row["similarity"] = sim
            rows.append(row)

    with open("output.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
