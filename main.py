import os
import requests
from PIL import Image
import torch
import clip
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Inference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def download_image(self, url, save_dir="images"):
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
                logging.error(f"Failed to download {url}: {e}")
                return None

        return path

    def compute_clip_similarity(self, image_path, caption):
        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            text = clip.tokenize([caption]).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()
            return similarity
        except Exception as e:
            logging.error(f"CLIP error on {image_path}: {e}")
            return None

    def run(self, input_csv="data.csv", output_csv="output.csv"):
        rows = []
        with open(input_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row["image_url"]
                caption = row["caption"]

                path = self.download_image(url)
                if path:
                    logging.info(f"Downloaded image from {url} to {path}")
                    sim = self.compute_clip_similarity(path, caption)
                else:
                    logging.warning(f"Could not download image from {url}")
                    sim = None

                row["image_path"] = path
                row["similarity"] = sim
                rows.append(row)

        if rows:
            with open(output_csv, "w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logging.info(f"Results written to {output_csv}")
        else:
            logging.warning("No data to write to output CSV.")

if __name__ == "__main__":
    logging.info("Starting inference process.")
    inference = Inference()
    inference.run()
    logging.info("Inference process completed.")
