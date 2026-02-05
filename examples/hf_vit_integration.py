"""
Example: Hugging Face ViT integration for VIRK.

Quick install:
  pip install transformers torch torchvision

Notes:
  - Uses CLS token from last hidden state as embedding.
  - Replace the model checkpoint with your production model.
"""

from typing import List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    AutoModel = None

from virk.integration.middleware import VirkMiddleware


class HFViTExtractor:
    def __init__(self, model_id: str = "google/vit-base-patch16-224", device: str = "cpu"):
        if AutoModel is None:
            raise ImportError("transformers is not installed. Run: pip install transformers")
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _load_images(self, images: List[Union[str, Path]]) -> List["Image.Image"]:
        imgs = []
        for p in images:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        return imgs

    def extract(self, images: List[Union[str, Path]]) -> np.ndarray:
        imgs = self._load_images(images)
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
        return cls.cpu().numpy()


def main():
    # 1. Reference data
    ref_paths = ["./data/ref_01.jpg", "./data/ref_02.jpg"]
    extractor = HFViTExtractor(model_id="google/vit-base-patch16-224", device="cpu")
    ref_embs = extractor.extract(ref_paths)

    # 2. Create monitor
    monitor = VirkMiddleware.create(
        reference_embeddings=ref_embs,
        extractor=extractor,
        reference_paths=ref_paths,
        output_dir="./incidents",
        metrics_type="prometheus",
        async_processing=True,
    )

    # 3. Inference loop example
    prod_paths = ["./data/prod_01.jpg", "./data/prod_02.jpg"]
    prod_embs = extractor.extract(prod_paths)
    metadata = [{"camera_id": "cam_01"}, {"camera_id": "cam_02"}]
    monitor.process_batch(prod_embs, image_paths=prod_paths, metadata=metadata)


if __name__ == "__main__":
    main()
