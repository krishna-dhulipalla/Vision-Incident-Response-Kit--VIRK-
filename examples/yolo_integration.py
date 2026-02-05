"""
Example: YOLOv8 integration for VIRK.

Quick install:
  pip install ultralytics torch torchvision

Notes:
  - This example uses a forward hook to capture embeddings from a backbone layer.
  - The exact layer index can vary by model size; adjust `feature_layer_idx` as needed.
  - This is a practical template, not a production-validated extractor.
"""

from typing import List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from virk.integration.middleware import VirkMiddleware


class YoloV8EmbeddingExtractor:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", feature_layer_idx: int = -2):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model.model.eval()
        self.feature_layer_idx = feature_layer_idx
        self._features = None

        # Register forward hook on chosen layer
        layer = self._get_feature_layer()
        layer.register_forward_hook(self._hook)

    def _get_feature_layer(self):
        # Ultralytics models expose a ModuleList at model.model
        model_layers = getattr(self.model.model, "model", None)
        if model_layers is None:
            model_layers = self.model.model
        return model_layers[self.feature_layer_idx]

    def _hook(self, module, inputs, output):
        # Save pooled features for the current batch
        if isinstance(output, (list, tuple)):
            output = output[0]
        # Global average pool to (N, C)
        pooled = output.mean(dim=(2, 3)) if output.ndim == 4 else output
        self._features = pooled.detach().cpu().numpy()

    def extract(self, images: List[Union[str, Path]]) -> np.ndarray:
        # Trigger forward pass via predict (uses internal preprocessing)
        self._features = None
        self.model.predict(source=images, device=self.device, verbose=False)
        if self._features is None:
            raise RuntimeError("Failed to capture features. Check feature_layer_idx.")
        return self._features


def main():
    # 1. Reference data
    ref_paths = ["./data/ref_01.jpg", "./data/ref_02.jpg"]
    extractor = YoloV8EmbeddingExtractor(model_path="yolov8n.pt", device="cpu", feature_layer_idx=-2)
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
