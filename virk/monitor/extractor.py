from abc import ABC, abstractmethod
import torch
from pathlib import Path
from PIL import Image
from typing import List, Union
import numpy as np

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    timm = None


class EmbeddingExtractor(ABC):
    """Abstract base class for model embedding extractors."""

    @abstractmethod
    def extract(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Args:
            images: List of file paths, PIL Images, or numpy arrays.

        Returns:
            np.ndarray: Matrix of shape (N, D) where N is batch size and D is embedding dim.
        """
        pass


class TimmExtractor(EmbeddingExtractor):
    """Wrapper for TIMM models to extract embeddings."""

    def __init__(
        self, model_name: str = "resnet50", pretrained: bool = True, device: str = "cpu"
    ):
        if timm is None:
            raise ImportError(
                "timm is not installed. Please install it with `pip install timm`."
            )

        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )  # num_classes=0 for features
        self.model.to(self.device)
        self.model.eval()

        config = resolve_data_config({}, model=self.model)
        self.data_config = config
        self.transform = create_transform(**config)

    def _load_image(self, img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img, str):
            with Image.open(img) as source:
                img = source.convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")

        return self.transform(img)

    def extract(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        if not images:
            raise ValueError("images cannot be empty")
        batch_tensors = torch.stack([self._load_image(img) for img in images]).to(
            self.device
        )

        with torch.no_grad():
            features = self.model(batch_tensors)

        return features.cpu().numpy()

    def describe(self):
        """Serializable model identity and evaluation preprocessing."""
        from importlib.metadata import version

        return {
            "class": type(self).__name__,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "timm_version": version("timm"),
            "torch_version": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "num_features": self.model.num_features,
            "preprocessing": self.data_config,
            "pretrained_config": self.model.pretrained_cfg,
        }

    def save_model_state(self, path):
        """Persist exact weights, including randomly initialized/custom models."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                name: tensor.detach().cpu()
                for name, tensor in self.model.state_dict().items()
            },
            path,
        )
        return str(path)
