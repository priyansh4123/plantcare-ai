"""Disease classifier inference (calibrated softmax over trained classes)."""
from dataclasses import dataclass, field

import numpy as np

from .config import CHECKPOINT_PATH
from .labels import parse_label
from .schemas import Candidate


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained checkpoint is available."""


@dataclass
class Prediction:
    top: list                                   # list[Candidate], best first
    species_probs: dict = field(default_factory=dict)   # species display name -> summed prob

    @classmethod
    def from_probs(cls, class_names: list, probs, k: int = 5) -> "Prediction":
        probs = np.asarray(probs, dtype=float)
        order = np.argsort(-probs)
        top, species = [], {}
        for i in order[:k]:
            lab = parse_label(class_names[i])
            top.append(Candidate(lab.raw, lab.species, lab.condition, float(probs[i])))
        for name, p in zip(class_names, probs):
            sp = parse_label(name).species
            species[sp] = species.get(sp, 0.0) + float(p)
        return cls(top=top, species_probs=species)


class DiseaseClassifier:
    """Loads a checkpoint produced by train.py and predicts class probabilities."""

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device: str = None):
        import torch
        from .model import build_model, eval_transform

        path = checkpoint_path
        if not path.exists():
            raise ModelNotFoundError(f"No model checkpoint at {path}. Run train.py first.")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.class_names = ckpt["class_names"]
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.metrics = ckpt.get("metrics", {})
        self.model = build_model(ckpt["arch"], len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform()

    @property
    def supported_species(self) -> list:
        return sorted({parse_label(c).species for c in self.class_names})

    def predict(self, image) -> Prediction:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(x)[0] / self.temperature
            probs = self._torch.softmax(logits, dim=0).cpu().numpy()
        return Prediction.from_probs(self.class_names, probs)
