"""CLIP-based validation of uploads and zero-shot symptom cues.

A separate, general-purpose vision-language model decides whether an image is a
plant/leaf at all. It is independent of the disease classifier, so non-plant
images (people, rooms, green cloth...) are rejected before any diagnosis.

torch/transformers are imported lazily so the rest of the package (and the unit
tests) work without them installed.
"""
import numpy as np

from .config import CLIP_MODEL, THRESHOLDS, Thresholds
from .schemas import GateResult, SymptomCue

GATE_PROMPTS = {
    "leaf": [
        "a close-up photo of a plant leaf",
        "a photo of a single green leaf showing its surface and veins",
        "a close-up photo of a diseased plant leaf",
        "a photo of a crop leaf held in a hand",
    ],
    "plant_other": [
        "a photo of a whole plant growing in a garden",
        "a photo of a potted houseplant",
        "a photo of a crop field",
        "a photo of fruit or vegetables on a plant",
    ],
    "not_plant": [
        "a photo of a person", "a photo of an animal", "a photo of a building",
        "a photo of a car or vehicle", "a photo of food on a plate",
        "a screenshot or a document with text", "a photo of green cloth or green paint",
        "a photo of an indoor room", "a photo of the sky", "a photo of a green plastic object",
    ],
}

SYMPTOM_PROMPTS = {
    "yellowing": "a plant leaf turning yellow",
    "brown_spots": "a leaf with brown spots",
    "dark_lesions": "a leaf with dark black lesions",
    "browning_edges": "a leaf with dry brown scorched edges",
    "wilting": "a wilting drooping leaf",
    "powdery_coating": "a leaf covered with white powdery coating",
    "insect_holes": "a leaf with holes chewed by insects",
    "curling": "a curled and distorted leaf",
}
_BASELINE = "a healthy green leaf"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ClipEngine:
    """Thin wrapper that scores an image against text prompts."""

    def __init__(self, model_name: str = CLIP_MODEL, device: str = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def logits(self, image, prompts: list) -> np.ndarray:
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model(**inputs)
        return out.logits_per_image[0].cpu().numpy()


class ClipPlantGate:
    """Decides: not a plant / plant without a clear leaf / usable leaf image."""

    def __init__(self, engine: ClipEngine, thresholds: Thresholds = THRESHOLDS):
        self.engine, self.t = engine, thresholds
        self._prompts, self._groups = [], []
        for group, prompts in GATE_PROMPTS.items():
            self._prompts += prompts
            self._groups += [group] * len(prompts)
        self._groups = np.array(self._groups)

    def check(self, image) -> GateResult:
        probs = _softmax(self.engine.logits(image, self._prompts))
        leaf = float(probs[self._groups == "leaf"].sum())
        plant = leaf + float(probs[self._groups == "plant_other"].sum())
        if plant < self.t.plant_prob:
            verdict = "not_plant"
        elif leaf < self.t.leaf_prob:
            verdict = "leaf_not_visible"
        else:
            verdict = "ok"
        return GateResult(verdict, plant_prob=plant, leaf_prob=leaf)


class ClipSymptomDetector:
    """Zero-shot visual cues (yellowing, spots...). Indicative only, never a diagnosis."""

    def __init__(self, engine: ClipEngine, thresholds: Thresholds = THRESHOLDS):
        self.engine, self.t = engine, thresholds
        self._keys = list(SYMPTOM_PROMPTS)
        self._prompts = [_BASELINE] + [SYMPTOM_PROMPTS[k] for k in self._keys]

    def detect(self, image) -> list:
        logits = self.engine.logits(image, self._prompts)
        base, cues = logits[0], []
        for key, logit in zip(self._keys, logits[1:]):
            score = float(_softmax(np.array([logit, base]))[0])
            if score >= self.t.symptom_cue:
                cues.append(SymptomCue(key, score))
        return sorted(cues, key=lambda c: -c.score)
