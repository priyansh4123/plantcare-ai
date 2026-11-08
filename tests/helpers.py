"""Shared test helpers: synthetic images and fake model components (no torch needed)."""
import numpy as np
from PIL import Image, ImageFilter

from plantcare.classifier import Prediction
from plantcare.pipeline import DiagnosisPipeline
from plantcare.schemas import GateResult, SymptomCue

CLASS_NAMES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Potato___healthy"]


def sharp_image(size=600) -> Image.Image:
    rng = np.random.RandomState(0)
    arr = rng.randint(40, 200, (size, size, 3)).astype("uint8")
    arr[..., 0] //= 2
    arr[..., 2] //= 2
    return Image.fromarray(arr, "RGB")


def blurry_image() -> Image.Image:
    return sharp_image().filter(ImageFilter.GaussianBlur(25))


def dark_image() -> Image.Image:
    return Image.fromarray((np.asarray(sharp_image()) * 0.08).astype("uint8"), "RGB")


class FakeGate:
    def __init__(self, verdict="ok"):
        self.verdict = verdict

    def check(self, image):
        return GateResult(self.verdict, plant_prob=0.9 if self.verdict != "not_plant" else 0.1,
                          leaf_prob=0.8 if self.verdict == "ok" else 0.1)


class FakeClassifier:
    def __init__(self, probs):
        self.pred = Prediction.from_probs(CLASS_NAMES, probs)

    def predict(self, image):
        return self.pred


class FakeDetector:
    def __init__(self, keys=()):
        self.keys = keys

    def detect(self, image):
        return [SymptomCue(k, 0.9) for k in self.keys]


def make_pipeline(probs=None, verdict="ok", cues=(), metrics=None):
    clf = FakeClassifier(probs) if probs is not None else None
    return DiagnosisPipeline(FakeGate(verdict), clf, FakeDetector(cues), metrics)
