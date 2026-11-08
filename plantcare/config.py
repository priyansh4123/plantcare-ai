"""Central configuration. Every tunable threshold lives here."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOCALE_DIR = ROOT / "locales"
MODEL_DIR = ROOT / "models"
CHECKPOINT_PATH = MODEL_DIR / "plant_classifier.pt"
CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class Thresholds:
    # --- image quality (simple measurements, tune for your camera/users) ---
    min_side_px: int = 224
    blur_laplacian_var: float = 60.0   # below this the image is treated as blurry
    dark_mean: float = 40.0            # mean grey level (0-255) below = too dark
    bright_mean: float = 225.0         # above = overexposed
    # --- plant / leaf gate (CLIP zero-shot probability mass) ---
    plant_prob: float = 0.50
    leaf_prob: float = 0.45
    # --- disease classifier decision rules ---
    min_confidence: float = 0.60       # below -> "Unable to determine"
    healthy_confidence: float = 0.80   # "healthy" needs stronger evidence than disease
    ambiguity_margin: float = 0.30     # top1 - top2 below this -> inconclusive
    species_min_confidence: float = 0.70
    symptom_cue: float = 0.70          # zero-shot symptom cue reporting threshold
    # --- model quality gate ---
    min_model_accuracy: float = 0.85   # below -> show an accuracy limitation warning


THRESHOLDS = Thresholds()
