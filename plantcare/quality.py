"""Image-quality checks (blur, exposure, resolution) using NumPy only.

These are measurements of the *photo*, not rules for deciding disease.
Thresholds are in config.Thresholds and should be tuned on real user photos.
"""
import numpy as np
from PIL import Image

from .config import THRESHOLDS, Thresholds
from .schemas import QualityReport

_ANALYSIS_SIDE = 512


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian: low values mean few sharp edges."""
    g = gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def assess_quality(img: Image.Image, t: Thresholds = THRESHOLDS) -> QualityReport:
    w, h = img.size
    issues = []
    if min(w, h) < t.min_side_px:
        issues.append("too_small")

    scale = _ANALYSIS_SIDE / max(w, h)
    small = img.convert("L").resize((max(3, round(w * scale)), max(3, round(h * scale))))
    gray = np.asarray(small)
    sharp = laplacian_variance(gray)
    mean = float(gray.mean())

    if sharp < t.blur_laplacian_var:
        issues.append("blurry")
    if mean < t.dark_mean:
        issues.append("too_dark")
    elif mean > t.bright_mean:
        issues.append("overexposed")

    return QualityReport(
        ok=not issues,
        issues=issues,
        metrics={"width": w, "height": h, "sharpness": round(sharp, 1), "brightness": round(mean, 1)},
    )
