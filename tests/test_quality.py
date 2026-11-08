from PIL import Image

from plantcare.quality import assess_quality
from helpers import blurry_image, dark_image, sharp_image


def test_sharp_image_passes():
    assert assess_quality(sharp_image()).ok


def test_blurry_image_flagged():
    assert "blurry" in assess_quality(blurry_image()).issues


def test_dark_image_flagged():
    assert "too_dark" in assess_quality(dark_image()).issues


def test_tiny_image_flagged():
    assert "too_small" in assess_quality(Image.new("RGB", (64, 64), (0, 120, 0))).issues
