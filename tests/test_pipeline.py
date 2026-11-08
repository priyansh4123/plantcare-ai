from plantcare.i18n import t
from plantcare.pipeline import Status
from helpers import blurry_image, make_pipeline, sharp_image

# probabilities over: Tomato healthy, Tomato Early blight, Tomato Late blight, Potato healthy
HEALTHY = [0.95, 0.02, 0.02, 0.01]
EARLY_BLIGHT = [0.02, 0.93, 0.03, 0.02]


def test_non_plant_image_rejected_with_exact_message():
    res = make_pipeline(HEALTHY, verdict="not_plant").analyze(sharp_image())
    assert res.status == Status.NOT_A_PLANT
    assert ("No recognizable plant or leaf detected. "
            "Please upload a clear image of a plant or leaf.") in res.messages
    assert res.condition is None and res.confidence is None


def test_plant_without_visible_leaf_is_unsuitable():
    res = make_pipeline(HEALTHY, verdict="leaf_not_visible").analyze(sharp_image())
    assert res.status == Status.LEAF_NOT_VISIBLE
    assert "unsuitable" in res.messages[0]


def test_blurry_image_requests_clearer_photo():
    res = make_pipeline(HEALTHY).analyze(blurry_image())
    assert res.status == Status.LOW_QUALITY
    assert any("clearer photograph" in m for m in res.messages)


def test_healthy_leaf():
    res = make_pipeline(HEALTHY).analyze(sharp_image())
    assert res.status == Status.HEALTHY
    assert ("No obvious signs of disease detected. "
            "The plant appears healthy based on the image.") in res.messages
    assert res.species == "Tomato"


def test_diseased_leaf_reports_condition_and_confidence():
    res = make_pipeline(EARLY_BLIGHT, cues=("brown_spots",)).analyze(sharp_image())
    assert res.status == Status.POSSIBLE_DISEASE
    assert res.condition == "Early blight" and abs(res.confidence - 0.93) < 1e-9
    assert res.disease_info and res.symptoms[0].key == "brown_spots"
    assert "preliminary" in res.disclaimer


def test_low_confidence_is_unable_to_determine():
    res = make_pipeline([0.35, 0.30, 0.20, 0.15]).analyze(sharp_image())
    assert res.status == Status.UNABLE_TO_DETERMINE
    assert res.condition is None  # no invented diagnosis


def test_close_candidates_are_inconclusive():
    res = make_pipeline([0.02, 0.62, 0.33, 0.03]).analyze(sharp_image())
    assert res.status == Status.INCONCLUSIVE
    assert res.condition is None and len(res.candidates) == 3


def test_healthy_prediction_with_visible_symptom_cues_is_not_called_healthy():
    res = make_pipeline(HEALTHY, cues=("yellowing",)).analyze(sharp_image())
    assert res.status == Status.INCONCLUSIVE


def test_weakly_healthy_is_not_called_healthy():
    res = make_pipeline([0.70, 0.10, 0.10, 0.10]).analyze(sharp_image())
    assert res.status == Status.INCONCLUSIVE


def test_unreadable_file_gives_friendly_error():
    res = make_pipeline(HEALTHY).analyze(b"this is not an image")
    assert res.status == Status.ERROR and "could not be read" in res.messages[0]


def test_missing_model_is_reported_not_crashed():
    res = make_pipeline(None).analyze(sharp_image())
    assert res.status == Status.ERROR and "not installed" in res.messages[0]


def test_low_model_accuracy_is_communicated():
    res = make_pipeline(EARLY_BLIGHT, metrics={"test_accuracy": 0.6}).analyze(sharp_image())
    assert any("60%" in w for w in res.warnings)
