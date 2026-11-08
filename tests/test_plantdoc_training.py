import json
from pathlib import Path

import pytest
from PIL import Image

from train_plantdoc import FIELD_CLASS_MAP, canonical_field_class, field_datasets


ROOT = Path(__file__).resolve().parents[1]


def test_plantdoc_labels_map_to_existing_browser_classes():
    classes = set(json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))["classes"])
    assert set(FIELD_CLASS_MAP.values()) <= classes


def test_plantdoc_folder_names_are_matched_case_insensitively():
    assert canonical_field_class("APPLE SCAB LEAF") == "Apple___Apple_scab"
    assert canonical_field_class(" grape leaf ") == "Grape___healthy"


def test_unknown_plantdoc_label_fails_explicitly():
    with pytest.raises(ValueError, match="Unmapped PlantDoc class folder"):
        canonical_field_class("unknown plant")


def test_field_split_excludes_test_duplicates_and_keeps_train_validation_disjoint(tmp_path):
    root = tmp_path / "PlantDoc-Dataset-master"
    train_scab = root / "train" / "Apple Scab Leaf"
    test_scab = root / "test" / "Apple Scab Leaf"
    train_healthy = root / "train" / "Apple leaf"
    test_healthy = root / "test" / "Apple leaf"
    for folder in (train_scab, test_scab, train_healthy, test_healthy):
        folder.mkdir(parents=True)

    duplicate = train_scab / "duplicate.jpg"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(duplicate)
    (test_scab / duplicate.name).write_bytes(duplicate.read_bytes())
    for index, folder in enumerate((train_scab, train_scab, train_healthy, train_healthy,
                                    test_scab, test_healthy, test_healthy), start=1):
        Image.new("RGB", (20, 20), (index * 20, 40, 60)).save(folder / f"unique_{index}.jpg")

    classes = json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))["classes"]
    train, validation, test, split_info = field_datasets(root, classes, seed=7)
    train_paths = {Path(train.dataset.samples[index][0]) for index in train.indices}
    validation_paths = {
        Path(validation.dataset.samples[index][0]) for index in validation.indices
    }

    assert split_info["train_test_duplicates_removed"] == 1
    assert train_paths.isdisjoint(validation_paths)
    assert all(path.name != "duplicate.jpg" for path in train_paths)
    assert len(test) == 4
