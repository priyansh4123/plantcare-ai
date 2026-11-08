from pathlib import Path

import pytest
from PIL import Image

from train_rice import RICE_CLASS_MAP, canonical_rice_class, expand_classifier, rice_splits

import torch
import torch.nn as nn

def test_rice_labels_map_to_six_explicit_browser_classes():
    assert canonical_rice_class("BROWN_SPOT") == "Rice___Brown_spot"
    assert len(set(RICE_CLASS_MAP.values())) == 6


def test_unknown_rice_label_fails_explicitly():
    with pytest.raises(ValueError, match="Unmapped rice dataset class folder"):
        canonical_rice_class("unknown disease")


def test_rice_split_groups_duplicates_and_removes_conflicting_labels(tmp_path):
    class_names = [f"Existing___{index}" for index in range(38)] + list(RICE_CLASS_MAP.values())
    for class_index, folder_name in enumerate(RICE_CLASS_MAP):
        folder = tmp_path / folder_name / "images"
        folder.mkdir(parents=True)
        for image_index in range(5):
            pixel = (class_index * 30 + image_index + 1, 40, 60)
            Image.new("RGB", (20, 20), pixel).save(folder / f"{image_index}.png")

    duplicate_source = tmp_path / "brown_spot" / "images" / "0.png"
    (tmp_path / "brown_spot" / "images" / "same.jpg").write_bytes(
        duplicate_source.read_bytes()
    )
    conflicting = tmp_path / "leaf_blast" / "images" / "0.png"
    (tmp_path / "healthy_rice_leaf" / "images" / "conflict.jpg").write_bytes(
        conflicting.read_bytes()
    )
    (tmp_path / "leaf_scald" / "images" / "empty.jpg").write_bytes(b"")

    splits, info = rice_splits(tmp_path, class_names, seed=7)
    split_hashes = {}
    for split, samples in splits.items():
        split_hashes[split] = {
            Path(path).read_bytes() for path, _ in samples
        }
    assert not (split_hashes["train"] & split_hashes["validation"])
    assert not (split_hashes["train"] & split_hashes["test"])
    assert not (split_hashes["validation"] & split_hashes["test"])
    assert info["duplicate_images_removed"] == 1
    assert info["conflicting_label_images_removed"] == 2
    assert info["invalid_images_removed"] == 1
    assert info["invalid_images_by_class"] == {"Rice___Leaf_scald": 1}
    assert sum(info["split_counts"].values()) == 29
    targets = [target for samples in splits.values() for _, target in samples]
    assert min(targets) == 38
    assert max(targets) == 43


def test_expanded_classifier_keeps_existing_rows_frozen():
    model = nn.Sequential(nn.Linear(4, 38))
    old_weight = model[0].weight.detach().clone()
    old_bias = model[0].bias.detach().clone()
    head = expand_classifier(model, old_class_count=38, new_class_count=6)
    optimizer = torch.optim.Adam([head.weight, head.bias], lr=0.1)

    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()

    assert model(torch.randn(1, 4)).shape == (1, 44)
    assert torch.equal(head.weight[:38], old_weight)
    assert torch.equal(head.bias[:38], old_bias)
