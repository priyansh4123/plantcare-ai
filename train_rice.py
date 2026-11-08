"""Expand the browser classifier with rice diseases using the CC BY 4.0 dataset."""
import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_MSVC_DLL_PATH = Path(sys.prefix) / "Scripts"
_MSVC_DLL_DIRECTORY = (
    os.add_dll_directory(str(_MSVC_DLL_PATH)) if _MSVC_DLL_PATH.is_dir() else None
)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from onnx2torch import convert
from torch.utils.data import DataLoader, Dataset

from plantcare.model import IMG_SIZE, eval_transform
from train_plantdoc import (
    ConvertedClassifier,
    classification_metrics,
    evaluate_onnx,
    field_datasets,
    fit_temperature,
)


RICE_CLASS_MAP = {
    "bacterial_leaf_blight": "Rice___Bacterial_leaf_blight",
    "brown_spot": "Rice___Brown_spot",
    "healthy_rice_leaf": "Rice___healthy",
    "leaf_blast": "Rice___Leaf_blast",
    "leaf_scald": "Rice___Leaf_scald",
    "sheath_blight": "Rice___Sheath_blight",
}
RICE_SOURCE = (
    "https://github.com/ai-agriculture-circuits-and-systems/"
    "rice_leaf_disease_classification"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RiceImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), target


def canonical_rice_class(folder_name):
    try:
        return RICE_CLASS_MAP[folder_name.strip().casefold()]
    except KeyError as exc:
        raise ValueError(f"Unmapped rice dataset class folder: {folder_name!r}") from exc


def rice_splits(root, class_names, seed):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Rice dataset folder not found: {root}")
    class_index = {name: index for index, name in enumerate(class_names)}
    digest_groups = defaultdict(list)
    invalid_by_class = defaultdict(int)
    seen_folders = set()
    duplicate_count = 0
    conflict_count = 0

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        class_name = canonical_rice_class(folder.name)
        if class_name not in class_index:
            raise ValueError(f"Rice dataset class is missing from metadata: {class_name}")
        seen_folders.add(folder.name.casefold())
        target = class_index[class_name]
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            image_bytes = path.read_bytes()
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
            except OSError:
                invalid_by_class[class_name] += 1
                continue
            digest = hashlib.sha256(image_bytes).digest()
            digest_groups[digest].append((target, path))

    expected_folders = set(RICE_CLASS_MAP)
    if seen_folders != expected_folders:
        missing = sorted(expected_folders - seen_folders)
        unexpected = sorted(seen_folders - expected_folders)
        raise ValueError(f"Rice dataset folders do not match expected labels; missing={missing}, "
                         f"unexpected={unexpected}.")

    grouped = defaultdict(dict)
    for digest, entries in digest_groups.items():
        labels = {target for target, _ in entries}
        if len(labels) > 1:
            conflict_count += len(entries)
            continue
        target = next(iter(labels))
        paths = [path for _, path in entries]
        grouped[target][digest] = paths
        duplicate_count += len(paths) - 1

    rng = random.Random(seed)
    split_samples = {"train": [], "validation": [], "test": []}
    for target, groups in grouped.items():
        unique_groups = list(groups.items())
        if len(unique_groups) < 3:
            raise ValueError(f"Rice class {class_names[target]} needs at least 3 unique images.")
        rng.shuffle(unique_groups)
        val_count = max(1, round(len(unique_groups) * 0.15))
        test_count = max(1, round(len(unique_groups) * 0.15))
        if val_count + test_count >= len(unique_groups):
            raise ValueError(f"Not enough unique images to split {class_names[target]}.")
        train_groups = unique_groups[val_count + test_count:]
        val_groups = unique_groups[:val_count]
        test_groups = unique_groups[val_count:val_count + test_count]
        for split, groups_for_split in (
            ("train", train_groups), ("validation", val_groups), ("test", test_groups)
        ):
            split_samples[split].extend(
                (paths[0], target) for _, paths in groups_for_split
            )
    return split_samples, {
        "unique_images_by_class": {
            class_names[target]: len(groups) for target, groups in sorted(grouped.items())
        },
        "duplicate_images_removed": duplicate_count,
        "conflicting_label_images_removed": conflict_count,
        "invalid_images_removed": sum(invalid_by_class.values()),
        "invalid_images_by_class": dict(sorted(invalid_by_class.items())),
        "split_counts": {
            split: len(samples) for split, samples in split_samples.items()
        },
    }


def last_linear(model):
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, nn.Linear)]
    if not layers:
        raise RuntimeError("Converted ONNX model has no linear classifier layer.")
    return layers[-1]


def expand_classifier(model, old_class_count, new_class_count):
    name, old_head = last_linear(model)
    parent_name, _, attribute = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    head = nn.Linear(old_head.in_features, old_class_count + new_class_count)
    with torch.no_grad():
        head.weight[:old_class_count].copy_(old_head.weight)
        head.bias[:old_class_count].copy_(old_head.bias)
    setattr(parent, attribute, head)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    weight_mask = torch.zeros_like(head.weight)
    bias_mask = torch.zeros_like(head.bias)
    weight_mask[old_class_count:] = 1
    bias_mask[old_class_count:] = 1
    head.weight.register_hook(lambda gradient: gradient * weight_mask)
    head.bias.register_hook(lambda gradient: gradient * bias_mask)
    return head


def extract_features(model, loader, head, device):
    features, logits, labels = [], [], []

    def capture_input(_module, inputs):
        features.append(inputs[0].detach().cpu())

    hook = head.register_forward_pre_hook(capture_input)
    model.eval()
    try:
        with torch.inference_mode():
            for images, target in loader:
                output = model(images.to(device)).cpu()
                logits.append(output)
                labels.append(target.cpu())
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(logits), torch.cat(labels)


def evaluate_features(head, features, labels, class_names):
    with torch.inference_mode():
        logits = head(features)
    metrics = classification_metrics(logits.argmax(dim=1).tolist(), labels.tolist(), class_names)
    return metrics, logits


def train_new_classes(head, rice_features, rice_labels, replay_features, replay_labels,
                      val_features, val_labels, val_rice_features, val_rice_labels,
                      class_names, old_class_count, baseline_old_accuracy,
                      baseline_old_macro_f1, epochs, batch_size, learning_rate, seed):
    optimizer = torch.optim.Adam([head.weight, head.bias], lr=learning_rate)
    old_targets = head.weight[:old_class_count].detach().clone()
    old_bias = head.bias[:old_class_count].detach().clone()
    best = {
        "rice_macro_f1": -1.0,
        "state": None,
        "validation": None,
        "baseline_old_accuracy": baseline_old_accuracy,
        "baseline_old_macro_f1": baseline_old_macro_f1,
    }
    rng = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        head.train()
        rice_order = torch.randperm(len(rice_features), generator=rng)
        replay_order = torch.randperm(len(replay_features), generator=rng)
        rice_steps = math.ceil(len(rice_order) / batch_size)
        replay_steps = math.ceil(len(replay_order) / batch_size)
        steps = max(rice_steps, replay_steps)
        total_loss = 0.0
        for step in range(steps):
            rice_indices = rice_order[(step * batch_size) % len(rice_order):
                                      (step * batch_size) % len(rice_order) + batch_size]
            replay_start = (step * batch_size) % len(replay_order)
            replay_indices = replay_order[replay_start:replay_start + batch_size]
            if len(replay_indices) < batch_size:
                replay_indices = torch.cat((replay_indices, replay_order[:batch_size - len(replay_indices)]))

            rice_logits = head(rice_features[rice_indices])
            replay_logits = head(replay_features[replay_indices])
            loss = F.cross_entropy(rice_logits, rice_labels[rice_indices]) + 0.5 * F.cross_entropy(
                replay_logits, replay_labels[replay_indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        validation, _ = evaluate_features(head, val_features, val_labels, class_names)
        rice_validation, _ = evaluate_features(
            head, val_rice_features, val_rice_labels, class_names
        )
        old_ok = (
            validation["accuracy"] >= best["baseline_old_accuracy"] - 0.02
            and validation["macro_f1_supported_classes"] >= best["baseline_old_macro_f1"] - 0.02
        )
        print(
            f"epoch {epoch + 1}/{epochs}: loss={total_loss / steps:.4f}, "
            f"rice_val_accuracy={rice_validation['accuracy']:.4f}, "
            f"rice_val_macro_f1={rice_validation['macro_f1_supported_classes']:.4f}, "
            f"old_val_accuracy={validation['accuracy']:.4f}",
            flush=True,
        )
        score = rice_validation["macro_f1_supported_classes"]
        if old_ok and score > best["rice_macro_f1"]:
            best["rice_macro_f1"] = score
            best["state"] = {key: value.detach().clone() for key, value in head.state_dict().items()}
            best["validation"] = {
                "existing_classes": validation,
                "rice_classes": rice_validation,
            }
        if not torch.equal(head.weight[:old_class_count], old_targets) or not torch.equal(
            head.bias[:old_class_count], old_bias
        ):
            raise RuntimeError("Training changed the existing classifier rows.")
    return best


def export_model(model, output_path):
    model.eval().cpu()
    torch.onnx.export(
        model, torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), output_path,
        input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rice-root", type=Path,
        default=Path("datasets/rice/rice_leaf_disease_classification-main/rice_leaves"),
    )
    parser.add_argument(
        "--plantdoc-root", type=Path,
        default=Path("datasets/plantdoc/PlantDoc-Dataset-master"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    random.seed(args.seed)
    device = "cpu"
    print("Device: cpu (frozen feature extraction)", flush=True)

    root = Path(__file__).resolve().parent
    model_path = root / "plant_classifier.onnx"
    metadata_path = root / "classes.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_classes = metadata["classes"]
    new_classes = [RICE_CLASS_MAP[key] for key in RICE_CLASS_MAP]
    if set(old_classes).intersection(new_classes):
        raise ValueError("Rice class labels already exist in the browser model.")
    class_names = old_classes + new_classes

    rice_data, split_info = rice_splits(args.rice_root, class_names, args.seed)
    rice_sets = {
        split: RiceImageDataset(samples, eval_transform())
        for split, samples in rice_data.items()
    }
    rice_loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        for split, dataset in rice_sets.items()
    }

    replay_train, replay_val, replay_test, replay_info = field_datasets(
        args.plantdoc_root, old_classes, args.seed
    )
    replay_train.dataset.transform = eval_transform()
    replay_train_loader = DataLoader(
        replay_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    replay_val_loader = DataLoader(
        replay_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    replay_test_loader = DataLoader(
        replay_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(
        f"Rice split={split_info['split_counts']}; duplicate images removed="
        f"{split_info['duplicate_images_removed']}; invalid images removed="
        f"{split_info['invalid_images_removed']}; PlantDoc replay train={len(replay_train)}, "
        f"validation={len(replay_val)}, held-out test={len(replay_test)}.",
        flush=True,
    )

    baseline_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    baseline_val = evaluate_onnx(baseline_session, replay_val_loader, old_classes)
    model = ConvertedClassifier(convert(onnx.load(str(model_path)))).cpu()
    sample = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.inference_mode():
        converted = model(sample).numpy()
        reference = baseline_session.run(
            None, {baseline_session.get_inputs()[0].name: sample.numpy()}
        )[0]
    if not np.allclose(converted, reference, rtol=2e-3, atol=2e-4):
        raise RuntimeError("Converted classifier does not match the deployed ONNX model.")

    head = expand_classifier(model, len(old_classes), len(new_classes))
    rice_train_features, _, rice_train_labels = extract_features(
        model, rice_loaders["train"], head, device
    )
    rice_val_features, _, rice_val_local_labels = extract_features(
        model, rice_loaders["validation"], head, device
    )
    replay_train_features, _, replay_train_labels = extract_features(
        model, replay_train_loader, head, device
    )
    replay_val_features, _, replay_val_labels = extract_features(
        model, replay_val_loader, head, device
    )
    rice_val_labels = rice_val_local_labels

    best = train_new_classes(
        head,
        rice_train_features,
        rice_train_labels,
        replay_train_features,
        replay_train_labels,
        replay_val_features,
        replay_val_labels,
        rice_val_features,
        rice_val_labels,
        class_names,
        len(old_classes),
        baseline_val["accuracy"],
        baseline_val["macro_f1_supported_classes"],
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.seed,
    )
    promote = best["state"] is not None
    selected = "candidate" if promote else "existing"
    print(f"Validation selected={selected}.", flush=True)
    if promote:
        head.load_state_dict(best["state"])

    candidate_val_metrics, candidate_val_logits = evaluate_features(
        head, replay_val_features, replay_val_labels, class_names
    )
    rice_val_metrics, rice_val_logits = evaluate_features(
        head, rice_val_features, rice_val_labels, class_names
    )

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    candidate_path = model_dir / "plant_classifier_rice_candidate.onnx"
    export_model(model, candidate_path)
    roundtrip = ort.InferenceSession(str(candidate_path), providers=["CPUExecutionProvider"])
    candidate_rice_val_onnx = evaluate_onnx(
        roundtrip, rice_loaders["validation"], class_names
    )
    candidate_old_val_onnx = evaluate_onnx(
        roundtrip, replay_val_loader, class_names
    )
    if (
        abs(candidate_rice_val_onnx["accuracy"] - rice_val_metrics["accuracy"]) > 0.005
        or abs(candidate_old_val_onnx["accuracy"] - candidate_val_metrics["accuracy"]) > 0.005
    ):
        raise RuntimeError("Exported ONNX candidate does not reproduce validation accuracy.")
    candidate_rice_test = evaluate_onnx(
        roundtrip, rice_loaders["test"], class_names
    )
    candidate_old_test = evaluate_onnx(
        roundtrip, replay_test_loader, class_names
    )

    baseline_test = evaluate_onnx(baseline_session, replay_test_loader, old_classes)
    temperature_logits = torch.cat((rice_val_logits, candidate_val_logits))
    temperature_labels = torch.cat((rice_val_labels, replay_val_labels))
    temperature = fit_temperature(temperature_logits, temperature_labels)
    metrics = {
        "dataset": "Rice Leaf Disease Classification Dataset",
        "source": RICE_SOURCE,
        "license": "CC BY 4.0",
        "seed": args.seed,
        "classes_added": new_classes,
        "rice_split_integrity": split_info,
        "plantdoc_replay_split_integrity": replay_info,
        "selection_rule": (
            "Candidate must preserve PlantDoc validation accuracy and supported-class macro-F1 "
            "within 2 percentage points of the current model."
        ),
        "selected_model": selected,
        "validation": {
            "existing_model_on_plantdoc": baseline_val,
            "candidate_on_plantdoc": candidate_val_metrics,
            "candidate_on_rice": rice_val_metrics,
        },
        "held_out_test": {
            "existing_model_on_plantdoc": baseline_test,
            "candidate_on_plantdoc": candidate_old_test,
            "candidate_on_rice": candidate_rice_test,
        },
    }
    metrics_path = model_dir / "rice_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if promote:
        backup_model = model_dir / "plant_classifier.before_rice.onnx"
        backup_metadata = model_dir / "classes.before_rice.json"
        if not backup_model.exists():
            shutil.copy2(model_path, backup_model)
        if not backup_metadata.exists():
            shutil.copy2(metadata_path, backup_metadata)
        promoted_metadata = {
            **metadata,
            "classes": class_names,
            "model_version": hashlib.sha256(candidate_path.read_bytes()).hexdigest()[:16],
            "temperature": temperature,
            "accuracy": candidate_rice_test["accuracy"],
            "accuracy_dataset": "Rice dataset held-out test split",
            "accuracy_classes": candidate_rice_test["supported_classes"],
        }
        shutil.copy2(candidate_path, model_path)
        metadata_path.write_text(json.dumps(promoted_metadata, indent=2), encoding="utf-8")
        print("Promoted the expanded model; the previous model is backed up in models/.", flush=True)
    else:
        print("Kept the current browser model; candidate and metrics are saved in models/.", flush=True)
    print(f"Rice metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
