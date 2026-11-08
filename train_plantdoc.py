"""Fine-tune the browser classifier on PlantDoc and evaluate on its official test split."""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
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
from onnx2torch import convert
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms as T

from plantcare.model import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, eval_transform


FIELD_CLASS_MAP = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple leaf": "Apple___healthy",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "bell_pepper leaf": "Pepper,_bell___healthy",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "peach leaf": "Peach___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato leaf": "Tomato___healthy",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def canonical_field_class(name: str) -> str:
    try:
        return FIELD_CLASS_MAP[name.strip().casefold()]
    except KeyError as exc:
        raise ValueError(f"Unmapped PlantDoc class folder: {name!r}") from exc


def training_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(0.25, 0.25, 0.25, 0.08),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ConvertedClassifier(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, images):
        output = self.module(images)
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise RuntimeError(f"Expected one classifier output, received {len(output)}.")
            return output[0]
        if isinstance(output, dict):
            if len(output) != 1:
                raise RuntimeError(f"Expected one classifier output, received {len(output)}.")
            return next(iter(output.values()))
        return output


def field_datasets(plantdoc_root: Path, class_names: list[str], seed: int):
    class_index = {name: index for index, name in enumerate(class_names)}
    roots = [plantdoc_root / split for split in ("train", "test")]
    datasets_by_split = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"PlantDoc split folder not found: {root}")
        folder = datasets.ImageFolder(root)
        remap = {}
        for index, folder_name in enumerate(folder.classes):
            name = canonical_field_class(folder_name)
            if name not in class_index:
                raise ValueError(f"PlantDoc class {folder_name!r} maps to unsupported class {name!r}.")
            remap[index] = class_index[name]
        datasets_by_split.append((remap, folder.classes))

    train_remap, _ = datasets_by_split[0]
    test_remap, _ = datasets_by_split[1]
    train = datasets.ImageFolder(
        roots[0], transform=training_transform(), target_transform=train_remap.__getitem__
    )
    train_eval = datasets.ImageFolder(
        roots[0], transform=eval_transform(), target_transform=train_remap.__getitem__
    )
    official_test = datasets.ImageFolder(
        roots[1], transform=eval_transform(), target_transform=test_remap.__getitem__
    )
    test_groups: dict[bytes, list[tuple[int, int]]] = {}
    for index, (path, target) in enumerate(official_test.samples):
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        test_groups.setdefault(digest, []).append((index, test_remap[target]))
    by_class: dict[int, dict[bytes, list[int]]] = {}
    train_groups: dict[bytes, list[tuple[int, int]]] = {}
    for index, (path, target) in enumerate(train.samples):
        mapped = train_remap[target]
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        train_groups.setdefault(digest, []).append((index, mapped))

    excluded_train_duplicates = 0
    excluded_train_label_conflicts = 0
    for digest, samples in train_groups.items():
        labels = {label for _, label in samples}
        if digest in test_groups:
            excluded_train_duplicates += len(samples)
        elif len(labels) > 1:
            excluded_train_label_conflicts += len(samples)
        else:
            label = next(iter(labels))
            by_class.setdefault(label, {})[digest] = [index for index, _ in samples]

    excluded_test_label_conflicts = set()
    valid_test_indices = []
    for samples in test_groups.values():
        labels = {label for _, label in samples}
        if len(labels) > 1:
            excluded_test_label_conflicts.update(index for index, _ in samples)
        else:
            valid_test_indices.extend(index for index, _ in samples)
    if not valid_test_indices:
        raise ValueError("No unambiguous images remain in the official PlantDoc test split.")
    official_test = Subset(official_test, sorted(valid_test_indices))

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for groups in by_class.values():
        group_keys = list(groups)
        rng.shuffle(group_keys)
        val_count = max(1, round(len(group_keys) * 0.2)) if len(group_keys) >= 2 else 0
        val_hashes = set(group_keys[:val_count])
        for digest, indices in groups.items():
            (val_indices if digest in val_hashes else train_indices).extend(indices)
    if not val_indices:
        raise ValueError("PlantDoc training split is too small to create a validation set.")
    if not train_indices:
        raise ValueError("PlantDoc training split is too small to train a model.")
    return (
        Subset(train, train_indices),
        Subset(train_eval, val_indices),
        official_test,
        {
            "train_test_duplicates_removed": excluded_train_duplicates,
            "train_label_conflict_images_removed": excluded_train_label_conflicts,
            "test_label_conflict_images_removed": len(excluded_test_label_conflicts),
        },
    )


def load_model_from_onnx(model_path: Path, session) -> nn.Module:
    model = ConvertedClassifier(convert(onnx.load(str(model_path)))).cpu().eval()
    if not any(True for _ in model.parameters()):
        raise RuntimeError("The ONNX converter produced a model without trainable weights.")
    sample = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.inference_mode():
        converted = model(sample)
        reference = session.run(
            None, {session.get_inputs()[0].name: sample.numpy()}
        )[0]
    if not np.allclose(converted.detach().numpy(), reference, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            "The PyTorch conversion does not match the deployed ONNX model; "
            "training stopped without changing the website model."
        )
    return model


def classification_metrics(predictions, labels, class_names):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    supported = confusion.sum(axis=1) > 0
    scores = []
    per_class = {}
    for index, name in enumerate(class_names):
        tp = confusion[index, index]
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            scores.append(f1)
            per_class[name] = {"support": support, "accuracy": recall, "f1": f1}
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "macro_f1_supported_classes": float(np.mean(scores)) if scores else 0.0,
        "supported_classes": len(scores),
        "per_class": per_class,
    }


def evaluate_torch(model, loader, device, class_names):
    model.eval()
    predictions, labels = [], []
    logits_all = []
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device)).cpu()
            logits_all.append(logits)
            predictions.extend(logits.argmax(dim=1).tolist())
            labels.extend(target.tolist())
    return classification_metrics(predictions, labels, class_names), torch.cat(logits_all), torch.tensor(labels)


def evaluate_onnx(session, loader, class_names):
    input_name = session.get_inputs()[0].name
    predictions, labels = [], []
    with torch.inference_mode():
        for images, target in loader:
            for image, label in zip(images, target):
                logits = session.run(None, {input_name: image.unsqueeze(0).numpy()})[0]
                predictions.append(int(np.asarray(logits).argmax(axis=1)[0]))
                labels.append(int(label))
    return classification_metrics(predictions, labels, class_names)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=100)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.05).item())


def train_phase(model, loader, val_loader, device, epochs, learning_rate, best):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics, _, _ = evaluate_torch(model, val_loader, device, best["classes"])
        print(
            f"epoch {epoch + 1}/{epochs}: loss={total_loss / len(loader.dataset):.4f}, "
            f"val_accuracy={metrics['accuracy']:.4f}, "
            f"val_macro_f1={metrics['macro_f1_supported_classes']:.4f}",
            flush=True,
        )
        score = metrics["macro_f1_supported_classes"]
        if score > best["macro_f1"]:
            best["macro_f1"] = score
            best["state"] = {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()}


def export_model(model, output_path: Path):
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),
        output_path,
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plantdoc-root", type=Path, default=Path("datasets/plantdoc/PlantDoc-Dataset-master")
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("At least one training epoch is required.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    root = Path(__file__).resolve().parent
    model_path = root / "plant_classifier.onnx"
    metadata_path = root / "classes.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Expected plant_classifier.onnx and classes.json beside this script.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = metadata["classes"]
    class_index = {name: index for index, name in enumerate(class_names)}

    train_set, val_set, test_set, split_info = field_datasets(
        args.plantdoc_root, class_names, args.seed
    )
    train_targets = [train_set.dataset.target_transform(train_set.dataset.targets[index])
                     for index in train_set.indices]
    counts = Counter(train_targets)
    weights = [1.0 / math.sqrt(counts[target]) for target in train_targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(
        f"PlantDoc train={len(train_set)}, validation={len(val_set)}, "
        f"official test={len(test_set)}; removed ambiguous/duplicate files: {split_info}.",
        flush=True,
    )

    baseline_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    baseline_val = evaluate_onnx(baseline_session, val_loader, class_names)
    model = load_model_from_onnx(model_path, baseline_session).to(device)
    best = {"macro_f1": -1.0, "state": None, "classes": class_names}

    train_phase(model, train_loader, val_loader, device, args.epochs, 1e-5, best)

    model.load_state_dict(best["state"])
    candidate_val, val_logits, val_labels = evaluate_torch(model, val_loader, device, class_names)
    promote = (
        candidate_val["macro_f1_supported_classes"] > baseline_val["macro_f1_supported_classes"]
        and candidate_val["accuracy"] >= baseline_val["accuracy"] - 0.02
    )
    selected = "candidate" if promote else "existing"
    print(
        f"Validation: current acc={baseline_val['accuracy']:.4f} "
        f"macro-F1={baseline_val['macro_f1_supported_classes']:.4f}; "
        f"candidate acc={candidate_val['accuracy']:.4f} "
        f"macro-F1={candidate_val['macro_f1_supported_classes']:.4f}; "
        f"selected={selected}",
        flush=True,
    )

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    candidate_onnx = model_dir / "plant_classifier_candidate.onnx"
    export_model(model, candidate_onnx)
    candidate_metadata = {
        **metadata,
        "temperature": fit_temperature(val_logits, val_labels),
        "accuracy": candidate_val["accuracy"],
        "accuracy_dataset": "PlantDoc validation split",
        "accuracy_classes": candidate_val["supported_classes"],
    }
    candidate_meta_path = model_dir / "classes_candidate.json"
    candidate_meta_path.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")

    candidate_session = ort.InferenceSession(str(candidate_onnx), providers=["CPUExecutionProvider"])
    candidate_roundtrip = evaluate_onnx(candidate_session, val_loader, class_names)
    if abs(candidate_roundtrip["accuracy"] - candidate_val["accuracy"]) > 0.005:
        raise RuntimeError("Exported ONNX candidate does not reproduce the PyTorch validation accuracy.")

    baseline_test = evaluate_onnx(baseline_session, test_loader, class_names)
    candidate_test = evaluate_onnx(candidate_session, test_loader, class_names)
    print(
        f"Official PlantDoc test (evaluated once after validation-based selection): "
        f"current acc={baseline_test['accuracy']:.4f}, "
        f"candidate acc={candidate_test['accuracy']:.4f}; selected={selected}",
        flush=True,
    )

    metrics = {
        "dataset": "PlantDoc",
        "source": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "seed": args.seed,
        "split_integrity": split_info,
        "classes_in_test": baseline_test["supported_classes"],
        "selection_rule": "Higher PlantDoc validation macro-F1 and no more than 2 percentage points lower validation accuracy.",
        "selected_model": selected,
        "validation": {"existing": baseline_val, "candidate": candidate_val},
        "official_test": {"existing": baseline_test, "candidate": candidate_test},
    }
    metrics_path = model_dir / "plantdoc_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if promote:
        backup_model = model_dir / "plant_classifier.before_plantdoc.onnx"
        backup_meta = model_dir / "classes.before_plantdoc.json"
        if not backup_model.exists():
            shutil.copy2(model_path, backup_model)
        if not backup_meta.exists():
            shutil.copy2(metadata_path, backup_meta)
        promoted_meta = {
            **metadata,
            "model_version": hashlib.sha256(candidate_onnx.read_bytes()).hexdigest()[:16],
            "temperature": candidate_metadata["temperature"],
            "accuracy": candidate_test["accuracy"],
            "accuracy_dataset": "PlantDoc official test split",
            "accuracy_classes": candidate_test["supported_classes"],
        }
        shutil.copy2(candidate_onnx, model_path)
        metadata_path.write_text(json.dumps(promoted_meta, indent=2), encoding="utf-8")
        print("Promoted the candidate model; the previous browser model is backed up in models/.", flush=True)
    else:
        print("Kept the current browser model; candidate and evaluation are saved in models/.", flush=True)

    print(f"Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
