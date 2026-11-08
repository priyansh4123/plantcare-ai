"""Train the plant-disease classifier with transfer learning.

Example:
    python train.py --data-dir data/plantvillage/color --epochs-head 3 --epochs-ft 5

Expects an ImageFolder layout:  <data-dir>/<Species___Condition>/*.jpg
Writes models/plant_classifier.pt (weights, class names, temperature, metrics).
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from plantcare.config import CHECKPOINT_PATH, MODEL_DIR
from plantcare.model import (IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, ARCHS,
                             build_model, eval_transform, set_backbone_trainable)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--arch", default="mobilenet_v3_large", choices=list(ARCHS))
    p.add_argument("--epochs-head", type=int, default=3, help="epochs with frozen backbone")
    p.add_argument("--epochs-ft", type=int, default=5, help="epochs fine-tuning everything")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(CHECKPOINT_PATH))
    return p.parse_args()


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(), T.RandomRotation(25),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        logits_all.append(model(x.to(device)).cpu())
        labels_all.append(y)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    return (logits.argmax(1) == labels).float().mean().item(), logits, labels


def fit_temperature(logits, labels) -> float:
    """Temperature scaling on the validation set so confidences are better calibrated."""
    temp = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temp], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(temp.clamp(min=0.05).item())


def run_epochs(model, loader, val_loader, device, epochs, lr, best):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        acc, _, _ = evaluate(model, val_loader, device)
        print(f"  epoch {ep + 1}/{epochs}  loss {total / len(loader.dataset):.4f}  "
              f"val_acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc > best["acc"]:
            best["acc"] = acc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = datasets.ImageFolder(a.data_dir, transform=train_transform())
    eval_ds = datasets.ImageFolder(a.data_dir, transform=eval_transform())
    n = len(train_ds)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed)).tolist()
    n_val, n_test = int(n * a.val_frac), int(n * a.test_frac)
    idx_val, idx_test, idx_train = perm[:n_val], perm[n_val:n_val + n_test], perm[n_val + n_test:]
    mk = lambda ds, idx, shuf: DataLoader(Subset(ds, idx), batch_size=a.batch_size, shuffle=shuf,
                                          num_workers=a.num_workers, pin_memory=(device == "cuda"))
    train_dl, val_dl, test_dl = mk(train_ds, idx_train, True), mk(eval_ds, idx_val, False), mk(eval_ds, idx_test, False)
    class_names = train_ds.classes
    print(f"{n} images, {len(class_names)} classes (train {len(idx_train)}, val {n_val}, test {n_test})")

    model = build_model(a.arch, len(class_names), pretrained=True).to(device)
    best = {"acc": -1.0, "state": None}

    print("Phase 1: training classifier head (backbone frozen)")
    set_backbone_trainable(model, False)
    run_epochs(model, train_dl, val_dl, device, a.epochs_head, a.lr_head, best)
    print("Phase 2: fine-tuning whole network")
    set_backbone_trainable(model, True)
    run_epochs(model, train_dl, val_dl, device, a.epochs_ft, a.lr_ft, best)

    model.load_state_dict(best["state"])
    val_acc, val_logits, val_labels = evaluate(model, val_dl, device)
    temperature = fit_temperature(val_logits, val_labels)
    test_acc, _, _ = evaluate(model, test_dl, device)
    print(f"val_acc {val_acc:.4f}  test_acc {test_acc:.4f}  temperature {temperature:.3f}")

    metrics = {
        "val_accuracy": val_acc, "test_accuracy": test_acc, "num_classes": len(class_names),
        "n_train": len(idx_train), "n_val": n_val, "n_test": n_test,
        "note": "Random split. PlantVillage has many near-duplicate photos of the same leaf, so "
                "this accuracy is optimistic; expect lower accuracy on real field photos.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"arch": a.arch, "class_names": class_names, "state_dict": best["state"],
                "temperature": temperature, "metrics": metrics}, a.out)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
