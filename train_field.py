"""Retrain with field photos: python train_field.py --train-dir merged --field-dir field
train-dir: Species___Condition folders (PlantVillage subset + PlantDoc train).
field-dir: field photos only. Half is used to pick the best epoch and fit the temperature,
the other half is a held-out honest test."""
import argparse, json, torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from plantcare.config import MODEL_DIR
from plantcare.model import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, build_model, eval_transform, set_backbone_trainable
from train import evaluate, fit_temperature, run_epochs

def aug():
    return T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.25, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(30), T.ColorJitter(0.4, 0.4, 0.4, 0.1), T.RandomGrayscale(0.05),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True); p.add_argument("--field-dir", required=True)
    p.add_argument("--epochs-head", type=int, default=2); p.add_argument("--epochs-ft", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42); a = p.parse_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = datasets.ImageFolder(a.train_dir, transform=aug()); names = train_ds.classes
    tmp = datasets.ImageFolder(a.field_dir)
    if len(tmp.samples) < 2:
        raise ValueError("Field evaluation needs at least two labelled images for validation and test splits.")
    unknown = sorted(set(tmp.classes) - set(names))
    if unknown:
        raise ValueError("Field classes are missing from the training data: " + ", ".join(unknown))
    remap = {index: names.index(class_name) for index, class_name in enumerate(tmp.classes)}
    fd = datasets.ImageFolder(a.field_dir, transform=eval_transform(),
                              target_transform=lambda index: remap[index])
    perm = torch.randperm(len(fd), generator=torch.Generator().manual_seed(a.seed)).tolist(); h = len(perm) // 2
    mk = lambda ds, shuf: DataLoader(ds, batch_size=a.batch_size, shuffle=shuf, num_workers=a.num_workers)
    train_dl, val_dl, test_dl = mk(train_ds, True), mk(Subset(fd, perm[:h]), False), mk(Subset(fd, perm[h:]), False)
    print(f"{len(train_ds)} train, {h} field-val, {len(perm) - h} field-test, {len(names)} classes")
    model = build_model("mobilenet_v3_large", len(names), pretrained=True).to(device); best = {"acc": -1.0, "state": None}
    set_backbone_trainable(model, False); run_epochs(model, train_dl, val_dl, device, a.epochs_head, 1e-3, best)
    set_backbone_trainable(model, True); run_epochs(model, train_dl, val_dl, device, a.epochs_ft, 1e-4, best)
    model.load_state_dict(best["state"])
    val_acc, vl, vy = evaluate(model, val_dl, device); temp = fit_temperature(vl, vy)
    test_acc, tl, ty = evaluate(model, test_dl, device)
    print(f"field-val {val_acc:.3f}  FIELD-TEST {test_acc:.3f}  temperature {temp:.2f}")
    conf, pred = torch.softmax(tl / temp, 1).max(1)
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        m = conf >= thr
        print(f"threshold {thr}: answers {m.float().mean():.0%} of photos, correct {(pred[m] == ty[m]).float().mean() if m.any() else 0:.0%} of those")
    metrics = {"val_accuracy": val_acc, "test_accuracy": test_acc, "note": "Accuracy on held-out FIELD photos."}
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"arch": "mobilenet_v3_large", "class_names": names, "state_dict": best["state"],
                "temperature": temp, "metrics": metrics}, MODEL_DIR / "plant_classifier.pt")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics)); print("Saved models/plant_classifier.pt")

if __name__ == "__main__":
    main()
