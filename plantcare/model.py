"""Model definition shared by training and inference (transfer learning)."""
import torch.nn as nn

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


# Register more backbones here (e.g. efficientnet) and pass --arch to train.py.
ARCHS = {"mobilenet_v3_large": _mobilenet_v3_large}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(ARCHS)}")
    return ARCHS[arch](num_classes, pretrained)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def eval_transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(int(IMG_SIZE * 1.14)), T.CenterCrop(IMG_SIZE),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
