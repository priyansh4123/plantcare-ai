"""Export the trained checkpoint to the browser app's ONNX model and metadata."""
import json
from pathlib import Path

import torch

from plantcare.model import build_model

ROOT = Path(__file__).resolve().parent
ck = torch.load(ROOT / "models" / "plant_classifier.pt", map_location="cpu", weights_only=True)
m = build_model(ck["arch"], len(ck["class_names"]), pretrained=False)
m.load_state_dict(ck["state_dict"]); m.eval()
torch.onnx.export(m, torch.zeros(1, 3, 224, 224), ROOT / "plant_classifier.onnx", input_names=["x"],
                  output_names=["logits"], opset_version=17, dynamo=False)
metadata = {"classes": ck["class_names"], "temperature": ck["temperature"],
            "accuracy": ck.get("metrics", {}).get("test_accuracy")}
with (ROOT / "classes.json").open("w", encoding="utf-8") as output:
    json.dump(metadata, output, indent=2)
print(f"Exported {ROOT / 'plant_classifier.onnx'} and {ROOT / 'classes.json'}")
