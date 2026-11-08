# PlantCare AI

This project keeps the lightweight browser app and adds the optional Python
assistant and model-training workflow. The browser app remains the default;
the Python app adds CLIP-based photo checks, symptom cues, a care guide,
optional user-entered growing measurements, and English-language UI support.

Predictions are preliminary, not a guaranteed diagnosis. Confirm concerns with
a plant pathologist or local agricultural extension service.

## Browser app

Serve this folder over HTTP; opening `index.html` directly does not reliably
allow the browser to load the model and metadata.

```powershell
py -m http.server 8000
```

Keep the terminal open and visit <http://localhost:8000>. The app uses the
bundled `plant_classifier.onnx`, `classes.json`, and `disease_info.json`. On the
first analysis it loads ONNX Runtime Web 1.20.1 from jsDelivr and uses the
matching local `ort-wasm-simd-threaded.wasm`; an internet connection is
required for the runtime JavaScript. The browser classifier only recognizes
its trained crop classes. It does not use CLIP, so non-leaf photos may get
misleading predictions.

The browser app's **Open plant care guide** button opens `care.html` in a
separate tab. The guide starts with **General / unknown** selected, even when
opened from a diagnosis. If the classifier recognized a crop, the guide shows
it as an optional suggestion; choose its button to view tailored guidance.
Care guidance covers watering, sunlight, soil, temperature/humidity, nutrients,
and pests for 28 crops and plants, including common houseplants and flowers,
plus preventive steps and signs that warrant expert help. Optional measured soil
moisture, direct sunlight, air temperature, and
humidity can be entered for context. The app does not measure these values
from an image, and it compares temperature/humidity only where crop-specific
ranges are provided. Observed symptom checkboxes show possible causes, not a
diagnosis.

## Field-data fine-tuning

`train_plantdoc.py` fine-tunes the current MobileNetV3-Large browser model on
PlantDoc's labelled `train/` images and evaluates it on PlantDoc's separate
official `test/` split. The deployed ONNX graph is converted to PyTorch and
checked against ONNX Runtime before training, preserving the actual website
weights without downloading PlantVillage or starting from ImageNet. PlantDoc
folder labels are explicitly mapped to the browser's existing classes; unknown
labels stop the run instead of being silently discarded. The PlantDoc test
split is not used for training or model selection. Exact duplicates are kept
in one validation group; byte-identical train copies of official test images
and duplicate images with conflicting labels are excluded, with counts saved
in the evaluation report.
Validation results decide whether the candidate replaces the current website
model; the previous model and class metadata are backed up if it is promoted.

Download and extract the [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
to `datasets\plantdoc`; the archive should create
`datasets\plantdoc\PlantDoc-Dataset-master\train` and `test`. On Windows:

```powershell
New-Item -ItemType Directory -Force datasets\downloads, datasets\plantdoc | Out-Null
curl.exe -L --fail https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master -o datasets\downloads\plantdoc.zip
Expand-Archive -Path datasets\downloads\plantdoc.zip -DestinationPath datasets\plantdoc -Force
```

Install the training dependencies, then run:

```powershell
.\.venv\Scripts\python -m pip install torch torchvision pillow numpy onnx onnx2torch onnxruntime msvc-runtime
.\.venv\Scripts\python train_plantdoc.py
```

The script uses conservative defaults for CPU-only Windows training (3
fine-tuning epochs, batch size 8). Candidate weights and
`models\plantdoc_metrics.json` are saved under `models\`. The held-out field
test score covers only classes present in PlantDoc; it is not an accuracy
estimate for every supported class or real-world growing condition.
PlantDoc is licensed CC BY 4.0; retain the dataset attribution when using or
sharing derived results.

## Add rice diseases

`train_rice.py` expands the browser model with six rice labels: bacterial leaf
blight, brown spot, leaf blast, leaf scald, sheath blight, and healthy rice.
It uses the [Rice Leaf Disease Classification dataset](https://github.com/ai-agriculture-circuits-and-systems/rice_leaf_disease_classification),
which identifies its license as CC BY 4.0. Preserve attribution to the dataset
repository and verify upstream image terms before redistribution.

Download and extract the dataset:

```powershell
New-Item -ItemType Directory -Force datasets\downloads, datasets\rice | Out-Null
curl.exe -L --fail https://codeload.github.com/ai-agriculture-circuits-and-systems/rice_leaf_disease_classification/zip/refs/heads/main -o datasets\downloads\rice_leaf_disease_classification.zip
@'
import zipfile
from pathlib import Path, PurePosixPath

destination = Path("datasets/rice").resolve()
prefix = "rice_leaf_disease_classification-main/rice_leaves/"
with zipfile.ZipFile("datasets/downloads/rice_leaf_disease_classification.zip") as archive:
    members = [item for item in archive.infolist()
               if item.filename.startswith(prefix)
               or item.filename in {
                   "rice_leaf_disease_classification-main/LICENSE",
                   "rice_leaf_disease_classification-main/README.md",
               }]
    for item in members:
        parts = PurePosixPath(item.filename).parts
        target = destination.joinpath(*parts).resolve()
        if PurePosixPath(item.filename).is_absolute() or ".." in parts or not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive path: {item.filename}")
        archive.extract(item, destination)
'@ | .\.venv\Scripts\python -
```

Then run:

```powershell
.\.venv\Scripts\python train_rice.py
```

Because this repository does not provide an independent official split, the
script creates deterministic, per-class training, validation, and held-out test
splits and keeps byte-identical images together. It freezes the existing
feature extractor and original class outputs, learns the six added outputs,
and replays PlantDoc training images to discourage predictions from changing
to the new classes on existing field examples. Validation decides promotion;
the PlantDoc test and rice test splits are evaluated afterward. The previous
browser model and metadata are backed up in `models\`. The held-out rice score
describes only these six rice labels and should not be read as accuracy across
all crops or rice diseases.

## Python assistant

The Streamlit app is an optional companion; it does not replace the browser app.
On Windows, install its dependencies and run it with:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

The first run downloads CLIP weights (about 600 MB) for plant/leaf validation
and visual symptom cues. The Python diagnosis pipeline also expects
`models/plant_classifier.pt`; that checkpoint is not included. Without it, the
care guide and image checks remain available, but disease classification
reports that the model is missing. The existing ONNX file is for the browser
app and is not the PyTorch checkpoint.

## Train and export a classifier

Install the requirements above, then arrange labelled images in
`<dataset>/<Species___Condition>/*` folders. For a standard training run:

```powershell
py train.py --data-dir path\to\plantvillage\color
```

For field-aware training, provide a merged training folder and a separately
labelled field-photo folder whose class names are a subset of the training
classes. The field set must contain at least two images:

```powershell
py train_field.py --train-dir merged --field-dir field
```

The field workflow reserves part of the field photos for validation and keeps
the other part for a held-out test. `retrain_field.ipynb` contains a Google
Colab workflow for preparing PlantVillage and PlantDoc data. After training,
export with:

```powershell
py export_onnx.py
```

This writes `plant_classifier.onnx` and `classes.json` beside the browser app,
so the web app uses the exported model and matching labels. Keep the saved
PyTorch checkpoint in `models/plant_classifier.pt` for the Streamlit app.
Review the measured field-test results before relying on a model; random
PlantVillage test accuracy can be optimistic for real field photos.

## Tests and project contents

Run the Python unit tests after installing the requirements:

```powershell
py -m pytest -q
```

`plantcare/`, `app.py`, `train.py`, `train_field.py`, and `export_onnx.py` are
the integrated Python companion and training workflow. `data/` and `locales/`
contain its editable care/disease guidance and translations. The original
`plant_care_ai.zip` is retained as the supplied archive; use the integrated
project files above for ongoing changes.
