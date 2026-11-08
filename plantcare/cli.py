"""Command-line use:  python -m plantcare.cli path/to/leaf.jpg"""
import sys

from .classifier import DiseaseClassifier, ModelNotFoundError
from .clip_engine import ClipEngine, ClipPlantGate, ClipSymptomDetector
from .pipeline import DiagnosisPipeline


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        print(__doc__)
        return 2
    engine = ClipEngine()
    try:
        clf = DiseaseClassifier()
    except ModelNotFoundError:
        clf = None
    pipe = DiagnosisPipeline(ClipPlantGate(engine), clf, ClipSymptomDetector(engine),
                             getattr(clf, "metrics", {}))
    res = pipe.analyze(argv[0])
    print(f"[{res.status.value}] {res.headline}")
    for m in res.messages + res.warnings:
        print(" -", m)
    if res.species:
        print("Species:", res.species)
    if res.confidence is not None:
        print(f"Confidence: {res.confidence:.0%}")
    for a in res.actions:
        print(" *", a)
    print(res.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
