"""Orchestrates validation -> quality -> classification -> decision.

The pipeline depends only on small interfaces (gate.check, classifier.predict,
symptom_detector.detect), so each part can be swapped or faked in tests.
"""
import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image, ImageOps

from .config import THRESHOLDS, Thresholds
from .i18n import t
from .labels import parse_label
from .quality import assess_quality
from .schemas import Candidate, GateResult, QualityReport, SymptomCue
from .care import load_disease_info

log = logging.getLogger(__name__)


class Status(str, Enum):
    HEALTHY = "healthy"
    POSSIBLE_DISEASE = "possible_disease"
    INCONCLUSIVE = "inconclusive"
    UNABLE_TO_DETERMINE = "unable_to_determine"
    NOT_A_PLANT = "not_a_plant"
    LEAF_NOT_VISIBLE = "leaf_not_visible"
    LOW_QUALITY = "low_quality"
    ERROR = "error"

    @property
    def category(self) -> str:
        """Coarse grouping used by the UI: healthy / potentially_diseased / inconclusive / rejected."""
        if self is Status.HEALTHY:
            return "healthy"
        if self is Status.POSSIBLE_DISEASE:
            return "potentially_diseased"
        if self in (Status.INCONCLUSIVE, Status.UNABLE_TO_DETERMINE):
            return "inconclusive"
        return "rejected"


@dataclass
class DiagnosisResult:
    status: Status
    headline: str
    messages: list = field(default_factory=list)
    species: Optional[str] = None
    species_confidence: Optional[float] = None
    condition: Optional[str] = None
    confidence: Optional[float] = None
    candidates: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    disease_info: Optional[dict] = None
    warnings: list = field(default_factory=list)
    gate: Optional[GateResult] = None
    quality: Optional[QualityReport] = None
    disclaimer: str = ""


def prepare_image(source) -> Image.Image:
    """Accept a PIL image, path, bytes or file-like object; return an upright RGB image."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class DiagnosisPipeline:
    def __init__(self, gate, classifier=None, symptom_detector=None, model_metrics: dict = None,
                 thresholds: Thresholds = THRESHOLDS, lang: str = "en"):
        self.gate, self.classifier, self.symptoms = gate, classifier, symptom_detector
        self.metrics = model_metrics or {}
        self.t, self.lang = thresholds, lang

    # ---------------------------------------------------------------- helpers
    def _m(self, key, **fmt):
        return t(key, self.lang, **fmt)

    def _result(self, status, headline_key, messages=(), **kw) -> DiagnosisResult:
        res = DiagnosisResult(status=status, headline=self._m(headline_key), messages=list(messages),
                              disclaimer=self._m("msg.preliminary"), **kw)
        acc = self.metrics.get("test_accuracy")
        if acc is not None and acc < self.t.min_model_accuracy:
            res.warnings.append(self._m("msg.low_accuracy", acc=f"{acc:.0%}"))
        return res

    # ------------------------------------------------------------------- main
    def analyze(self, source) -> DiagnosisResult:
        try:
            img = prepare_image(source)
        except Exception:
            log.exception("could not read image")
            return self._result(Status.ERROR, "head.error", [self._m("msg.read_error")])

        quality = assess_quality(img, self.t)
        if not quality.ok:
            msgs = [self._m(f"msg.quality.{i}") for i in quality.issues] + [self._m("msg.retake")]
            return self._result(Status.LOW_QUALITY, "head.low_quality", msgs, quality=quality)

        try:
            gate = self.gate.check(img)
        except Exception:
            log.exception("gate failed")
            return self._result(Status.ERROR, "head.error", [self._m("msg.analysis_failed")], quality=quality)
        if gate.verdict == "not_plant":
            return self._result(Status.NOT_A_PLANT, "head.not_plant", [self._m("msg.no_plant")],
                                gate=gate, quality=quality)
        if gate.verdict == "leaf_not_visible":
            return self._result(Status.LEAF_NOT_VISIBLE, "head.leaf_not_visible",
                                [self._m("msg.leaf_not_visible")], gate=gate, quality=quality)

        if self.classifier is None:
            return self._result(Status.ERROR, "head.error", [self._m("msg.model_missing")],
                                gate=gate, quality=quality)
        try:
            pred = self.classifier.predict(img)
            cues = self.symptoms.detect(img) if self.symptoms else []
        except Exception:
            log.exception("classification failed")
            return self._result(Status.ERROR, "head.error", [self._m("msg.analysis_failed")],
                                gate=gate, quality=quality)
        return self._decide(pred, cues, gate, quality)

    # --------------------------------------------------------------- decision
    def _decide(self, pred, cues, gate, quality) -> DiagnosisResult:
        t_ = self.t
        top, second = pred.top[0], pred.top[1] if len(pred.top) > 1 else None
        best_species, sp_prob = max(pred.species_probs.items(), key=lambda kv: kv[1])
        species = best_species if sp_prob >= t_.species_min_confidence else None
        common = dict(gate=gate, quality=quality, symptoms=list(cues), candidates=pred.top[:3],
                      species=species, species_confidence=sp_prob if species else None)
        cue_names = ", ".join(self._m(f"symptom.{c.key}") for c in cues)
        lab = parse_label(top.label)

        # 1. Not confident enough at all -> unable to determine (never invent a diagnosis).
        if top.prob < t_.min_confidence:
            return self._finish(Status.UNABLE_TO_DETERMINE, "head.unable", [self._m("msg.unable")],
                                actions=self._m_list("actions.inconclusive"), **common)

        # 2. Two candidates too close -> report uncertainty.
        if second is not None and top.prob - second.prob < t_.ambiguity_margin:
            return self._finish(Status.INCONCLUSIVE, "head.inconclusive", [self._m("msg.inconclusive")],
                                actions=self._m_list("actions.inconclusive"), **common)

        # 3. Healthy needs stronger evidence, and must not contradict visible symptom cues.
        if lab.is_healthy:
            if top.prob < t_.healthy_confidence:
                return self._finish(Status.INCONCLUSIVE, "head.inconclusive", [self._m("msg.lean_healthy")],
                                    actions=self._m_list("actions.inconclusive"), **common)
            if cues:
                return self._finish(Status.INCONCLUSIVE, "head.inconclusive",
                                    [self._m("msg.healthy_cues_conflict", cues=cue_names)],
                                    actions=self._m_list("actions.inconclusive"), **common)
            common.update(species=lab.species, species_confidence=sp_prob)
            return self._finish(Status.HEALTHY, "head.healthy", [self._m("msg.healthy")],
                                confidence=top.prob, actions=self._m_list("actions.healthy"),
                                warnings=[self._m("msg.healthy_caveat")], **common)

        # 4. Possible disease.
        info = load_disease_info().get(lab.condition_key)
        common.update(species=lab.species, species_confidence=sp_prob)
        actions = list(info["management"]) if info else []
        actions += self._m_list("actions.disease")
        msgs = [self._m("msg.disease_headline", condition=lab.condition, species=lab.species)]
        if cues:
            msgs.append(self._m("msg.observed_cues", cues=cue_names))
        return self._finish(Status.POSSIBLE_DISEASE, "head.possible_disease", msgs, condition=lab.condition,
                            confidence=top.prob, actions=actions, disease_info=info, **common)

    def _m_list(self, key) -> list:
        return [s for s in self._m(key).split("\n") if s.strip()]

    def _finish(self, status, headline_key, messages, warnings=None, **kw) -> DiagnosisResult:
        res = self._result(status, headline_key, messages, **kw)
        res.warnings = (warnings or []) + res.warnings
        return res
