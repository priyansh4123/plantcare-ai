"""Care guidance from a small editable knowledge base (data/*.json).

Important: nothing here measures soil moisture, light, temperature or humidity
from a photo. General guidance comes from the knowledge base; the user's own
readings (or sensors) are only *compared* with typical ranges when supplied.
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from .config import DATA_DIR
from .i18n import t
from .labels import species_key
from .schemas import Readings

SECTION_KEYS = ("water", "sunlight", "soil", "temp_humidity", "nutrients", "pests")


@lru_cache(maxsize=None)
def load_care_kb() -> dict:
    return json.loads((DATA_DIR / "care_knowledge.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_disease_info() -> dict:
    return json.loads((DATA_DIR / "disease_info.json").read_text(encoding="utf-8"))


def known_species() -> list:
    return sorted(v["display"] for k, v in load_care_kb().items() if not k.startswith("_"))


@dataclass
class CareSection:
    key: str
    title: str
    guidance: str
    your_data: list = field(default_factory=list)
    needs_measurement: str = ""


@dataclass
class CareReport:
    species: Optional[str]
    species_known: bool
    sections: list
    symptom_causes: list      # [(observed symptom, [possible causes])]
    preventive: list
    escalate: list
    questions: list           # what we still need from the user


def _range_msg(label, value, rng, unit, species, lang):
    lo, hi = rng
    pos = "below" if value < lo else "above" if value > hi else "within"
    return t(f"care.range_{pos}", lang, label=label, value=f"{value:g}", unit=unit,
             lo=f"{lo:g}", hi=f"{hi:g}", species=species)


def build_care_report(species: Optional[str], readings: Optional[Readings] = None, symptoms=(),
                      climate: Optional[str] = None, lang: str = "en") -> CareReport:
    kb = load_care_kb()
    key = species_key(species) if species else ""
    entry = kb.get(key)
    known = entry is not None
    entry = entry or kb["_generic"]
    name = entry.get("display", species or t("care.this_plant", lang))
    r = readings or Readings()

    data = {k: [] for k in SECTION_KEYS}
    if r.soil_moisture_pct is not None:
        data["water"].append(t("care.moisture_reading", lang, v=f"{r.soil_moisture_pct:g}"))
    if r.light_hours is not None:
        need = entry.get("light_hours_min")
        msg = t("care.light_reading", lang, v=f"{r.light_hours:g}")
        if need:
            msg += " " + t("care.light_need", lang, need=need, species=name)
        data["sunlight"].append(msg)
    if r.air_temp_c is not None and entry.get("temp_c"):
        data["temp_humidity"].append(_range_msg(t("care.temperature", lang), r.air_temp_c,
                                                entry["temp_c"], "°C", name, lang))
    if r.humidity_pct is not None and entry.get("humidity_pct"):
        data["temp_humidity"].append(_range_msg(t("care.humidity", lang), r.humidity_pct,
                                                entry["humidity_pct"], "%", name, lang))

    sections = [CareSection(k, t(f"care.title.{k}", lang), entry[k], data[k], t(f"care.needs.{k}", lang))
                for k in SECTION_KEYS]

    causes_kb = kb["_symptom_causes"]
    causes = [(t(f"symptom.{s}", lang), causes_kb[s]) for s in symptoms if s in causes_kb]

    questions = []
    if not known:
        questions.append(t("care.q.species", lang))
    if not climate:
        questions.append(t("care.q.climate", lang))
    questions += t("care.q.general", lang).split("\n")

    return CareReport(species=name if known else species, species_known=known, sections=sections,
                      symptom_causes=causes, preventive=kb["_preventive"], escalate=kb["_escalate"],
                      questions=questions)
