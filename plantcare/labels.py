"""Parse PlantVillage-style class names such as 'Tomato___Early_blight'."""
import re
from dataclasses import dataclass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def species_key(species: str) -> str:
    """'Corn_(maize)' -> 'corn', 'Pepper,_bell' -> 'pepper'."""
    return _slug(re.split(r"[,(]", species)[0])


def condition_key(condition: str) -> str:
    """'Esca_(Black_Measles)' -> 'esca'; 'Common_rust_' -> 'common_rust'."""
    return _slug(re.sub(r"\(.*?\)", "", condition))


@dataclass(frozen=True)
class Label:
    raw: str
    species: str
    condition: str
    species_key: str
    condition_key: str

    @property
    def is_healthy(self) -> bool:
        return self.condition_key == "healthy"


def parse_label(raw: str) -> Label:
    species, _, cond = raw.partition("___")
    if not cond:
        species, cond = raw, "unknown"
    return Label(
        raw=raw,
        species=species.replace("_", " ").strip(),
        condition=cond.replace("_", " ").strip(),
        species_key=species_key(species),
        condition_key=condition_key(cond),
    )
