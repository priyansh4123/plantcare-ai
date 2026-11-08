import re
from pathlib import Path

from plantcare.care import SECTION_KEYS, build_care_report
from plantcare.i18n import _load
from plantcare.labels import parse_label
from plantcare.schemas import Readings

ROOT = Path(__file__).resolve().parent.parent


def test_parse_label_variants():
    lab = parse_label("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    assert lab.species_key == "corn"
    assert lab.condition_key == "cercospora_leaf_spot_gray_leaf_spot"
    assert parse_label("Pepper,_bell___healthy").species_key == "pepper"
    assert parse_label("Grape___Esca_(Black_Measles)").condition_key == "esca"
    assert parse_label("Apple___healthy").is_healthy


def test_care_report_has_all_sections_and_no_invented_measurements():
    rep = build_care_report("tomato")
    assert [s.key for s in rep.sections] == list(SECTION_KEYS)
    assert all(s.your_data == [] for s in rep.sections)   # nothing supplied -> nothing reported
    assert all(s.needs_measurement for s in rep.sections)


def test_user_readings_are_compared_with_typical_ranges():
    rep = build_care_report("tomato", Readings(air_temp_c=40, humidity_pct=55, soil_moisture_pct=30))
    th = next(s for s in rep.sections if s.key == "temp_humidity")
    assert any("above" in d for d in th.your_data) and any("within" in d for d in th.your_data)


def test_unknown_species_asks_questions():
    rep = build_care_report("Dragonfruit")
    assert not rep.species_known
    assert any("species" in q for q in rep.questions)


def test_common_houseplants_and_flowers_have_care_requirements():
    kb_species = {
        "african_violet", "aloe_vera", "geranium", "hibiscus", "marigold",
        "monstera", "orchid", "peace_lily", "pothos", "rose",
        "snake_plant", "spider_plant", "sunflower", "tulip",
    }
    from plantcare.care import load_care_kb

    kb = load_care_kb()
    assert kb_species <= kb.keys()
    for key in kb_species:
        entry = kb[key]
        assert all(entry.get(section) for section in SECTION_KEYS), key
        assert len(entry["temp_c"]) == 2 and entry["temp_c"][0] < entry["temp_c"][1], key
        assert len(entry["humidity_pct"]) == 2 and entry["humidity_pct"][0] < entry["humidity_pct"][1], key
        assert entry["light_hours_min"] > 0, key

    report = build_care_report("pothos", Readings(air_temp_c=35))
    assert report.species_known
    temp = next(section for section in report.sections if section.key == "temp_humidity")
    assert any("above" in reading for reading in temp.your_data)


def test_every_message_key_used_in_code_exists_in_locale():
    en = _load("en")
    used = set()
    for path in list(ROOT.glob("*.py")) + list((ROOT / "plantcare").glob("*.py")):
        used |= set(re.findall(
            r'"((?:head|msg|actions|symptom|care|ui)\.[A-Za-z_.]+)"',
            path.read_text(encoding="utf-8"),
        ))
    missing = sorted(k for k in used if k not in en)
    assert not missing, missing
