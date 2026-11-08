"""Streamlit user interface.  Run:  streamlit run app.py"""
import streamlit as st

from plantcare.care import build_care_report, load_care_kb
from plantcare.classifier import DiseaseClassifier, ModelNotFoundError
from plantcare.clip_engine import ClipEngine, ClipPlantGate, ClipSymptomDetector
from plantcare.i18n import available_languages, t
from plantcare.pipeline import DiagnosisPipeline
from plantcare.schemas import Readings

st.set_page_config(page_title="PlantCare AI", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading models (first run downloads CLIP)...")
def load_parts():
    engine = ClipEngine()
    try:
        clf = DiseaseClassifier()
    except ModelNotFoundError:
        clf = None
    return ClipPlantGate(engine), ClipSymptomDetector(engine), clf


def render_result(res, lang):
    tr = lambda k, **f: t(k, lang, **f)
    cat = res.status.category
    label = f"{tr('ui.status.' + cat)}: {res.headline}"
    # Status is shown in words AND colour/icon (not colour alone) for accessibility.
    {"healthy": st.success, "potentially_diseased": st.warning,
     "inconclusive": st.info, "rejected": st.error}[cat](label, icon={"healthy": "✅",
        "potentially_diseased": "⚠️", "inconclusive": "❓", "rejected": "⛔"}[cat])
    for m in res.messages:
        st.write(m)

    with st.expander(tr("ui.validation"), expanded=cat == "rejected"):
        if res.gate and res.gate.verdict == "ok":
            st.write(tr("ui.validation_ok"))
        if res.gate:
            c1, c2 = st.columns(2)
            c1.metric(tr("ui.plant_prob"), f"{res.gate.plant_prob:.0%}")
            c2.metric(tr("ui.leaf_prob"), f"{res.gate.leaf_prob:.0%}")
        if res.quality:
            c1, c2 = st.columns(2)
            c1.metric(tr("ui.sharpness"), res.quality.metrics["sharpness"])
            c2.metric(tr("ui.brightness"), res.quality.metrics["brightness"])

    if cat != "rejected":
        st.subheader(tr("ui.health"))
        st.markdown(f"**{tr('ui.species')}:** {res.species or tr('ui.species_unknown')}")
        if res.condition:
            st.markdown(f"**{tr('ui.disease')}:** {res.condition}")
        if res.confidence is not None:
            st.markdown(f"**{tr('ui.confidence')}:** {res.confidence:.0%}")
            st.progress(float(res.confidence))
            st.caption(tr("ui.confidence_help"))
        if res.symptoms:
            st.markdown(f"**{tr('ui.symptoms')}:** " +
                        ", ".join(f"{t('symptom.' + c.key, lang)} ({c.score:.0%})" for c in res.symptoms))
            report = build_care_report(res.species, symptoms=[c.key for c in res.symptoms], lang=lang)
            if report.symptom_causes:
                with st.expander(tr("ui.symptoms_vs_causes")):
                    for name, causes in report.symptom_causes:
                        st.markdown(f"**{name}**\n" + "\n".join(f"- {c}" for c in causes))
        if res.candidates and cat in ("inconclusive", "potentially_diseased"):
            with st.expander(tr("ui.candidates")):
                for c in res.candidates:
                    st.write(f"{c.species} - {c.condition}: {c.prob:.0%}")

    for w in res.warnings:
        st.warning(w)
    if res.actions:
        st.subheader(tr("ui.actions"))
        for a in res.actions:
            st.markdown(f"- {a}")
    if res.disease_info:
        with st.expander(tr("ui.about_disease") + f": {res.disease_info['name']}"):
            st.write(f"**{tr('ui.disease_symptoms')}:** {res.disease_info['symptoms']}")
    st.caption(f"ℹ️ {res.disclaimer}")


def optional_reading(column, key, default, minimum, maximum, lang):
    label = t(key, lang)
    if not column.checkbox(t("ui.provide_reading", lang, measurement=label)):
        return None
    return column.number_input(label, value=default, min_value=minimum, max_value=maximum)


def render_care(detected, symptoms, lang):
    tr = lambda k, **f: t(k, lang, **f)
    kb = load_care_kb()
    keys = [k for k in kb if not k.startswith("_")]
    st.write(tr("ui.care_intro"))
    options = ["__auto__"] + keys + ["__other__"]
    names = {"__auto__": tr("ui.care_species_auto") + (f" ({detected})" if detected else ""),
             "__other__": tr("ui.species_unknown")}
    choice = st.selectbox(tr("ui.care_species"), options,
                          format_func=lambda k: names.get(k) or kb[k]["display"])
    species = detected if choice == "__auto__" else (None if choice == "__other__" else choice)
    climate = st.text_input(tr("ui.care_climate"))
    st.markdown(f"**{tr('ui.care_readings')}**")
    c1, c2 = st.columns(2)
    moisture = optional_reading(c1, "ui.soil_moisture", 50.0, 0.0, 100.0, lang)
    temp = optional_reading(c2, "ui.air_temp", 20.0, -40.0, 60.0, lang)
    hum = optional_reading(c1, "ui.humidity", 50.0, 0.0, 100.0, lang)
    light = optional_reading(c2, "ui.light_hours", 6.0, 0.0, 24.0, lang)

    rep = build_care_report(species, Readings(moisture, temp, hum, light), symptoms, climate, lang)
    for s in rep.sections:
        with st.expander(s.title, expanded=False):
            st.write(s.guidance)
            for d in s.your_data:
                st.info(f"{tr('ui.your_data')}: {d}")
            st.caption(f"{tr('ui.needs_measurement')}: {s.needs_measurement}")
    st.subheader(tr("ui.questions"))
    for q in rep.questions:
        st.markdown(f"- {q}")
    st.subheader(tr("ui.preventive"))
    for p in rep.preventive:
        st.markdown(f"- {p}")
    st.subheader(tr("ui.escalate"))
    for e in rep.escalate:
        st.markdown(f"- {e}")


def main():
    gate, detector, clf = load_parts()
    langs = available_languages()
    lang = st.sidebar.selectbox(t("ui.language"), langs) if len(langs) > 1 else "en"
    st.title("🌿 " + t("ui.title", lang))
    st.write(t("ui.subtitle", lang))
    st.sidebar.subheader(t("ui.supported", lang))
    st.sidebar.caption(t("ui.supported_help", lang))
    st.sidebar.write(", ".join(clf.supported_species) if clf else t("msg.model_missing", lang))
    if clf and clf.metrics.get("test_accuracy") is not None:
        st.sidebar.metric(t("ui.model_info", lang), f"{clf.metrics['test_accuracy']:.0%}")
    st.sidebar.warning(t("msg.preliminary", lang))

    pipeline = DiagnosisPipeline(gate, clf, detector, getattr(clf, "metrics", {}), lang=lang)
    tab_diag, tab_care = st.tabs([t("ui.tab_diagnosis", lang), t("ui.tab_care", lang)])
    detected, symptoms = None, []
    with tab_diag:
        up = st.file_uploader(t("ui.upload", lang), type=["jpg", "jpeg", "png", "webp"],
                              help=t("ui.upload_help", lang))
        if up is not None:
            data = up.getvalue()
            st.image(data, caption=up.name, use_container_width=True)
            with st.spinner("..."):
                res = pipeline.analyze(data)
            render_result(res, lang)
            if res.status.category in ("healthy", "potentially_diseased"):
                detected = res.species
            symptoms = [c.key for c in res.symptoms]
    with tab_care:
        render_care(detected, symptoms, lang)


main()
