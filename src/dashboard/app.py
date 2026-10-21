import sys
from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# POLYLIFE - REAL ENVIRONMENTAL RUL DASHBOARD
# ============================================================
#
# Inputs:
#   Polymer type
#   Baseline polymer performance
#   Temperature
#   Humidity
#   UV irradiance
#   Optional UV dose
#   Current aging time
#   Failure threshold
#
# The dashboard calls the actual PolyLife inference engine:
#
#   src/models/polylife_environment_predictor.py
#
# Current validated model:
#   ExtraTrees degradation model
#   Validation R² ≈ 88.21%
#
# IMPORTANT:
#   88.21% refers to degradation-model R², NOT RUL accuracy.
#
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)


# ------------------------------------------------------------
# 2. ADD PROJECT ROOT TO PYTHON PATH
# ------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ------------------------------------------------------------
# 3. IMPORT POLYLIFE MODEL
# ------------------------------------------------------------

try:

    from src.models.polylife_environment_predictor import (
        predict_rul,
        SUPPORTED_POLYMERS,
        VALIDATED_TIME_MAX_HOURS,
    )
    from src.models.polymer_uv_experimental import predict_uv_degradation
except Exception as exc:

    st.error(
        "Could not load the PolyLife prediction engine."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# 4. PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "PolyLife | RUL Predictor",

    page_icon=
        "🌱",

    layout=
        "wide",

)


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.html("""
<style>
:root {
  --pl-ink: #10233f;
  --pl-muted: #61748b;
  --pl-blue: #2879d8;
  --pl-cyan: #42d6d0;
  --pl-coral: #ff795f;
  --pl-bg: #f1f8fc;
  --pl-panel: #ffffff;
  --pl-border: #dceaf2;
  --pl-soft: #e4f5f7;
}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {background:var(--pl-bg)!important;color:var(--pl-ink)}
.block-container {max-width:1500px;padding:1.2rem 2rem 3rem}
header[data-testid="stHeader"] {background:rgba(241,248,252,.94)}
#MainMenu,footer {visibility:hidden}
section[data-testid="stSidebar"] {display:none}
h1,h2,h3,h4 {color:var(--pl-ink)!important;letter-spacing:-.45px}
.stMarkdown p {color:#526980;line-height:1.65}
.pl-topbar {display:flex;align-items:center;justify-content:space-between;padding:.55rem 0 1rem;border-bottom:1px solid var(--pl-border);margin-bottom:1.3rem}
.pl-brand {display:flex;align-items:center;gap:.75rem}.pl-brand-icon {width:44px;height:44px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(145deg,var(--pl-blue),#164b94);color:#fff;font-size:1.35rem;font-weight:850;box-shadow:0 8px 22px #2879d82a}
.pl-brand-name {font-size:1.55rem;line-height:1.1;font-weight:850;letter-spacing:-.7px;color:var(--pl-ink)}.pl-brand-name span {color:var(--pl-blue)}.pl-brand-caption {margin-top:4px;color:var(--pl-muted);font-size:.63rem;letter-spacing:1.8px;font-weight:800}
.pl-topbar-tag {padding:.55rem .9rem;border:1px solid #c9e7e9;border-radius:999px;color:#177f86;background:#e3f8f5;font-size:.7rem;font-weight:800;letter-spacing:.8px}
.pl-hero {position:relative;overflow:hidden;display:grid;grid-template-columns:1.08fr .92fr;gap:1.4rem;align-items:center;padding:2.1rem 2.2rem;margin-bottom:1.3rem;border:1px solid #d7e9f0;border-radius:26px;background:radial-gradient(ellipse at 78% 12%,#c9f5f0 0,transparent 36%),linear-gradient(120deg,#fff 0%,#eef9fc 58%,#e5f6f7 100%);box-shadow:0 14px 38px #244f7110;animation:plRise .55s ease-out both}
.pl-eyebrow {display:flex;align-items:center;gap:.55rem;color:#60788e;font-size:.7rem;font-weight:850;letter-spacing:1.6px}.pl-eyebrow-dot {width:9px;height:9px;border-radius:50%;background:var(--pl-coral);box-shadow:0 0 0 5px #ff795f20}
.pl-hero h1 {font-size:clamp(2.15rem,3.7vw,3.7rem);line-height:1.03;letter-spacing:-2px;margin:1rem 0 .8rem;font-weight:850}.pl-hero h1 span {color:var(--pl-blue)}.pl-hero p {max-width:650px;font-size:1rem;color:#61748b;line-height:1.75}
.pl-hero-meta {display:flex;flex-wrap:wrap;gap:.6rem 1rem;margin-top:1.35rem;padding-top:1rem;border-top:1px solid #dceaf2;color:#71859a;font-size:.64rem;font-weight:850;letter-spacing:.7px}
.pl-hero-art {width:100%;min-height:260px;display:flex;align-items:center;justify-content:center}.pl-hero-art svg {width:100%;max-width:480px;height:auto;filter:drop-shadow(0 15px 22px #23648b15);animation:plFloat 5s ease-in-out infinite}
.pl-model-strip {display:flex;align-items:center;gap:.8rem;padding:1rem 1.15rem;margin:.8rem 0 1.2rem;border:1px solid #cfe8e7;border-radius:15px;background:linear-gradient(110deg,#e9f9f6,#fff);box-shadow:0 6px 20px #10233f07}.pl-status-dot {flex-shrink:0;width:10px;height:10px;border-radius:50%;background:#1eaa91;box-shadow:0 0 0 5px #1eaa9118}.pl-model-copy {display:flex;flex-direction:column;gap:3px}.pl-model-copy strong {color:var(--pl-ink);font-size:.95rem}.pl-model-copy span {color:var(--pl-muted);font-size:.83rem}.pl-model-label {margin-left:auto;color:#178a83;font-size:.66rem;font-weight:850;letter-spacing:1px}
/* Full-width navigation: stretch the Streamlit element AND its radio group. */
.block-container div[data-testid="stElementContainer"]:has(> div[data-testid="stRadio"]) {
  display:block!important;
  width:100%!important;
  max-width:none!important;
  min-width:0!important;
  align-self:stretch!important;
}
.block-container div[data-testid="stElementContainer"]:has(> div[data-testid="stRadio"]) > div,
.block-container div[data-testid="stElementContainer"]:has(> div[data-testid="stRadio"]) div[data-testid="stRadio"] {
  display:block!important;
  width:100%!important;
  max-width:none!important;
  min-width:0!important;
  box-sizing:border-box!important;
}
div[data-testid="stRadio"] {
  display:block!important; width:100%!important; max-width:none!important;
  min-width:0!important; margin:.25rem 0 1.45rem!important; padding:0!important;
}
div[data-testid="stRadio"] > label {display:none!important;}
div[data-testid="stRadio"] > div[role="radiogroup"] {
  display:grid!important;
  grid-template-columns:repeat(6,minmax(0,1fr))!important;
  align-items:stretch!important;
  justify-content:stretch!important;
  gap:.35rem!important;
  width:100%!important; min-width:0!important; max-width:none!important;
  box-sizing:border-box!important; overflow:visible!important;
  padding:.42rem!important; background:#e2f0f7!important;
  border:1px solid #cfe3ee!important; border-radius:16px!important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label {
  display:flex!important; width:100%!important; min-width:0!important;
  align-items:center!important; justify-content:center!important;
  white-space:normal!important; text-align:center!important;
  padding:.68rem .4rem!important; box-sizing:border-box!important;
  margin:0!important; border-radius:11px!important;
  border:1px solid transparent!important; background:transparent!important;
  color:#536a82!important; font-size:clamp(.7rem,.85vw,.88rem)!important;
  font-weight:700!important; transition:background .18s ease,box-shadow .18s ease,color .18s ease;
}
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {margin-right:.35rem!important;}
div[data-testid="stRadio"] label:has(input:checked) {background:#fff!important;border-color:#cfe4ef!important;color:var(--pl-blue)!important;box-shadow:0 4px 12px #173f5b0c}
div[data-testid="stRadio"] label p {font-size:inherit!important;white-space:nowrap!important;margin:0!important;}
div[data-testid="stRadio"] input {accent-color:#2879d8}
.pl-topbar {position:relative; z-index:2; min-height:66px; padding:.7rem 0 1rem; margin-bottom:.65rem!important;}
.pl-brand-name {color:#10233f!important; opacity:1!important;}
.pl-brand-caption {color:#61748b!important; opacity:1!important;}
.pl-topbar-tag {color:#177f86!important; opacity:1!important;}
@media(max-width:1000px) {
  div[data-testid="stRadio"] > div[role="radiogroup"] {grid-template-columns:repeat(3,minmax(0,1fr))!important;}
}
@media(max-width:520px) {
  div[data-testid="stRadio"] > div[role="radiogroup"] {grid-template-columns:repeat(2,minmax(0,1fr))!important;}
  div[data-testid="stRadio"] > div[role="radiogroup"] > label {font-size:.78rem!important;padding:.58rem .3rem!important;}
}
div[data-testid="stMetric"] {background:var(--pl-panel)!important;border:1px solid var(--pl-border)!important;border-radius:16px!important;padding:1rem!important;box-shadow:0 6px 20px #10233f08!important;transition:transform .2s ease,box-shadow .2s ease}
div[data-testid="stMetric"]:hover {transform:translateY(-3px);box-shadow:0 12px 26px #10233f12!important}div[data-testid="stMetricLabel"] {color:var(--pl-muted)!important}div[data-testid="stMetricValue"] {color:var(--pl-ink)!important;font-weight:800!important}
.stButton>button,.stDownloadButton>button {min-height:2.75rem;border-radius:11px!important;font-weight:750!important;transition:transform .18s ease,box-shadow .18s ease}.stButton>button[kind="primary"] {background:linear-gradient(110deg,#236fca,#1d56a4)!important;color:white!important;border:1px solid #236fca!important}.stButton>button[kind="primary"]:hover {background:linear-gradient(110deg,#168c94,#167f8b)!important;border-color:#168c94!important;transform:translateY(-2px);box-shadow:0 7px 18px #168c942b}
input,textarea {border-radius:9px!important}div[data-baseweb="select"]>div,div[data-baseweb="input"]>div {border-radius:10px!important;border-color:#d6e5ed!important;background:#fff!important}
div[data-testid="stExpander"] {background:#fff;border:1px solid var(--pl-border);border-radius:13px}div[data-testid="stVegaLiteChart"],div[data-testid="stArrowVegaLiteChart"],div[data-testid="stLineChart"],div[data-testid="stScatterChart"] {background:#fff;border:1px solid var(--pl-border);border-radius:15px;padding:.5rem;box-shadow:0 5px 18px #10233f06}div[data-testid="stDataFrame"] {border:1px solid var(--pl-border);border-radius:13px;overflow:hidden}div[data-testid="stAlert"] {border-radius:13px}hr {border-color:var(--pl-border)!important}
.section-title {font-size:1.65rem;font-weight:850;letter-spacing:-.7px;margin:1rem 0 .7rem;color:var(--pl-ink)}.pl-section-kicker {color:#168c94;font-size:.7rem;font-weight:850;letter-spacing:1.6px;text-transform:uppercase}.pl-page-title {font-size:2rem;font-weight:850;letter-spacing:-1px;margin:.25rem 0}.pl-page-desc {color:var(--pl-muted);margin-bottom:1.4rem}.pl-info-card {height:100%;padding:1.15rem 1.25rem;border:1px solid var(--pl-border);border-radius:17px;background:#fff;box-shadow:0 7px 22px #10233f07;transition:transform .2s ease,box-shadow .2s ease}.pl-info-card:hover {transform:translateY(-3px);box-shadow:0 13px 28px #10233f12}.pl-info-card h3 {font-size:1.02rem;margin:.2rem 0 .45rem}.pl-info-card p {font-size:.9rem;color:var(--pl-muted);margin:0}.pl-card-icon {font-size:1.5rem;margin-bottom:.65rem}.pl-footer {margin-top:2.2rem;padding-top:1rem;border-top:1px solid var(--pl-border);font-size:.78rem;color:#8a9aad}
.pl-step {height:100%;padding:1.1rem;border:1px solid var(--pl-border);border-radius:16px;background:#fff}.pl-step-num {display:inline-flex;width:35px;height:35px;align-items:center;justify-content:center;border-radius:11px;background:#dff7f4;color:#087f86;font-weight:850;margin-bottom:.7rem}.pl-step h3 {margin:.2rem 0 .4rem;font-size:1rem}.pl-step p {font-size:.88rem;color:var(--pl-muted);margin:0}.pl-image-caption {font-size:.8rem;color:var(--pl-muted);margin-top:.4rem}
@keyframes plRise {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}@keyframes plFloat {0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
@media(max-width:850px){.block-container{padding:.8rem .8rem 2rem}.pl-hero{grid-template-columns:1fr;padding:1.3rem}.pl-hero h1{letter-spacing:-1px}.pl-hero-art{min-height:170px}.pl-model-label{display:none}}
</style>
""")


# ============================================================
# 6. POLYLIFE PREMIUM HEADER
# ============================================================

st.markdown(
    """
    <div class="pl-topbar">
        <div class="pl-brand">
            <div class="pl-brand-icon">P</div>
            <div><div class="pl-brand-name">Poly<span>Life</span></div><div class="pl-brand-caption">POLYMER INTELLIGENCE</div></div>
        </div>
        <div class="pl-topbar-tag">● &nbsp; AI RESEARCH WORKSPACE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main navigation sits directly under the brand, above the page content.
page = st.radio(
    "Main navigation",
    ["Overview", "Predict", "How it works", "Model Insights", "Data Explorer", "About PolyLife"],
    horizontal=True,
    label_visibility="collapsed",
    key="polylife_navigation",
)

st.markdown(
    """
    <section class="pl-hero">
      <div class="pl-hero-copy">
        <div class="pl-eyebrow"><span class="pl-eyebrow-dot"></span>MATERIAL SCIENCE • DEGRADATION ANALYTICS</div>
        <h1>Materials change.<br><span>Understand what comes next.</span></h1>
        <p>Explore how temperature, humidity, UV exposure and aging relate to polymer performance. Run a scenario, review the charts and understand the limits behind every result.</p>
        <div class="pl-hero-meta"><span>01 MATERIAL</span><span>02 ENVIRONMENT</span><span>03 PREDICTION</span><span>04 INSIGHTS</span></div>
      </div>
      <div class="pl-hero-art" aria-label="Illustration of polymer science and data analysis">
        <svg viewBox="0 0 500 330" role="img" xmlns="http://www.w3.org/2000/svg">
          <defs><linearGradient id="plGlass" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ffffff"/><stop offset="1" stop-color="#dff8f7"/></linearGradient><linearGradient id="plBlue" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#51dcd1"/><stop offset="1" stop-color="#2879d8"/></linearGradient></defs>
          <circle cx="260" cy="160" r="128" fill="#d9f7f3" opacity=".72"/><circle cx="260" cy="160" r="100" fill="none" stroke="#8fdedb" stroke-width="1.5" stroke-dasharray="5 8"/>
          <path d="M60 244 C115 210 125 267 173 224 S252 252 295 193 S375 207 435 115" fill="none" stroke="#2879d8" stroke-width="4" stroke-linecap="round"/>
          <path d="M60 244 C115 210 125 267 173 224 S252 252 295 193 S375 207 435 115 L435 270 L60 270 Z" fill="#42d6d0" opacity=".16"/>
          <rect x="146" y="43" width="220" height="190" rx="22" fill="url(#plGlass)" stroke="#b8dfe8" stroke-width="2" transform="rotate(-7 256 138)"/>
          <rect x="170" y="65" width="70" height="9" rx="4.5" fill="#2879d8" opacity=".85" transform="rotate(-7 205 69)"/><rect x="170" y="87" width="150" height="6" rx="3" fill="#c0d6e3" transform="rotate(-7 245 90)"/><rect x="168" y="103" width="125" height="6" rx="3" fill="#c0d6e3" transform="rotate(-7 230 106)"/>
          <path d="M183 176 L216 151 L245 164 L277 126 L306 140 L337 105" fill="none" stroke="#ff795f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="183" cy="176" r="5" fill="#ff795f"/><circle cx="277" cy="126" r="5" fill="#ff795f"/><circle cx="337" cy="105" r="5" fill="#ff795f"/>
          <rect x="28" y="77" width="125" height="64" rx="15" fill="#fff" stroke="#d6e9ef"/><circle cx="49" cy="98" r="7" fill="#42d6d0"/><text x="64" y="102" font-size="11" font-weight="700" fill="#61748b">UV EXPOSURE</text><text x="45" y="124" font-size="15" font-weight="800" fill="#10233f">Environment</text>
          <rect x="329" y="229" width="142" height="64" rx="15" fill="#10233f"/><circle cx="351" cy="251" r="7" fill="#42d6d0"/><text x="365" y="255" font-size="10" font-weight="700" fill="#b7cadc">MODEL OUTPUT</text><text x="347" y="277" font-size="14" font-weight="800" fill="#ffffff">Insights</text>
          <circle cx="410" cy="60" r="19" fill="#ff795f" opacity=".95"/><path d="M402 60 l6 6 11 -13" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pl-model-strip">
        <div class="pl-status-dot"></div>
        <div class="pl-model-copy"><strong>PolyLife prediction engine</strong><span>PE scenario-based RUL • PP/HDPE/LDPE experimental UV estimates</span></div>
        <div class="pl-model-label">MODEL INFORMATION</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Scientific scope and model assumptions", expanded=False):
    st.markdown(
        f"""
        **Supported RUL scenario:** Polyethylene (PE). PP, HDPE and LDPE use the experimental UV estimate only.

        **Environmental inputs:** temperature, humidity, UV irradiance, optional UV dose and aging time.

        **Forecast assumption:** entered environmental conditions remain constant throughout the future scenario.

        **Validated age range:** {VALIDATED_TIME_MAX_HOURS:.0f} hours.

        **Model evaluation:** reported degradation-model R² is 88.21%; this is not RUL accuracy.

        **Failure threshold:** user-selected for scenario analysis; it is not a validated physical failure criterion.
        """
    )

# Navigation is rendered beneath the brand above.

if page != "Predict":
    if page == "Overview":
        st.markdown('<div class="pl-section-kicker">WORKSPACE OVERVIEW</div><div class="pl-page-title">Your polymer analysis workspace</div><div class="pl-page-desc">A clear starting point for exploring material degradation and performance scenarios.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="pl-info-card"><div class="pl-card-icon">🧪</div><h3>Material inputs</h3><p>Configure polymer type, baseline performance and environmental exposure in the Predict workspace.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pl-info-card"><div class="pl-card-icon">📈</div><h3>Prediction & charts</h3><p>Run the existing model and review its output, observed data and trajectory plots in one place.</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="pl-info-card"><div class="pl-card-icon">🔬</div><h3>Scientific context</h3><p>Review the supported scope, evaluation context and limitations before interpreting results.</p></div>', unsafe_allow_html=True)
        st.markdown("### Model at a glance")
        m1, m2, m3 = st.columns(3)
        m1.metric("RUL scenario", "PE only")
        m2.metric("Degradation-model R²", "88.21%")
        m3.metric("Validated age horizon", f"{VALIDATED_TIME_MAX_HOURS:.0f} h")
        st.info("The reported R² describes degradation-model evaluation, not RUL accuracy. PP, HDPE and LDPE results are experimental UV estimates only.")
        st.info("Select **Predict** in the navigation above to configure inputs and run the model.")

    elif page == "How it works":
        st.markdown('<div class="pl-section-kicker">THE PREDICTION JOURNEY</div><div class="pl-page-title">From material inputs to model insights</div><div class="pl-page-desc">A guided workflow to help you set up a scenario and interpret its output.</div>', unsafe_allow_html=True)
        step_cols = st.columns(4)
        steps = [
            ("01", "Choose material", "Select polyethylene for the PE performance/RUL scenario, or choose PP, HDPE or LDPE for the experimental UV estimate."),
            ("02", "Set conditions", "Enter baseline performance, temperature, humidity and UV irradiance. Add a measured UV dose if you have one."),
            ("03", "Define aging", "Enter current exposure time and, for PE, choose the performance threshold used by the scenario."),
            ("04", "Run & review", "Click Predict Polymer Degradation, then review the output, charts, warnings and downloadable CSV."),
        ]
        for col, step in zip(step_cols, steps):
            with col:
                st.markdown(f'<div class="pl-step"><div class="pl-step-num">{step[0]}</div><h3>{step[1]}</h3><p>{step[2]}</p></div>', unsafe_allow_html=True)
        st.markdown("### What happens behind the scenes?")
        st.markdown('<div class="pl-info-card"><h3>Inputs → inference function → results</h3><p>The dashboard passes your selected values to the existing PolyLife model functions. This UI redesign does not retrain the model or change its prediction calculations.</p></div>', unsafe_allow_html=True)
        st.info("Important: the PE remaining-useful-life value is a scenario estimate. PP, HDPE and LDPE currently display experimental UV estimates only—not validated physical degradation or RUL.")
        if st.button("Open Predict workspace", type="primary", use_container_width=True):
            st.session_state["polylife_navigation"] = "Predict"
            st.rerun()

    elif page == "Model Insights":
        st.markdown('<div class="pl-section-kicker">EVALUATION & LIMITATIONS</div><div class="pl-page-title">Model insights</div><div class="pl-page-desc">Interpret model outputs with the correct scientific context.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="pl-info-card"><div class="pl-card-icon">📊</div><h3>Degradation model</h3><p>The project reports an ExtraTrees degradation-model R² of 88.21% under its group-held-out evaluation. This score is not RUL accuracy.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="pl-info-card"><div class="pl-card-icon">⏱️</div><h3>RUL scenario</h3><p>RUL is a scenario estimate based on a user-selected performance threshold and constant future environmental conditions.</p></div>', unsafe_allow_html=True)
        st.markdown("### Supported prediction modes")
        st.dataframe(pd.DataFrame([
            {"Material": "Polyethylene (PE)", "Mode": "Performance / scenario-based RUL", "Interpretation": "Use within the stated model limits"},
            {"Material": "Polypropylene (PP)", "Mode": "Experimental UV estimate", "Interpretation": "Not validated physical degradation or RUL"},
            {"Material": "HDPE", "Mode": "Experimental UV estimate", "Interpretation": "Not validated physical degradation or RUL"},
            {"Material": "LDPE", "Mode": "Experimental UV estimate", "Interpretation": "Not validated physical degradation or RUL"},
        ]), hide_index=True, use_container_width=True)
        st.warning("Do not interpret the R² value as an accuracy percentage for remaining useful life.")

    elif page == "Data Explorer":
        st.markdown('<div class="pl-section-kicker">DATA WORKSPACE</div><div class="pl-page-title">Data explorer</div><div class="pl-page-desc">Check which processed polymer-aging data files are present in this project.</div>', unsafe_allow_html=True)
        data_roots = [PROJECT_ROOT / "Data" / "processed", PROJECT_ROOT / "Data" / "raw"]
        found_files = []
        for root in data_roots:
            if root.exists():
                for file in root.rglob("*"):
                    if file.is_file() and file.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                        try:
                            rel = file.relative_to(PROJECT_ROOT)
                        except ValueError:
                            rel = file
                        found_files.append({"Type": "Processed" if "processed" in str(root).lower() else "Raw", "File": str(rel), "Format": file.suffix.lower()})
        if found_files:
            st.metric("Data files found", len(found_files))
            st.dataframe(pd.DataFrame(found_files), hide_index=True, use_container_width=True)
        else:
            st.info("No CSV or Excel data files were found under Data/raw or Data/processed.")
        st.caption("This page lists files only; it does not modify datasets or change the prediction pipeline.")

    else:  # About PolyLife
        st.markdown('<div class="pl-section-kicker">PROJECT GUIDE</div><div class="pl-page-title">About PolyLife</div><div class="pl-page-desc">Polymer degradation analytics with transparent assumptions and model boundaries.</div>', unsafe_allow_html=True)
        image_cols = st.columns(3)
        with image_cols[0]:
            st.image("https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=900&q=80", use_container_width=True)
            st.markdown('<div class="pl-info-card"><h3>Material science</h3><p>PolyLife explores polymer performance under changing environmental exposure and aging conditions.</p></div>', unsafe_allow_html=True)
        with image_cols[1]:
            st.image("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=900&q=80", use_container_width=True)
            st.markdown('<div class="pl-info-card"><h3>Environmental inputs</h3><p>Temperature, humidity, UV irradiance and exposure time define the scenario you want to examine.</p></div>', unsafe_allow_html=True)
        with image_cols[2]:
            st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80", use_container_width=True)
            st.markdown('<div class="pl-info-card"><h3>Decision support</h3><p>Charts and reported metrics help you inspect model output, with scientific limitations kept visible.</p></div>', unsafe_allow_html=True)
        st.markdown("### What PolyLife does")
        st.markdown("PolyLife is a research dashboard for exploring polymer degradation and performance scenarios from environmental and aging inputs. It presents model outputs and their context; it is not a certification or physical testing replacement.")
        st.markdown("### Inputs used")
        st.markdown("""- Polymer type and baseline elongation at break
- Temperature and humidity
- UV irradiance and optional measured UV dose
- Current aging time and user-selected performance threshold""")
        st.warning("The reported 88.21% R² is a degradation-model evaluation metric, not RUL accuracy. PE RUL is scenario-based; PP, HDPE and LDPE outputs are experimental UV estimates only.")

    st.markdown('<div class="pl-footer">PolyLife · Polymer intelligence workspace · Research estimates require careful interpretation.</div>', unsafe_allow_html=True)
    st.stop()



# ============================================================
# 8. PREDICTION WORKSPACE
# ============================================================


# ============================================================
# 9. MATERIAL AND ENVIRONMENT INPUTS
# ============================================================

st.markdown('<div class="pl-section-kicker">MODEL INPUTS</div><div class="pl-page-title">Configure your experiment</div><div class="pl-page-desc">Set material properties and exposure conditions, then run the prediction. Results and charts appear below.</div>', unsafe_allow_html=True)


st.subheader("01 · Material")



polymer_type = st.selectbox(
    "Polymer type",
    options=[
        "Polyethylene",
        "Polypropylene (PP)",
        "High-Density Polyethylene (HDPE)",
        "Low-Density Polyethylene (LDPE)",
    ],
    help=(
        "Polyethylene is currently the only supported "
        "polymer for environmental RUL prediction. "
        "Other polymers are listed for future model validation."
    ),
)




baseline_performance = st.number_input(
    "Baseline elongation at break (%)",
    min_value=0.1,
    value=800.0,
    step=10.0,
    help=(
        "Enter the measured elongation at break of the "
        "healthy/reference polyethylene sample (%). "
        "The default 800% is an example input, not a "
        "verified measurement for your material."
    ),
)

st.caption(
    "Use your measured reference value when available. "
    "The default is for demonstration only."
)



st.subheader("02 · Environment")


temperature_C = st.number_input(

    "Temperature (°C)",

    min_value=-50.0,

    max_value=100.0,

    value=27.0,

    step=1.0,

)


humidity_percent = st.number_input(

    "Humidity (%)",

    min_value=0.0,

    max_value=100.0,

    value=77.0,

    step=1.0,

)


uv_irradiance_W_m2 = st.number_input(

    "UV irradiance (W/m²)",

    min_value=0.0,

    value=584.43,

    step=0.1,

)

if not (415.391 <= uv_irradiance_W_m2 <= 749.642):
    st.warning(
        "UV irradiance is outside the observed training range "
        "(415.39–749.64 W/m² for mean UV). "
        "This range is not a physical safety limit, and the model "
        "also expects a separate maximum-UV feature."
    )
use_manual_uv_dose = st.checkbox(

    "Enter measured UV dose manually",

    value=False,

)


manual_uv_dose = None


if use_manual_uv_dose:

    manual_uv_dose = st.number_input(

        "UV dose (J/m²)",

        min_value=0.0,

        value=1_000_000_000.0,

        step=100_000_000.0,

        format="%.2f",

    )


st.subheader("03 · Aging and failure definition")


aging_time_hours = st.number_input(

    "Current aging time (hours)",

    min_value=0.0,

    max_value=10000.0,

    value=500.0,

    step=10.0,

)


failure_threshold_percent = st.slider(

    "Failure threshold (% of baseline performance)",

    min_value=40,

    max_value=95,

    value=70,

    step=5,

    help=(
        "For example, 70% means the threshold is reached "
        "when performance falls to 70% of baseline."
    ),

)


# ============================================================
# 10. CURRENT INPUT SUMMARY
# ============================================================

st.markdown('<div class="pl-section-kicker">INPUT SUMMARY</div><div class="pl-page-title">Current component condition</div>', unsafe_allow_html=True)


input_col1, input_col2, input_col3 = st.columns(3)


with input_col1:

    st.metric(
        "Polymer",
        polymer_type,
    )


with input_col2:

    st.metric(
        "Baseline Performance",
        f"{baseline_performance:.1f}%",
    )


with input_col3:

    st.metric(
        "Current Aging",
        f"{aging_time_hours:.1f} h",
    )


env_col1, env_col2, env_col3 = st.columns(3)


with env_col1:

    st.metric(
        "Temperature",
        f"{temperature_C:.1f} °C",
    )


with env_col2:

    st.metric(
        "Humidity",
        f"{humidity_percent:.1f}%",
    )


with env_col3:

    st.metric(
        "UV Irradiance",
        f"{uv_irradiance_W_m2:.2f} W/m²",
    )


# ============================================================
# 11. UV DOSE INFORMATION
# ============================================================

if use_manual_uv_dose:

    st.info(
        f"Manual UV dose supplied: "
        f"{manual_uv_dose:,.0f} J/m²"
    )

else:

    estimated_uv_dose = (

        uv_irradiance_W_m2
        *
        aging_time_hours
        *
        3600.0

    )

    st.info(
        "UV dose is calculated from entered UV irradiance "
        "and aging time under the constant-condition scenario. "
        f"Estimated dose: {estimated_uv_dose:,.0f} J/m²"
    )


# ============================================================
# 12. PREDICTION BUTTON
# ============================================================

st.markdown('<div class="pl-section-kicker">MODEL INFERENCE</div><div class="pl-page-title">Run PolyLife prediction</div>', unsafe_allow_html=True)



is_pe = polymer_type == "Polyethylene"
is_experimental = polymer_type in {
    "Polypropylene (PP)",
    "High-Density Polyethylene (HDPE)",
    "Low-Density Polyethylene (LDPE)",
}

if is_experimental:
    st.warning(
        "Experimental UV degradation estimate only. "
        "This mode does not predict RUL or remaining lifetime."
    )

predict_button = st.button(
    "🔮 Predict Polymer Degradation",
    type="primary",
    use_container_width=True,
    disabled=not (is_pe or is_experimental),
)


if predict_button:

    try:

        # ----------------------------------------------------
        # RUN ACTUAL MODEL
        # ----------------------------------------------------

        if is_pe:
            result = predict_rul(
                polymer_type=polymer_type,
                baseline_performance=baseline_performance,
                temperature_C=temperature_C,
                humidity_percent=humidity_percent,
                uv_irradiance_W_m2=uv_irradiance_W_m2,
                aging_time_hours=aging_time_hours,
                performance_failure_threshold_percent=failure_threshold_percent,
                uv_dose_J_m2=(
                    manual_uv_dose if use_manual_uv_dose else None
                ),
            )
        else:
            material_map = {
                "Polypropylene (PP)": "PP",
                "High-Density Polyethylene (HDPE)": "HDPE",
                "Low-Density Polyethylene (LDPE)": "LDPE",
            }
            result = predict_uv_degradation(
                material=material_map[polymer_type],
                exposure_time_hours=aging_time_hours,
            )


    except Exception as exc:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(exc)
        )

        st.stop()


    # ========================================================
    # 13. RESULT HEADER
    # ========================================================

    st.markdown("---")

    # Experimental UV mode: show degradation estimate only, never RUL.
    if not is_pe:
        st.markdown(
            '<div class="section-title">3. Experimental UV Degradation Result</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Experimental estimate only. This output is not a validated physical "
            "degradation measurement and does not estimate remaining useful life (RUL)."
        )
        st.metric(
            "Raw degradation estimate (unbounded)",
            f"{result['raw_degradation_estimate']:.4f}",
        )
        st.write("**Material:**", result["material"])
        st.write(
            "**Exposure time:**",
            f"{result['exposure_time_hours']:.1f} hours",
        )
        st.caption(result["warning"])

        st.subheader("Observed Data vs Experimental Model")

        observed = result["observed_data"].copy()
        curve = result["prediction_curve"].copy()

        st.line_chart(
            curve.set_index("Exposure time (hours)")
        )

        st.scatter_chart(
            observed.set_index("Exposure time (hours)")
        )

        st.caption(
            "Line = experimental model estimate. Points = observed data "
            "from one sample trajectory. This is exploratory, not independent validation."
        )

        result_df = pd.DataFrame([{
            "material": result["material"],
            "sample_trajectory": result["sample_trajectory"],
            "exposure_time_hours": result["exposure_time_hours"],
            "raw_degradation_estimate": result["raw_degradation_estimate"],
            "prediction_type": result["prediction_type"],
            "validated_rul": result["validated_rul"],
            "warning": result["warning"],
        }])

        st.download_button(
            label="⬇️ Download Experimental Result",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="experimental_uv_degradation.csv",
            mime="text/csv",
        )

        st.stop()

    st.markdown('<div class="pl-section-kicker">RESULTS</div><div class="pl-page-title">PolyLife prediction result</div>', unsafe_allow_html=True)


    # ========================================================
    # 14. PRIMARY RESULTS
    # ========================================================

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )


    current_performance = result.get(
        "current_performance"
    )


    current_degradation = result.get(
        "current_degradation"
    )


    estimated_rul = result.get(
        "estimated_RUL_hours"
    )


    status = result.get(
        "status",
        "unknown"
    )


    # --------------------------------------------------------
    # Predicted performance
    # --------------------------------------------------------

    with result_col1:

        st.metric(

            "Predicted Performance",

            (
                f"{current_performance:.2f}%"
                if current_performance is not None
                else "—"
            ),

        )


    # --------------------------------------------------------
    # Degradation
    # --------------------------------------------------------

    with result_col2:

        st.metric(

            "Degradation",

            (
                f"{current_degradation * 100:.2f}%"
                if current_degradation is not None
                else "—"
            ),

        )


    # --------------------------------------------------------
    # RUL
    # --------------------------------------------------------

    with result_col3:

        if status == "outside_validated_age_range":

            st.metric(
                "Estimated RUL",
                "Not estimated",
            )

        elif status == "threshold_reached":

            st.metric(
                "Estimated RUL",
                "0.0 h",
            )

        elif status == "beyond_validated_forecast":

            st.metric(
                "RUL",
                f"> {estimated_rul:.0f} h",
            )

        else:

            st.metric(

                "Estimated RUL",

                (
                    f"{estimated_rul:.1f} h"
                    if estimated_rul is not None
                    else "—"
                ),

            )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    with result_col4:

        st.metric(
            "Status",
            status.replace(
                "_",
                " "
            ).title(),
        )


    # ========================================================
    # 15. RUL INTERVAL
    # ========================================================

    lower_rul = result.get(
        "lower_RUL_hours"
    )


    upper_rul = result.get(
        "upper_RUL_hours"
    )


    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True,
    )


    st.subheader(
        "RUL Uncertainty"
    )


    if status == "outside_validated_age_range":

        st.info(
            "RUL is not estimated because the current age "
            "exceeds the validated forecast horizon of "
            f"{result.get('forecast_limit_hours', 2016):.0f} hours. "
            "The displayed zero must not be interpreted as "
            "a confirmed physical failure."
        )

    elif (
        status
        ==
        "beyond_validated_forecast"
    ):

        st.write(
            f"**RUL interpretation:** "
            f"{result.get('RUL_interpretation', '')}"
        )


        st.write(
            f"**Remaining validated horizon:** "
            f"{result.get('remaining_validated_horizon_hours', 0):.1f} h"
        )


        if lower_rul is not None:

            st.write(
                f"**Lower scenario bound:** "
                f"{lower_rul:.1f} h"
            )


        st.warning(
            "The failure threshold was not reached within "
            "the model's validated time horizon. The result "
            "should therefore be interpreted as a lower-bound "
            "statement rather than an exact lifetime."
        )


    else:

        if (
            lower_rul is not None
            and
            upper_rul is not None
        ):

            st.write(
                f"Estimated RUL interval: "
                f"**{lower_rul:.1f} – {upper_rul:.1f} hours**"
            )

        elif estimated_rul is not None:

            st.write(
                f"Estimated RUL: "
                f"**{estimated_rul:.1f} hours**"
            )

        else:

            st.write(
                "RUL interval unavailable."
            )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # ========================================================
    # 16. DEGRADATION VISUAL
    # ========================================================

    st.subheader(
        "Current Degradation State"
    )


    if current_degradation is not None:

        st.progress(
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        current_degradation
                    )
                )
            )
        )


        st.caption(
            f"Normalized degradation: "
            f"{current_degradation:.4f}"
        )


    # ========================================================
    # 17. MODEL DETAILS
    # ========================================================

    with st.expander(
        "Model details",
        expanded=False,
    ):

        detail_data = {

            "Model":
                "ExtraTrees",

            "Validated degradation R²":
                "88.21%",

            "Polymer":
                polymer_type,

            "Baseline performance":
                f"{baseline_performance:.2f}%",

            "Temperature":
                f"{temperature_C:.2f} °C",

            "Humidity":
                f"{humidity_percent:.2f}%",

            "UV irradiance":
                f"{uv_irradiance_W_m2:.4f} W/m²",

            "Current aging":
                f"{aging_time_hours:.2f} hours",

            "Failure threshold":
                f"{failure_threshold_percent:.1f}% of baseline",

            "Forecast limit":
                f"{VALIDATED_TIME_MAX_HOURS:.0f} hours",

            "Uncertainty method":
                result.get(
                    "uncertainty_method",
                    "Prototype"
                ),

        }


        detail_df = pd.DataFrame({

            "Parameter":
                list(
                    detail_data.keys()
                ),

            "Value":
                list(
                    detail_data.values()
                ),

        })


        st.dataframe(

            detail_df,

            hide_index=True,

            use_container_width=True,

        )


    # ========================================================
    # 18. DOWNLOAD RESULT
    # ========================================================

    result_df = pd.DataFrame(
        [
            result
        ]
    )


    csv_data = (
        result_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(

        label=
            "⬇️ Download Prediction",

        data=
            csv_data,

        file_name=
            "polylife_prediction.csv",

        mime=
            "text/csv",

    )


    # ========================================================
    # 19. SCIENTIFIC WARNING
    # ========================================================

    st.markdown(
        """
        <div class="warning-box">

        <b>Important:</b>

        The current model is validated for environmental
        conditions to polymer degradation/performance
        prediction. Its reported 88.21% R² is <b>not</b>
        an 88.21% RUL-accuracy claim.

        The RUL is a scenario estimate based on the selected
        performance threshold and the assumption that the
        entered environmental conditions remain constant.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown('<div class="pl-footer">PolyLife · Temperature + humidity + UV + aging time → degradation → performance-based RUL scenario</div>', unsafe_allow_html=True)


