from pathlib import Path
from html import escape
import importlib.metadata as importlib_metadata
import importlib.util
import re
import subprocess
import sys


BASE_DIR = Path(__file__).parent
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"
REQUIRED_PACKAGES = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "pyarrow": "pyarrow",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
}


def pinned_requirements() -> dict[str, str]:
    pins = {}
    if not REQUIREMENTS_PATH.exists():
        return pins
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def ensure_runtime_dependencies():
    pins = pinned_requirements()
    needs_install = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            needs_install.append(package_name)
            continue
        expected_version = pins.get(package_name.lower())
        if not expected_version:
            continue
        try:
            installed_version = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            needs_install.append(package_name)
            continue
        if installed_version != expected_version:
            needs_install.append(package_name)

    if needs_install and REQUIREMENTS_PATH.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
        )


ensure_runtime_dependencies()

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ASSET_DIR = BASE_DIR / "streamlit_assets"
DATA_PATH = ASSET_DIR / "presentation_data.parquet"
MODEL_PATH = ASSET_DIR / "optimized_hgbr_model.joblib"
KMEANS_PATH = ASSET_DIR / "kmeans_personas.joblib"
SCALER_PATH = ASSET_DIR / "kmeans_scaler.joblib"

FEATURE_COLUMNS = ["job_zone", "median_debt", "tuition_in_state", "credential_level"]
CLUSTER_COLUMNS = ["median_debt", "median_salary", "job_zone"]

PERSONA_NAMES = {
    0: "Accessible Entry Pathways",
    1: "High-Earning Technical/Professional Tracks",
    2: "Extensive Prep, Mixed Payoff",
    3: "Mainstream Zone 4 Careers",
}

PERSONA_SHORT_NAMES = {
    "Accessible Entry Pathways": "Accessible Entry",
    "High-Earning Technical/Professional Tracks": "Technical / Professional",
    "Extensive Prep, Mixed Payoff": "Extensive Prep",
    "Mainstream Zone 4 Careers": "Mainstream Zone 4",
}

SECTION_LABELS = [
    "Overview",
    "Data Foundation",
    "Exploration",
    "Model Lab",
    "Performance",
    "Personas",
    "Decision Takeaways",
]

PALETTE = ["#2f6f9f", "#2f7d68", "#a97524", "#b84747", "#6b7280", "#7c3aed"]
CONTINUOUS_SCALE = ["#d7e9f7", "#74a9cf", "#2f6f9f", "#2f7d68", "#a97524"]

JOB_ZONE_INFO = {
    1: {
        "label": "Very little preparation",
        "plain": "Usually entry-level work with short demonstration or brief on-the-job training.",
        "credential": "Often no postsecondary credential required.",
    },
    2: {
        "label": "Some preparation",
        "plain": "Usually high-school-level preparation plus some experience, training, or job-specific instruction; O*NET often groups Zones 1 and 2 together.",
        "credential": "Often high school, certificate, or short training.",
    },
    3: {
        "label": "Medium preparation",
        "plain": "Often requires vocational training, related work experience, or an associate-level pathway.",
        "credential": "Often certificate, apprenticeship, or associate degree.",
    },
    4: {
        "label": "Considerable preparation",
        "plain": "Often requires a bachelor's-level pathway plus stronger technical, analytical, or professional preparation.",
        "credential": "Often bachelor's degree or equivalent experience.",
    },
    5: {
        "label": "Extensive preparation",
        "plain": "Usually advanced professional, graduate, research, or highly specialized preparation.",
        "credential": "Often master's, doctoral, or professional degree.",
    },
}

METRIC_INFO = {
    "R2": "Share of salary variation explained by the model features. Higher is better, but a modest value can be informative when real salaries depend on missing context.",
    "RMSE": "Typical error with extra penalty for very large misses. It rises when the model is surprised by high-variance salaries.",
    "MAE": "Average absolute prediction miss. This is the easiest error metric to read in dollars.",
    "Median Error": "The middle prediction miss. Half of visible records have lower absolute error and half have higher.",
}

DISPLAY_COLUMN_NAMES = {
    "occupation_title": "Occupation",
    "program_name": "Program",
    "institution_name": "Institution",
    "credential_level": "Credential",
    "job_zone_label": "Job Zone",
    "tuition_in_state": "In-state tuition",
    "median_debt": "Median debt",
    "median_salary": "Median salary",
    "predicted_salary": "Predicted salary",
    "absolute_error": "Absolute error",
    "salary_to_tuition": "Salary / tuition",
    "persona": "Persona",
}

SEARCH_COLUMNS = {
    "occupation_title": ("Occupation", 5),
    "program_name": ("Program", 4),
    "soc_code": ("SOC", 5),
    "credential_level": ("Credential", 3),
    "institution_name": ("Institution", 2),
    "persona": ("Persona", 2),
    "job_zone_label": ("Job Zone", 2),
    "job_zone_plain": ("Preparation meaning", 1),
}


st.set_page_config(
    page_title="Education ROI Intelligence",
    page_icon="data:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #07111f;
        --muted: #243449;
        --subtle: #506176;
        --line: #c4d0dc;
        --surface: #eef4f8;
        --panel: #ffffff;
        --blue: #1f6389;
        --green: #25715e;
        --gold: #9a6b20;
        --red: #a94343;
        --violet: #6554a3;
    }
    html, body {
        background:
            linear-gradient(180deg, #eaf2f7 0, #f8fafc 260px, #f4f7fa 100%) !important;
        color: var(--ink) !important;
    }
    .stApp {
        background: var(--surface) !important;
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main,
    .main,
    [data-testid="stHeader"] {
        background: var(--surface) !important;
        color: var(--ink) !important;
    }
    .main .block-container {
        padding-top: 0.4rem;
        padding-left: clamp(1.25rem, 2.2vw, 2.4rem);
        padding-right: clamp(1.25rem, 2.2vw, 2.4rem);
        max-width: 1380px;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    label,
    label *,
    p,
    li,
    span {
        color: var(--ink) !important;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink) !important;
    }
    h1 {
        font-size: clamp(2rem, 5vw, 2.85rem);
        line-height: 1.04;
    }
    h2 {
        font-size: clamp(1.25rem, 3vw, 1.65rem);
    }
    h3 {
        font-size: 1.08rem;
    }
    p, li, label, span {
        letter-spacing: 0;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] {
        background: #fbfdff !important;
        border-right: 1px solid var(--line) !important;
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarContent"] *,
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.05rem !important;
        margin-bottom: 0.45rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stPills"] {
        margin-bottom: 0.65rem;
    }
    [data-testid="stSidebar"] [data-testid="stPills"] button {
        min-height: 1.75rem !important;
        padding: 0.18rem 0.55rem !important;
        border-radius: 999px !important;
    }
    [data-testid="stSidebar"] [data-testid="stPills"] button p,
    [data-testid="stSidebar"] [data-testid="stPills"] button span {
        font-size: 0.78rem !important;
        line-height: 1.12 !important;
    }
    [data-testid="stRadioOption"] {
        border-radius: 7px;
        padding: 0.18rem 0.35rem;
    }
    [data-testid="stRadioOption"][data-selected="true"] {
        background: #e7f0f8 !important;
        border: 1px solid #9dc3df !important;
        font-weight: 700 !important;
    }
    .page-hero {
        position: relative;
        overflow: hidden;
        background: #ffffff;
        border: 1px solid #b8c7d7;
        border-radius: 14px;
        padding: 1.05rem 1.2rem;
        margin: 0 0 0.85rem;
        box-shadow: 0 18px 40px rgba(7, 17, 31, 0.08);
    }
    .page-hero:before {
        content: "";
        position: absolute;
        inset: 0;
        height: 8px;
        background: linear-gradient(90deg, rgba(31, 99, 137, 0.9), rgba(37, 113, 94, 0.85), rgba(154, 107, 32, 0.8));
    }
    .page-hero .kicker {
        color: var(--blue) !important;
        font-size: 0.78rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 0.35rem;
    }
    .page-hero .title {
        color: var(--ink) !important;
        font-size: clamp(1.9rem, 4.2vw, 2.75rem);
        font-weight: 850;
        line-height: 1.02;
        margin: 0;
        max-width: 920px;
    }
    .page-hero .subtitle {
        color: var(--muted) !important;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 900px;
        margin-top: 0.65rem;
    }
    .page-hero .chapter {
        position: absolute;
        right: 1rem;
        bottom: 0.45rem;
        color: rgba(7, 17, 31, 0.055) !important;
        font-size: clamp(3rem, 14vw, 7.5rem);
        font-weight: 900;
        line-height: 0.9;
        pointer-events: none;
    }
    [data-testid="stMultiSelect"],
    [data-testid="stTextInput"],
    [data-testid="stSlider"],
    [data-testid="stPills"] {
        color: var(--ink) !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="listbox"] *,
    [role="option"],
    [role="option"] * {
        background-color: #ffffff !important;
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stPills"] button,
    [data-testid="stPills"] button * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stPills"] button[aria-selected="true"] {
        background: #dcecf6 !important;
        border-color: #6da8cf !important;
    }
    input,
    textarea,
    [role="combobox"],
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
    }
    .lede {
        max-width: 980px;
        color: var(--muted) !important;
        font-size: 1.02rem;
        line-height: 1.55;
        margin: -0.4rem 0 1.1rem;
    }
    .hero-panel {
        background: var(--panel);
        border: 1px solid #b9c7d5;
        border-left: 6px solid var(--blue);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0 0.95rem;
        box-shadow: 0 12px 28px rgba(7, 17, 31, 0.06);
    }
    .hero-panel h3 {
        margin: 0 0 0.35rem;
        font-size: 1.05rem;
    }
    .hero-panel p {
        margin: 0;
        color: var(--muted) !important;
        line-height: 1.52;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
        gap: 0.7rem;
        margin: 0.75rem 0 0.95rem;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid #b8c7d7;
        border-top: 5px solid var(--accent);
        border-radius: 10px;
        padding: 0.8rem 0.82rem;
        min-height: 92px;
        box-shadow: 0 10px 22px rgba(7, 17, 31, 0.055);
    }
    .metric-card .label {
        color: var(--subtle) !important;
        font-size: 0.72rem;
        font-weight: 850;
        text-transform: uppercase;
        line-height: 1.2;
    }
    .metric-card .value {
        color: var(--ink) !important;
        font-size: 1.36rem;
        font-weight: 850;
        line-height: 1.05;
        margin-top: 0.35rem;
        word-break: break-word;
    }
    .metric-card .caption {
        color: var(--subtle) !important;
        font-size: 0.74rem;
        line-height: 1.25;
        margin-top: 0.35rem;
    }
    .zone-ladder {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(138px, 1fr));
        gap: 0.62rem;
        margin: 0.55rem 0 0.8rem;
    }
    .zone-step-card {
        background: #ffffff;
        border: 1px solid #b8c7d7;
        border-left: 5px solid var(--accent);
        border-radius: 10px;
        padding: 0.74rem 0.78rem;
        min-height: 124px;
        box-shadow: 0 8px 18px rgba(7, 17, 31, 0.045);
    }
    .zone-step-card.selected {
        background: #f0f7fb;
        border-color: #6da8cf;
        box-shadow: 0 12px 24px rgba(31, 99, 137, 0.12);
    }
    .zone-step-card .zone {
        color: var(--blue) !important;
        font-size: 0.76rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
    }
    .zone-step-card .salary {
        color: var(--ink) !important;
        font-size: 1.12rem;
        font-weight: 850;
        margin-top: 0.26rem;
        line-height: 1.05;
    }
    .zone-step-card .meaning {
        color: var(--subtle) !important;
        font-size: 0.78rem;
        line-height: 1.28;
        margin-top: 0.34rem;
    }
    .result-panel {
        background: #ffffff;
        border: 1px solid #b8c7d7;
        border-left: 6px solid var(--green);
        border-radius: 12px;
        padding: 0.95rem 1rem;
        margin: 0.25rem 0 0.85rem;
        box-shadow: 0 12px 28px rgba(7, 17, 31, 0.055);
    }
    .result-panel .eyebrow {
        color: var(--green) !important;
        font-size: 0.74rem;
        font-weight: 850;
        letter-spacing: 0.07rem;
        text-transform: uppercase;
    }
    .result-panel .value {
        color: var(--ink) !important;
        font-size: clamp(2rem, 4.6vw, 3.1rem);
        font-weight: 900;
        line-height: 1;
        margin-top: 0.28rem;
    }
    .result-panel .context {
        color: var(--muted) !important;
        font-size: 0.92rem;
        line-height: 1.42;
        margin-top: 0.45rem;
    }
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.55rem;
        margin-top: 0.75rem;
    }
    .stat-strip .stat {
        background: #f5f8fb;
        border: 1px solid #d3dde7;
        border-radius: 8px;
        padding: 0.52rem 0.58rem;
    }
    .stat-strip .stat .label {
        color: var(--subtle) !important;
        font-size: 0.68rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    .stat-strip .stat .value {
        color: var(--ink) !important;
        font-size: 0.98rem;
        font-weight: 850;
        margin-top: 0.2rem;
        line-height: 1.12;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 0.95rem;
    }
    .insight-card {
        background: var(--panel);
        border: 1px solid #b8c7d7;
        border-left: 5px solid var(--accent);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        min-height: 86px;
        box-shadow: 0 8px 20px rgba(7, 17, 31, 0.045);
    }
    .insight-card .title {
        color: var(--ink) !important;
        font-weight: 850;
        margin-bottom: 0.28rem;
    }
    .insight-card .body {
        color: var(--subtle) !important;
        line-height: 1.42;
        font-size: 0.93rem;
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.55rem;
        margin: 0.4rem 0 1rem;
    }
    .flow-step {
        background: var(--panel);
        border: 1px solid #b8c7d7;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 6px 16px rgba(7, 17, 31, 0.04);
    }
    .flow-step .num {
        color: #2f6f9f;
        font-weight: 800;
        font-size: 0.78rem;
    }
    .flow-step .name {
        color: var(--ink) !important;
        font-weight: 760;
        margin-top: 0.18rem;
    }
    .flow-step .detail {
        color: var(--subtle) !important;
        font-size: 0.84rem;
        line-height: 1.35;
        margin-top: 0.22rem;
    }
    .callout {
        background: #ffffff;
        border: 1px solid #b8c7d7;
        border-left: 6px solid var(--blue);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.85rem 0 1.1rem;
        color: var(--ink) !important;
        box-shadow: 0 8px 18px rgba(7, 17, 31, 0.045);
    }
    .search-preview {
        background: #ffffff;
        border: 1px solid #c4d0dc;
        border-radius: 10px;
        padding: 0.65rem 0.7rem;
        margin: 0.45rem 0;
        box-shadow: 0 6px 14px rgba(7, 17, 31, 0.04);
    }
    .search-preview .title {
        font-size: 0.82rem;
        font-weight: 850;
        line-height: 1.18;
        color: var(--ink) !important;
    }
    .search-preview .meta {
        font-size: 0.72rem;
        line-height: 1.25;
        margin-top: 0.25rem;
        color: var(--subtle) !important;
    }
    .search-preview .salary {
        font-size: 0.78rem;
        font-weight: 800;
        margin-top: 0.28rem;
        color: var(--green) !important;
    }
    .search-status {
        background: #ffffff;
        border: 1px solid #b8c7d7;
        border-left: 6px solid var(--green);
        border-radius: 10px;
        padding: 0.72rem 0.9rem;
        margin: 0.15rem 0 0.85rem;
        box-shadow: 0 8px 18px rgba(7, 17, 31, 0.045);
    }
    .search-status .label {
        color: var(--green) !important;
        font-size: 0.74rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.07rem;
        margin-bottom: 0.18rem;
    }
    .search-status .text {
        color: var(--muted) !important;
        font-size: 0.9rem;
        line-height: 1.35;
    }
    .search-status strong {
        color: var(--ink) !important;
    }
    .section-kicker {
        color: var(--muted) !important;
        font-size: 0.88rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05rem;
        margin-bottom: 0.25rem;
    }
    .story-band {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 0.8rem;
        margin: 0.75rem 0 0.85rem;
    }
    .story-panel {
        background: #0f2537;
        border-radius: 12px;
        padding: 1rem 1.05rem;
        min-height: 108px;
        box-shadow: 0 14px 30px rgba(7, 17, 31, 0.13);
    }
    .story-panel.alt {
        background: #17392f;
    }
    .story-panel .eyebrow {
        color: #b9d5e8 !important;
        font-size: 0.74rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 0.35rem;
    }
    .story-panel .headline {
        color: #ffffff !important;
        font-size: 1.22rem;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 0.42rem;
    }
    .story-panel .copy {
        color: #e7f2f8 !important;
        line-height: 1.45;
        font-size: 0.93rem;
    }
    .story-panel * {
        color: inherit !important;
    }
    .divider-label {
        margin: 1rem 0 0.4rem;
        color: var(--blue) !important;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
    }
    .stPlotlyChart {
        background: #ffffff;
        border: 1px solid #c4d0dc;
        border-radius: 12px;
        padding: 0.2rem;
        box-shadow: 0 10px 24px rgba(7, 17, 31, 0.045);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: #ffffff !important;
    }
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    [data-testid="stAlert"],
    [data-testid="stAlert"] * {
        color: #111827 !important;
    }
    details {
        color: var(--ink) !important;
        background-color: #ffffff !important;
    }
    details summary,
    details summary * {
        color: var(--ink) !important;
        opacity: 1 !important;
    }
    @media (max-width: 760px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .metric-card .value {
            font-size: 1.35rem;
        }
        .story-band {
            grid-template-columns: 1fr;
        }
        .page-hero .chapter {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def dollars(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def compact_dollars(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return dollars(value)


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def job_zone_label(value: float) -> str:
    if pd.isna(value):
        return "Unknown Job Zone"
    zone = int(value)
    info = JOB_ZONE_INFO.get(zone, {"label": "Unknown preparation"})
    return f"Zone {zone}: {info['label']}"


def job_zone_plain(value: float) -> str:
    if pd.isna(value):
        return "Unknown preparation requirements."
    zone = int(value)
    return JOB_ZONE_INFO.get(zone, {"plain": "Unknown preparation requirements."})["plain"]


def job_zone_short_label(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    zone = int(value)
    info = JOB_ZONE_INFO.get(zone, {"label": "Unknown"})
    return f"Zone {zone}: {info['label'].replace(' preparation', '')}"


def persona_short_name(value: str) -> str:
    return PERSONA_SHORT_NAMES.get(str(value), str(value))


def add_page_header(title: str, subtitle: str, chapter: str) -> None:
    st.markdown(
        (
            "<div class='page-hero'>"
            "<div class='kicker'>Education ROI Intelligence</div>"
            f"<div class='title'>{escape(title)}</div>"
            f"<div class='subtitle'>{escape(subtitle)}</div>"
            f"<div class='chapter'>{escape(chapter)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def add_callout(text: str) -> None:
    st.markdown(f"<div class='callout'>{text}</div>", unsafe_allow_html=True)


def story_band(left: dict[str, str], right: dict[str, str]) -> None:
    def panel(item: dict[str, str], alt: bool = False) -> str:
        klass = "story-panel alt" if alt else "story-panel"
        return (
            f"<div class='{klass}'>"
            f"<div class='eyebrow'>{escape(item['eyebrow'])}</div>"
            f"<div class='headline'>{escape(item['headline'])}</div>"
            f"<div class='copy'>{escape(item['copy'])}</div>"
            "</div>"
        )

    st.markdown(f"<div class='story-band'>{panel(left)}{panel(right, True)}</div>", unsafe_allow_html=True)


def divider_label(text: str) -> None:
    st.markdown(f"<div class='divider-label'>{escape(text)}</div>", unsafe_allow_html=True)


def job_zone_legend(zones: list[int]) -> None:
    cards = []
    for zone in sorted(set(int(z) for z in zones if z in JOB_ZONE_INFO)):
        info = JOB_ZONE_INFO[zone]
        cards.append(
            {
                "title": f"Zone {zone}: {info['label']}",
                "body": f"{info['plain']} {info['credential']}",
            }
        )
    insight_grid(cards)
    st.caption("O*NET Job Zones summarize the education, related experience, and on-the-job training typically needed for an occupation.")


def metric_explainer(metrics: list[str]) -> None:
    insight_grid(
        [
            {"title": metric, "body": METRIC_INFO[metric]}
            for metric in metrics
            if metric in METRIC_INFO
        ]
    )


def metric_grid(items: list[dict[str, str]]) -> None:
    accents = ["#2f6f9f", "#2f7d68", "#a97524", "#b84747", "#6b7280", "#7c3aed"]
    cards = []
    for i, item in enumerate(items):
        label = escape(str(item["label"]))
        value = escape(str(item["value"]))
        caption = escape(str(item.get("caption", "")))
        accent = item.get("accent", accents[i % len(accents)])
        caption_html = f"<div class='caption'>{caption}</div>" if caption else ""
        cards.append(
            f"<div class='metric-card' style='--accent:{accent};'>"
            f"<div class='label'>{label}</div><div class='value'>{value}</div>{caption_html}</div>"
        )
    st.markdown(f"<div class='metric-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def result_panel(eyebrow: str, value: str, context: str, stats: list[dict[str, str]]) -> None:
    stat_html = []
    for stat in stats:
        stat_html.append(
            "<div class='stat'>"
            f"<div class='label'>{escape(str(stat['label']))}</div>"
            f"<div class='value'>{escape(str(stat['value']))}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='result-panel'>"
        f"<div class='eyebrow'>{escape(eyebrow)}</div>"
        f"<div class='value'>{escape(value)}</div>"
        f"<div class='context'>{escape(context)}</div>"
        f"<div class='stat-strip'>{''.join(stat_html)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def zone_prediction_ladder(model, zones: list[int], debt: float, tuition: float, credential: str, selected_zone: int) -> None:
    accents = ["#2f6f9f", "#2f7d68", "#a97524", "#b84747", "#6b7280"]
    cards = []
    for i, zone in enumerate(zones):
        info = JOB_ZONE_INFO.get(int(zone), {"label": "Unknown", "plain": "Unknown preparation requirements."})
        salary = build_prediction(model, int(zone), debt, tuition, credential)
        selected = " selected" if int(zone) == int(selected_zone) else ""
        cards.append(
            f"<div class='zone-step-card{selected}' style='--accent:{accents[i % len(accents)]};'>"
            f"<div class='zone'>Zone {int(zone)}: {escape(info['label'].replace(' preparation', ''))}</div>"
            f"<div class='salary'>{escape(compact_dollars(salary))}</div>"
            f"<div class='meaning'>{escape(info['plain'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='zone-ladder'>{''.join(cards)}</div>", unsafe_allow_html=True)


def insight_grid(items: list[dict[str, str]]) -> None:
    accents = ["#2f6f9f", "#2f7d68", "#a97524", "#b84747", "#7c3aed"]
    cards = []
    for i, item in enumerate(items):
        title = escape(str(item["title"]))
        body = escape(str(item["body"]))
        accent = item.get("accent", accents[i % len(accents)])
        cards.append(
            f"<div class='insight-card' style='--accent:{accent};'>"
            f"<div class='title'>{title}</div><div class='body'>{body}</div></div>"
        )
    st.markdown(f"<div class='insight-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def flow_grid(items: list[dict[str, str]]) -> None:
    steps = []
    for i, item in enumerate(items, start=1):
        name = escape(str(item["name"]))
        detail = escape(str(item["detail"]))
        steps.append(
            f"<div class='flow-step'><div class='num'>STEP {i}</div>"
            f"<div class='name'>{name}</div><div class='detail'>{detail}</div></div>"
        )
    st.markdown(f"<div class='flow-grid'>{''.join(steps)}</div>", unsafe_allow_html=True)


def tune_fig(fig: go.Figure, height: int = 430) -> go.Figure:
    raw_title = fig.layout.title.text
    title_text = "" if raw_title in (None, "", "undefined") else raw_title
    show_legend = bool(fig.layout.showlegend) if fig.layout.showlegend is not None else any(
        getattr(trace, "showlegend", None) is not False and getattr(trace, "name", None)
        for trace in fig.data
    )
    top_margin = 70 if title_text else (56 if show_legend else 24)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=18, r=18, t=top_margin, b=46),
        font=dict(family="Inter, Arial, sans-serif", color="#111827"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        title=dict(text=title_text, font=dict(size=16, color="#07111f"), x=0.02, xanchor="left"),
        hoverlabel=dict(bgcolor="#07111f", font_size=12, font_color="#ffffff"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6edf3", zeroline=False, linecolor="#9eb0c0", tickfont=dict(color="#263445"))
    fig.update_yaxes(showgrid=True, gridcolor="#e6edf3", zeroline=False, linecolor="#9eb0c0", tickfont=dict(color="#263445"))
    return fig


@st.cache_data(show_spinner="Loading presentation data...")
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH).copy()
    for col in ["job_zone", "median_salary", "median_debt", "tuition_in_state"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["credential_level", "program_name", "occupation_title", "institution_name", "soc_code"]:
        df[col] = df[col].fillna("Unknown")
    df["job_zone_label"] = df["job_zone"].map(job_zone_label)
    df["job_zone_plain"] = df["job_zone"].map(job_zone_plain)
    return df


@st.cache_resource(show_spinner="Loading trained artifacts...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, kmeans, scaler


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].copy()
    tuition_median = features["tuition_in_state"].median()
    debt_median = features["median_debt"].median()
    features["tuition_in_state"] = features["tuition_in_state"].fillna(15000 if pd.isna(tuition_median) else tuition_median)
    features["median_debt"] = features["median_debt"].fillna(25000 if pd.isna(debt_median) else debt_median)
    return features


@st.cache_data(show_spinner="Scoring salary model and personas...")
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    model, kmeans, scaler = load_artifacts()
    enriched = df.copy()
    enriched["predicted_salary"] = model.predict(feature_frame(enriched))
    enriched["absolute_error"] = (enriched["median_salary"] - enriched["predicted_salary"]).abs()
    enriched["residual"] = enriched["median_salary"] - enriched["predicted_salary"]
    enriched["salary_to_tuition"] = enriched["median_salary"] / enriched["tuition_in_state"].replace(0, np.nan)

    cluster_source = enriched[CLUSTER_COLUMNS].copy()
    for col in CLUSTER_COLUMNS:
        cluster_source[col] = cluster_source[col].fillna(cluster_source[col].median())
    enriched["persona_id"] = kmeans.predict(scaler.transform(cluster_source))
    enriched["persona"] = enriched["persona_id"].map(PERSONA_NAMES).fillna(
        enriched["persona_id"].map(lambda value: f"Cluster {value}")
    )
    return enriched


def search_tokens(query: str) -> list[str]:
    return [token for token in re.split(r"\s+", query.lower().strip()) if token]


def score_search_results(results: pd.DataFrame, query: str, tokens: list[str]) -> pd.DataFrame:
    if results.empty:
        return results

    query_l = query.lower().strip()
    scored = results.copy()
    scores = pd.Series(0, index=scored.index, dtype="int64")
    matched_fields = pd.Series("", index=scored.index, dtype="object")

    for col, (label, weight) in SEARCH_COLUMNS.items():
        if col not in scored.columns:
            continue
        values = scored[col].fillna("").astype(str).str.lower()
        any_token = pd.Series(False, index=scored.index)
        all_tokens = pd.Series(True, index=scored.index)
        for token in tokens:
            contains_token = values.str.contains(token, regex=False)
            any_token |= contains_token
            all_tokens &= contains_token
        phrase_match = values.str.contains(query_l, regex=False) if query_l else pd.Series(False, index=scored.index)
        starts_match = values.str.startswith(tokens[0]) if tokens else pd.Series(False, index=scored.index)
        scores += (
            (any_token.astype(int) * weight)
            + (all_tokens.astype(int) * weight)
            + (phrase_match.astype(int) * weight * 2)
            + (starts_match.astype(int) * 2)
        )
        matched_fields = matched_fields.mask(any_token, matched_fields + label + ", ")

    scored["search_score"] = scores
    scored["matched_fields"] = matched_fields.str.rstrip(", ").replace("", "Pathway")
    return scored.sort_values(["search_score", "median_salary"], ascending=[False, False])


def apply_pathway_search(df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, dict[str, object]]:
    tokens = search_tokens(query)
    state = {
        "query": query.strip(),
        "tokens": tokens,
        "active": bool(tokens),
        "pre_search_count": len(df),
        "match_mode": "exact",
    }
    if not tokens:
        return df.copy(), state

    search_cols = [col for col in SEARCH_COLUMNS if col in df.columns]
    combined = df[search_cols].astype(str).agg(" ".join, axis=1).str.lower()
    exact_mask = pd.Series(True, index=df.index)
    partial_mask = pd.Series(False, index=df.index)
    for token in tokens:
        token_match = combined.str.contains(token, regex=False)
        exact_mask &= token_match
        partial_mask |= token_match

    results = df[exact_mask].copy()
    if results.empty and len(tokens) > 1:
        results = df[partial_mask].copy()
        state["match_mode"] = "partial"

    state["matches"] = len(results)
    if results.empty:
        return results, state

    return score_search_results(results, query, tokens), state


def render_search_status(state: dict[str, object]) -> None:
    if not state.get("active"):
        return

    matches = int(state.get("matches", 0))
    pre_count = int(state.get("pre_search_count", matches))
    query = escape(str(state.get("query", "")).strip())
    mode = "all search terms" if state.get("match_mode") == "exact" else "partial-term fallback"
    st.markdown(
        "<div class='search-status'>"
        "<div class='label'>Active pathway search</div>"
        f"<div class='text'>Showing <strong>{matches:,}</strong> of <strong>{pre_count:,}</strong> filtered pathways for "
        f"<strong>{query}</strong>, ranked by the strongest matches in occupation, program, SOC code, credential, institution, persona, and Job Zone meaning. "
        f"Match mode: <strong>{mode}</strong>.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar_search_preview(df: pd.DataFrame, state: dict[str, object]) -> None:
    if not state.get("active"):
        st.sidebar.caption("Search matches words across occupations, programs, institutions, credentials, SOC codes, personas, and Job Zone meanings.")
        return

    matches = int(state.get("matches", len(df)))
    pre_count = int(state.get("pre_search_count", matches))
    if state.get("match_mode") == "partial":
        st.sidebar.caption(f"No all-word match; showing {matches:,} partial matches from {pre_count:,} filtered pathways.")
    else:
        st.sidebar.caption(f"Search matched {matches:,} of {pre_count:,} filtered pathways.")
    if df.empty:
        st.sidebar.info("No matches. Try a broader term such as software, nursing, business, certificate, or a SOC prefix.")
        return

    st.sidebar.markdown("**Top matches**")
    cards = []
    for _, row in df.head(4).iterrows():
        title = escape(str(row["occupation_title"]))
        program = escape(str(row["program_name"]))
        meta = escape(f"{row['credential_level']} | {row['job_zone_label']} | {row.get('matched_fields', 'Pathway')}")
        salary = escape(f"{compact_dollars(row['median_salary'])} median salary")
        cards.append(
            "<div class='search-preview'>"
            f"<div class='title'>{title}</div>"
            f"<div class='meta'>{program}</div>"
            f"<div class='meta'>{meta}</div>"
            f"<div class='salary'>{salary}</div>"
            "</div>"
        )
    st.sidebar.markdown("".join(cards), unsafe_allow_html=True)


def sidebar_filters(df: pd.DataFrame) -> tuple[str, pd.DataFrame, dict[str, object]]:
    st.sidebar.header("Supplement Navigator")
    section = st.sidebar.radio("Section", SECTION_LABELS, label_visibility="collapsed")
    st.sidebar.divider()

    zone_options = sorted(df["job_zone"].dropna().astype(int).unique().tolist())
    zones = st.sidebar.pills(
        "Job Zone",
        zone_options,
        selection_mode="multi",
        default=zone_options,
        format_func=job_zone_short_label,
        width="stretch",
    )
    zones = zones or zone_options

    salary_min = int(df["median_salary"].min())
    salary_max = int(df["median_salary"].max())
    salary_range = st.sidebar.slider(
        "Median salary",
        salary_min,
        salary_max,
        (salary_min, salary_max),
        step=5000,
        format="$%d",
    )

    query = st.sidebar.text_input("Search pathways", placeholder="try: nursing bachelor, software, 15-")
    filtered = df[
        df["job_zone"].astype(int).isin(zones)
        & df["median_salary"].between(salary_range[0], salary_range[1])
    ].copy()

    filtered, search_state = apply_pathway_search(filtered, query)

    st.sidebar.caption(f"{len(filtered):,} of {len(df):,} pathways visible")
    sidebar_search_preview(filtered, search_state)
    return section, filtered, search_state


def metrics_summary(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {}
    y_true = df["median_salary"]
    y_pred = df["predicted_salary"]
    return {
        "records": len(df),
        "occupations": df["occupation_title"].nunique(),
        "programs": df["program_name"].nunique(),
        "institutions": df["institution_name"].nunique(),
        "r2": r2_score(y_true, y_pred) if len(df) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_salary": float(df["median_salary"].median()),
        "median_tuition": float(df["tuition_in_state"].median()),
    }


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for col in ["median_salary", "predicted_salary", "absolute_error", "median_debt", "tuition_in_state"]:
        if col in table:
            table[col] = table[col].map(dollars)
    if "salary_to_tuition" in table:
        table["salary_to_tuition"] = table["salary_to_tuition"].map(lambda value: f"{value:.1f}x" if pd.notna(value) else "N/A")
    table = table.rename(columns=DISPLAY_COLUMN_NAMES)
    return table


def overview_page(df: pd.DataFrame) -> None:
    summary = metrics_summary(df)
    add_page_header(
        "Education ROI Intelligence",
        "An interactive supplement that connects education costs, credentials, occupational preparation, and expected salary outcomes.",
        "01",
    )

    metric_grid(
        [
            {"label": "Pathways", "value": f"{summary['records']:,}", "caption": "Visible records"},
            {"label": "Occupations", "value": f"{summary['occupations']:,}", "caption": "SOC-linked roles"},
            {"label": "Programs", "value": f"{summary['programs']:,}", "caption": "CIP-linked programs"},
            {"label": "Median Salary", "value": compact_dollars(summary["median_salary"]), "caption": "Outcome signal"},
            {"label": "Model R2", "value": f"{summary['r2']:.3f}", "caption": "Signal captured"},
        ]
    )

    left, right = st.columns([0.92, 1.08])
    with left:
        st.markdown(
            """
            <div class='hero-panel'>
              <h3>Central finding</h3>
              <p>Preparation level is a stronger salary signal than education cost alone. The model is useful because it quantifies both the signal and the uncertainty in education-to-career ROI.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        insight_grid(
            [
                {"title": "Unified pathway", "body": "CIP-to-SOC mapping links education programs to occupations and wages."},
                {"title": "Transparent limitation", "body": "A modest R2 shows signal without pretending cost determines salary."},
            ]
        )
    with right:
        divider_label("Salary outcome by preparation level")
        zone_summary = (
            df.groupby("job_zone", as_index=False)
            .agg(median_salary=("median_salary", "median"), pathways=("median_salary", "size"))
            .sort_values("job_zone")
        )
        zone_summary["job_zone_label"] = zone_summary["job_zone"].map(job_zone_label)
        zone_summary["job_zone_short"] = zone_summary["job_zone"].map(lambda zone: f"Zone {int(zone)}")
        fig = px.bar(
            zone_summary,
            x="job_zone_short",
            y="median_salary",
            text=zone_summary["median_salary"].map(compact_dollars),
            color="median_salary",
            color_continuous_scale=CONTINUOUS_SCALE,
            hover_data=["job_zone_label", "pathways"],
            labels={"job_zone_short": "O*NET Job Zone", "job_zone_label": "Meaning", "median_salary": "Median salary"},
            title="",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, zone_summary["median_salary"].max() * 1.16])
        st.plotly_chart(tune_fig(fig, 330), width="stretch")

    story_band(
        {
            "eyebrow": "Problem",
            "headline": "Education ROI is hard to see before enrollment.",
            "copy": "Students can usually find tuition, wages, and program details, but not as one connected pathway from credential to likely career outcome.",
        },
        {
            "eyebrow": "Result",
            "headline": "Preparation level explains more than cost alone.",
            "copy": "The analysis shows a clear salary relationship with O*NET Job Zone, while cost signals remain noisy and incomplete.",
        },
    )
    divider_label("How to read O*NET Job Zones")
    job_zone_legend(df["job_zone"].dropna().astype(int).unique().tolist())


def data_foundation_page(df: pd.DataFrame) -> None:
    add_page_header(
        "Data Foundation",
        "The dataset integrates wage outcomes, preparation levels, and education program cost signals through CIP-to-SOC mapping.",
        "02",
    )

    metric_grid(
        [
            {"label": "BLS Wages", "value": f"{df['soc_code'].nunique():,}", "caption": "SOC codes"},
            {"label": "O*NET Zones", "value": f"{df['job_zone'].nunique()}", "caption": "Preparation levels"},
            {"label": "Scorecard Programs", "value": f"{df['program_name'].nunique():,}", "caption": "Credential paths"},
            {"label": "Institutions", "value": f"{df['institution_name'].nunique():,}", "caption": "Education providers"},
        ]
    )

    flow_grid(
        [
            {"name": "Wage outcomes", "detail": "BLS OEWS provides occupation-level salary targets."},
            {"name": "Preparation levels", "detail": "O*NET Job Zones add training and education intensity."},
            {"name": "Program costs", "detail": "Scorecard contributes tuition, debt, credential, and institution data."},
            {"name": "Crosswalk join", "detail": "CIP-to-SOC mapping turns disconnected files into pathway records."},
            {"name": "Model layer", "detail": "Salary prediction and personas translate the table into decisions."},
        ]
    )

    story_band(
        {
            "eyebrow": "Engineering focus",
            "headline": "The crosswalk is the project’s backbone.",
            "copy": "Without CIP-to-SOC linkage, education programs and occupations remain separate facts rather than comparable pathways.",
        },
        {
            "eyebrow": "Data reality",
            "headline": "Suppressed debt required a transparent caveat.",
            "copy": "Median imputation preserved usable records, but the supplement clearly treats debt as limited evidence rather than a precise signal.",
        },
    )
    divider_label("Preparation scale used in the model")
    job_zone_legend(df["job_zone"].dropna().astype(int).unique().tolist())
    divider_label("Vocabulary used across the supplement")
    insight_grid(
        [
            {"title": "CIP", "body": "Education program classification: the degree, certificate, or field of study side of the pathway."},
            {"title": "SOC", "body": "Occupation classification: the labor-market role whose wage outcome is being estimated."},
            {"title": "Pathway", "body": "A joined education-to-career record combining program, institution, preparation level, and salary outcome."},
        ]
    )

    insight_grid(
        [
            {
                "title": "Data engineering challenge",
                "body": "The hard step was joining education programs to occupations through CIP-to-SOC mapping while preserving enough records for modeling.",
            },
            {
                "title": "Debt caveat",
                "body": "Debt values are heavily privacy-suppressed, so the supplement treats imputed debt as limited evidence rather than a precise individual signal.",
            },
        ]
    )
    divider_label("How the data becomes an ROI pathway")
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="#d8dee8", width=1),
                    label=[
                        "BLS OEWS wages",
                        "O*NET Job Zones",
                        "College Scorecard",
                        "CIP-to-SOC Crosswalk",
                        "Unified pathway table",
                        "Salary model + personas",
                    ],
                    color=["#2f6f9f", "#2f7d68", "#a97524", "#7c3aed", "#4b5563", "#b84747"],
                ),
                link=dict(
                    source=[0, 1, 2, 3, 4],
                    target=[4, 4, 4, 4, 5],
                    value=[1, 1, 1, 1, 4],
                    color=["rgba(47,111,159,.25)", "rgba(47,125,104,.25)", "rgba(169,117,36,.25)", "rgba(124,58,237,.22)", "rgba(184,71,71,.25)"],
                ),
            )
        ]
    )
    fig.update_layout(font_size=12)
    st.plotly_chart(tune_fig(fig, 360), width="stretch")

    divider_label("Data quality snapshot")
    quality = []
    for col in ["occupation_title", "soc_code", "job_zone", "program_name", "credential_level", "median_debt", "tuition_in_state", "median_salary"]:
        example_series = df[col].dropna().astype(str)
        example = example_series.head(1).iloc[0] if not example_series.empty else "N/A"
        if col == "job_zone":
            example = job_zone_label(float(example))
        quality.append(
            {
                "Column": "Job Zone (preparation level)" if col == "job_zone" else DISPLAY_COLUMN_NAMES.get(col, col.replace("_", " ").title()),
                "Completeness": 1 - df[col].isna().mean(),
                "Unique values": df[col].nunique(dropna=True),
                "Example": example,
            }
        )
    quality_df = pd.DataFrame(quality)
    quality_df["Completeness"] = quality_df["Completeness"].map(pct)
    st.dataframe(quality_df, width="stretch", hide_index=True)


def exploration_page(df: pd.DataFrame) -> None:
    add_page_header(
        "Exploration",
        "The exploratory view tests whether cost, credentials, and preparation levels move with expected salary outcomes.",
        "03",
    )

    sample_size = min(7000, len(df))
    sample = (df.sample(min(len(df), sample_size), random_state=22) if len(df) > sample_size else df).copy()
    sample["job_zone_short"] = sample["job_zone"].map(job_zone_short_label)
    corr_tuition = df[["tuition_in_state", "median_salary"]].corr(numeric_only=True).iloc[0, 1]
    corr_zone = df[["job_zone", "median_salary"]].corr(numeric_only=True).iloc[0, 1]
    insight_grid(
        [
            {"title": "Job Zone signal", "body": f"Correlation with salary is {corr_zone:.2f} in the visible data."},
            {"title": "Cost signal", "body": f"Tuition-to-salary correlation is {corr_tuition:.2f}, much noisier than preparation level."},
        ]
    )
    add_callout(
        "<strong>Correlation cue:</strong> values closer to 1.0 move together strongly, values near 0.0 are weak. In this dataset, preparation level is the clearer salary signal."
    )
    story_band(
        {
            "eyebrow": "Hypothesis",
            "headline": "More preparation should correspond with higher wages.",
            "copy": "Job Zones encode the level of education, experience, and training usually required for an occupation.",
        },
        {
            "eyebrow": "Evidence",
            "headline": "Cost is a weak shortcut for outcome quality.",
            "copy": "Tuition and debt help describe risk, but they do not reliably rank salary outcomes across pathways.",
        },
    )
    divider_label("Tuition vs. salary by preparation level")
    fig = px.scatter(
        sample,
        x="tuition_in_state",
        y="median_salary",
        color="job_zone_short",
        opacity=0.58,
        hover_data=["occupation_title", "program_name", "credential_level", "job_zone_label", "job_zone_plain", "tuition_in_state"],
        labels={
            "tuition_in_state": "In-state tuition",
            "job_zone_short": "Job Zone",
            "job_zone_label": "Job Zone",
            "job_zone_plain": "Meaning",
            "median_salary": "Median salary",
            "credential_level": "Credential",
        },
        title="",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(tune_fig(fig, 440), width="stretch")

    summary = (
        df.groupby("job_zone_label", as_index=False)
        .agg(
            pathways=("median_salary", "size"),
            median_salary=("median_salary", "median"),
            median_tuition=("tuition_in_state", "median"),
            mean_error=("absolute_error", "mean"),
        )
        .sort_values("median_salary", ascending=False)
    )
    summary["job_zone_short"] = summary["job_zone_label"].str.extract(r"^(Zone \d+)")[0]

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        divider_label("Median salary by Job Zone")
        fig = px.bar(
            summary,
            x="median_salary",
            y="job_zone_short",
            orientation="h",
            color="median_salary",
            color_continuous_scale=CONTINUOUS_SCALE,
            text=summary["median_salary"].map(compact_dollars),
            labels={"median_salary": "Median salary", "job_zone_short": ""},
            title="",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, summary["median_salary"].max() * 1.15])
        st.plotly_chart(tune_fig(fig, 340), width="stretch")
    with c2:
        divider_label("Visible group summary")
        table = summary[["job_zone_label", "pathways", "median_salary", "median_tuition", "mean_error"]].rename(
            columns={
                "pathways": "Pathways",
                "median_salary": "Median salary",
                "median_tuition": "Median tuition",
                "mean_error": "Mean model error",
                "job_zone_label": "Job Zone",
            }
        )
        for col in ["Median salary", "Median tuition", "Mean model error"]:
            table[col] = table[col].map(dollars)
        st.dataframe(table, width="stretch", height=260, hide_index=True)


def build_prediction(model, job_zone: int, debt: float, tuition: float, credential: str) -> float:
    row = pd.DataFrame(
        [
            {
                "job_zone": job_zone,
                "median_debt": float(debt),
                "tuition_in_state": float(tuition),
                "credential_level": credential,
            }
        ]
    )
    return float(model.predict(row)[0])


def program_profile_matches(
    df: pd.DataFrame,
    target_tuition: float,
    target_salary: float,
    target_zone: str,
    credential: str,
    limit: int = 8,
) -> pd.DataFrame:
    candidates = df.dropna(subset=["program_name", "credential_level", "tuition_in_state", "median_salary", "job_zone"]).copy()
    candidates = candidates[candidates["tuition_in_state"] > 0]
    if credential != "Any credential":
        candidates = candidates[candidates["credential_level"] == credential]
    if target_zone != "Any preparation level":
        zone_value = int(target_zone.split()[1])
        candidates = candidates[candidates["job_zone"].astype(int) == zone_value]
    if candidates.empty:
        return candidates

    profiles = (
        candidates.groupby(["program_name", "credential_level"], as_index=False)
        .agg(
            pathways=("median_salary", "size"),
            median_salary=("median_salary", "median"),
            median_tuition=("tuition_in_state", "median"),
            typical_zone=("job_zone", "median"),
        )
    )
    profiles = profiles[profiles["pathways"] >= 3].copy()
    if profiles.empty:
        return profiles

    feature_cols = ["median_tuition", "median_salary"]
    target_values = [float(target_tuition), float(target_salary)]
    matrix = profiles[feature_cols].astype(float)
    scale = matrix.std(ddof=0).replace(0, 1)
    target = pd.Series(target_values, index=feature_cols)
    distance = (((matrix - target) / scale) ** 2).mean(axis=1).pow(0.5)
    profiles["match_score"] = 1 / (1 + distance)
    return profiles.sort_values(["match_score", "median_salary", "pathways"], ascending=False).head(limit)


def model_lab_page(df: pd.DataFrame) -> None:
    model, _, _ = load_artifacts()
    add_page_header(
        "Model Lab",
        "A scenario sandbox for testing how the trained salary model responds to credential, cost, and preparation assumptions.",
        "04",
    )

    input_col, result_col = st.columns([0.72, 1.28])
    with input_col:
        divider_label("Scenario builder")
        zone_options = sorted(df["job_zone"].dropna().astype(int).unique().tolist())
        default_zone = 4 if 4 in zone_options else zone_options[0]
        job_zone = st.select_slider(
            "O*NET Job Zone",
            options=zone_options,
            value=default_zone,
            format_func=job_zone_label,
        )
        credentials = sorted(df["credential_level"].unique())
        default_credential = "Bachelor's Degree" if "Bachelor's Degree" in credentials else credentials[0]
        credential = st.pills(
            "Credential level",
            credentials,
            default=default_credential,
            width="stretch",
        )
        credential = credential or default_credential
        debt_default = int(df["median_debt"].median())
        tuition_default = int(df["tuition_in_state"].median()) if pd.notna(df["tuition_in_state"].median()) else 15000
        debt = st.slider("Median debt", 0, 150000, debt_default, step=2500, format="$%d")
        tuition = st.slider("In-state tuition", 0, 150000, tuition_default, step=2500, format="$%d")

        prediction = build_prediction(model, job_zone, debt, tuition, credential)

    nearby = df[
        (df["job_zone"].astype(int) == int(job_zone))
        & (df["credential_level"] == credential)
    ].copy()

    with result_col:
        divider_label("Scenario result")
        result_panel(
            "Predicted median salary",
            dollars(prediction),
            f"{credential} pathway at {job_zone_label(job_zone)}. {job_zone_plain(job_zone)}",
            [
                {
                    "label": "Comparable records",
                    "value": f"{len(nearby):,}",
                },
                {
                    "label": "Scenario cost",
                    "value": compact_dollars(debt + tuition),
                },
                {
                    "label": "Credential",
                    "value": credential,
                },
            ]
        )
        divider_label("Preparation ladder at this cost and credential")
        zone_prediction_ladder(model, zone_options, debt, tuition, credential, int(job_zone))
        st.caption("The highlighted card is the selected Job Zone; the other cards keep credential, debt, and tuition fixed.")

    divider_label("Cost sensitivity at the selected preparation level")
    tuition_values = np.linspace(0, 150000, 31)
    curve = pd.DataFrame(
        {
            "Tuition": tuition_values,
            "Predicted salary": [
                build_prediction(model, job_zone, debt, value, credential) for value in tuition_values
            ],
        }
    )
    fig = px.line(
        curve,
        x="Tuition",
        y="Predicted salary",
        markers=True,
        labels={"Tuition": "In-state tuition", "Predicted salary": "Predicted salary"},
        title="",
    )
    fig.update_traces(line=dict(color="#2f6f9f", width=4), marker=dict(size=6))
    fig.add_trace(
        go.Scatter(
            x=[tuition],
            y=[prediction],
            mode="markers+text",
            marker=dict(size=13, color="#b84747", line=dict(width=2, color="#ffffff")),
            text=["Current scenario"],
            textposition="top center",
            name="Current scenario",
        )
    )
    fig.update_yaxes(range=[curve["Predicted salary"].min() * 0.96, curve["Predicted salary"].max() * 1.08])
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    sensitivity_fig = tune_fig(fig, 390)
    sensitivity_fig.update_layout(margin=dict(l=70, r=28, t=28, b=54))
    st.plotly_chart(sensitivity_fig, width="stretch")
    st.caption("This isolates tuition while holding credential, debt, and selected Job Zone fixed.")

    divider_label("Observed anchors for this scenario")
    if nearby.empty:
        st.info("No pathways match the exact scenario under the current filters.")
    else:
        nearby["prediction_gap"] = (nearby["median_salary"] - prediction).abs()
        table = nearby.sort_values(["prediction_gap", "median_salary"], ascending=[True, False])[
            [
                "occupation_title",
                "program_name",
                "institution_name",
                "job_zone_label",
                "tuition_in_state",
                "median_salary",
                "predicted_salary",
                "persona",
            ]
        ].head(8)
        table = table.rename(columns={"job_zone_label": "Job Zone"})
        st.dataframe(format_table(table), width="stretch", height=300, hide_index=True)

    divider_label("How to interpret the lab")
    insight_grid(
        [
            {"title": "Directional estimate", "body": "Use the prediction to compare scenarios, not to guarantee an individual wage."},
            {"title": "Strongest lever", "body": "The preparation ladder shows Job Zone has a clearer model effect than raw cost."},
            {"title": "Debt caveat", "body": "Debt is included, but many values were imputed, so it should be read as incomplete evidence."},
            {"title": "Model layer", "body": "HistGradientBoosting captures non-linear credential, cost, and preparation patterns."},
        ]
    )


def performance_page(df: pd.DataFrame) -> None:
    summary = metrics_summary(df)
    add_page_header(
        "Model Performance",
        "The model captures meaningful salary structure while exposing where education-cost features are insufficient.",
        "05",
    )

    metric_grid(
        [
            {"label": "R2", "value": f"{summary['r2']:.3f}", "caption": "How much variation is explained"},
            {"label": "RMSE", "value": dollars(summary["rmse"]), "caption": "Penalizes large misses"},
            {"label": "MAE", "value": dollars(summary["mae"]), "caption": "Average dollar miss"},
            {"label": "Median Error", "value": dollars(df["absolute_error"].median()), "caption": "Typical dollar miss"},
        ]
    )
    add_callout(
        "<strong>Metric guide:</strong> R2 shows how much variation the model explains; MAE and Median Error are the easiest dollar-error checks; RMSE rises when a few misses are very large."
    )

    story_band(
        {
            "eyebrow": "Strong area",
            "headline": "The model finds broad preparation-level structure.",
            "copy": "The R2 and error charts show the model has real signal, especially where career requirements are more standardized.",
        },
        {
            "eyebrow": "Weak area",
            "headline": "Advanced roles carry larger unexplained variance.",
            "copy": "Job Zones 4 and 5 are more sensitive to missing drivers like experience, geography, employer, specialization, and negotiation.",
        },
    )
    left, right = st.columns([0.95, 1.05])
    with left:
        divider_label("Mean error by Job Zone")
        error = (
            df.groupby("job_zone", as_index=False)
            .agg(mean_error=("absolute_error", "mean"), median_error=("absolute_error", "median"), pathways=("absolute_error", "size"))
            .sort_values("job_zone")
        )
        error["job_zone_label"] = error["job_zone"].map(job_zone_label)
        error["job_zone_short"] = error["job_zone"].map(lambda zone: f"Zone {int(zone)}")
        fig = px.bar(
            error,
            x="job_zone_short",
            y="mean_error",
            color="mean_error",
            text=error["mean_error"].map(compact_dollars),
            color_continuous_scale=["#2f7d68", "#a97524", "#b84747"],
            hover_data=["job_zone_label", "pathways"],
            labels={"mean_error": "Mean absolute error", "job_zone_short": "O*NET preparation level", "job_zone_label": "Job Zone"},
            title="",
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, error["mean_error"].max() * 1.18])
        st.plotly_chart(tune_fig(fig, 450), width="stretch")
    with right:
        divider_label("Error spread by Job Zone")
        sample = (df.sample(min(len(df), 9000), random_state=44) if len(df) > 9000 else df).copy()
        sample["job_zone_short"] = sample["job_zone"].map(lambda zone: f"Zone {int(zone)}")
        fig = px.box(
            sample,
            x="job_zone_short",
            y="absolute_error",
            color="job_zone_short",
            points=False,
            color_discrete_sequence=PALETTE,
            hover_data=["job_zone_label"],
            labels={
                "absolute_error": "Absolute prediction error",
                "job_zone_short": "Job Zone",
                "job_zone_label": "Job Zone",
            },
            title="",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(tune_fig(fig, 450), width="stretch")

    divider_label("Review model misses")
    miss_type = st.radio("Show", ["Largest misses", "Under-predicted", "Over-predicted"], horizontal=True)
    records = df.copy()
    if miss_type == "Under-predicted":
        records = records[records["residual"] > 0].sort_values("residual", ascending=False)
    elif miss_type == "Over-predicted":
        records = records[records["residual"] < 0].sort_values("residual", ascending=True)
    else:
        records = records.sort_values("absolute_error", ascending=False)
    table = records[
        [
            "occupation_title",
            "program_name",
            "credential_level",
            "job_zone_label",
            "median_salary",
            "predicted_salary",
            "absolute_error",
            "persona",
        ]
    ].head(12)
    table = table.rename(columns={"job_zone_label": "Job Zone"})
    st.dataframe(format_table(table), width="stretch", hide_index=True)


def personas_page(df: pd.DataFrame) -> None:
    _, kmeans, scaler = load_artifacts()
    add_page_header(
        "Pathway Personas",
        "K-Means summarizes pathway patterns into interpretable groups for salary, debt, and preparation level.",
        "06",
    )

    selected_personas = st.pills(
        "Personas to display",
        sorted(df["persona"].unique()),
        selection_mode="multi",
        default=sorted(df["persona"].unique()),
        width="stretch",
    )
    selected_personas = selected_personas or sorted(df["persona"].unique())
    persona_df = df[df["persona"].isin(selected_personas)].copy()
    if persona_df.empty:
        st.warning("No persona records match the current selection.")
        return

    persona_summary = (
        persona_df.groupby("persona", as_index=False)
        .agg(
            pathways=("median_salary", "size"),
            median_salary=("median_salary", "median"),
            median_tuition=("tuition_in_state", "median"),
            avg_job_zone=("job_zone", "mean"),
            mean_error=("absolute_error", "mean"),
        )
        .sort_values("median_salary", ascending=False)
    )
    persona_summary["typical_zone"] = persona_summary["avg_job_zone"].round().astype(int).map(job_zone_label)
    persona_summary["persona_short"] = persona_summary["persona"].map(persona_short_name)

    add_callout(
        "<strong>How to read personas:</strong> K-Means groups pathways into decision segments using salary, debt, and preparation level. These are summaries for comparison, not best-to-worst rankings."
    )

    divider_label("Compare persona profiles")
    overview_col, table_col = st.columns([0.95, 1.05])
    with overview_col:
        fig = px.bar(
            persona_summary,
            x="median_salary",
            y="persona_short",
            orientation="h",
            text=persona_summary["median_salary"].map(compact_dollars),
            color="median_salary",
            color_continuous_scale=CONTINUOUS_SCALE,
            hover_data=["persona", "pathways", "median_tuition", "typical_zone"],
            labels={"median_salary": "Median salary", "persona_short": "", "persona": "Persona"},
            title="",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, persona_summary["median_salary"].max() * 1.24])
        salary_fig = tune_fig(fig, 330)
        salary_fig.update_layout(margin=dict(l=160, r=24, t=24, b=48))
        st.plotly_chart(salary_fig, width="stretch")
    with table_col:
        table = persona_summary[
            ["persona", "pathways", "median_salary", "median_tuition", "typical_zone", "mean_error"]
        ].rename(
            columns={
                "persona": "Persona",
                "pathways": "Pathways",
                "median_salary": "Median salary",
                "median_tuition": "Median tuition",
                "typical_zone": "Typical preparation level",
                "mean_error": "Mean model error",
            }
        )
        for col in ["Median salary", "Median tuition", "Mean model error"]:
            table[col] = table[col].map(dollars)
        st.dataframe(table, width="stretch", height=330, hide_index=True)

    divider_label("Persona map: debt, salary, and preparation")
    sample = persona_df.sample(min(len(persona_df), 10000), random_state=55) if len(persona_df) > 10000 else persona_df
    sample = sample.copy()
    sample["persona_short"] = sample["persona"].map(persona_short_name)
    sample["job_zone_short"] = sample["job_zone"].map(lambda zone: f"Zone {int(zone)}")
    fig = px.scatter(
        sample,
        x="median_debt",
        y="median_salary",
        color="persona_short",
        symbol="job_zone_short",
        opacity=0.6,
        hover_data={
            "occupation_title": True,
            "program_name": True,
            "credential_level": True,
            "job_zone_label": True,
            "job_zone_plain": True,
            "persona_short": False,
            "job_zone_short": False,
            "median_debt": ":$,.0f",
            "median_salary": ":$,.0f",
        },
        color_discrete_sequence=PALETTE,
        labels={
            "median_debt": "Median debt",
            "median_salary": "Median salary",
            "persona_short": "Persona",
            "job_zone_short": "Job Zone",
            "job_zone_label": "Job Zone",
            "job_zone_plain": "Meaning",
        },
        title="",
    )
    persona_map = tune_fig(fig, 450)
    persona_map.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=72, r=24, t=42, b=112),
    )
    st.plotly_chart(persona_map, width="stretch")

    divider_label("Inspect one persona")
    focus = st.pills(
        "Persona detail",
        persona_summary["persona"].tolist(),
        default=persona_summary["persona"].iloc[0],
        label_visibility="collapsed",
        width="stretch",
    )
    focus = focus or persona_summary["persona"].iloc[0]
    focus_df = persona_df[persona_df["persona"] == focus].copy()
    focus_record = persona_summary[persona_summary["persona"] == focus].iloc[0]

    detail_col, mix_col = st.columns([0.8, 1.2])
    with detail_col:
        result_panel(
            f"Selected persona: {persona_short_name(focus)}",
            compact_dollars(focus_record["median_salary"]),
            str(focus),
            [
                {"label": "Pathways", "value": f"{int(focus_record['pathways']):,}"},
                {"label": "Median tuition", "value": compact_dollars(focus_record["median_tuition"])},
                {"label": "Typical prep", "value": f"Zone {int(round(focus_record['avg_job_zone']))}"},
            ],
        )
        st.caption(job_zone_plain(round(focus_record["avg_job_zone"])))
    with mix_col:
        divider_label("Credential mix")
        cred_mix = focus_df["credential_level"].value_counts().rename_axis("credential_level").reset_index(name="pathways")
        fig = px.bar(
            cred_mix,
            x="pathways",
            y="credential_level",
            orientation="h",
            color="pathways",
            color_continuous_scale=CONTINUOUS_SCALE,
            text="pathways",
            labels={"pathways": "Pathways", "credential_level": ""},
            title="",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, cred_mix["pathways"].max() * 1.2])
        cred_fig = tune_fig(fig, 330)
        cred_fig.update_layout(margin=dict(l=200, r=28, t=24, b=48))
        st.plotly_chart(cred_fig, width="stretch")

    divider_label("Top occupations in selected persona")
    top_occ = (
        focus_df.groupby("occupation_title", as_index=False)
        .agg(pathways=("median_salary", "size"), median_salary=("median_salary", "median"), median_tuition=("tuition_in_state", "median"))
        .sort_values(["median_salary", "pathways"], ascending=False)
        .head(12)
    )
    top_occ["median_salary"] = top_occ["median_salary"].map(dollars)
    top_occ["median_tuition"] = top_occ["median_tuition"].map(dollars)
    st.dataframe(
        top_occ.rename(columns={"occupation_title": "Occupation", "pathways": "Pathways", "median_salary": "Median salary", "median_tuition": "Median tuition"}),
        width="stretch",
        height=330,
        hide_index=True,
    )


def decision_takeaways_page(df: pd.DataFrame) -> None:
    add_page_header(
        "Decision Takeaways",
        "The output converts the model and EDA into concrete decisions about education investment and pathway risk.",
        "07",
    )

    top_value = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["salary_to_tuition"]).copy()
    top_value = top_value[top_value["tuition_in_state"] > 0]

    insight_grid(
        [
            {"title": "Primary signal", "body": "Job Zone provides the clearest relationship to salary."},
            {"title": "ROI warning", "body": "Higher spending does not reliably produce proportionally higher salary."},
            {"title": "Preparation tradeoff", "body": "Zones 2 and 3 often mean faster entry; Zones 4 and 5 often mean longer preparation and wider salary variation."},
            {"title": "Best use", "body": "The supplement supports comparison, counseling, and policy analysis rather than one-size-fits-all ranking."},
        ]
    )
    story_band(
        {
            "eyebrow": "Bottom line",
            "headline": "Education investment needs pathway-level transparency.",
            "copy": "The data supports better decisions when cost, preparation level, credential, and salary are viewed together.",
        },
        {
            "eyebrow": "Surprise",
            "headline": "Higher spending is not automatically higher ROI.",
            "copy": "The weak cost-to-salary relationship is the most important practical warning from the analysis.",
        },
    )

    st.subheader("Decision impact")
    insight_grid(
        [
            {"title": "Students", "body": "Compare expected salary against credential, preparation level, and cost before taking on debt."},
            {"title": "Counselors", "body": "Explain tradeoffs with pathway-level evidence rather than general degree averages."},
            {"title": "Policy makers", "body": "Identify fields where higher education cost does not translate cleanly into earnings."},
            {"title": "Researchers", "body": "Use the error profile to prioritize next data: geography, experience, placement, and employers."},
        ]
    )
    divider_label("Pathway profile matcher")
    add_callout(
        "<strong>Recommendation lens:</strong> match program profiles to a target tuition, salary, credential, and preparation level using the same pathway evidence shown throughout the supplement."
    )
    tuition_col, salary_col, zone_col, credential_col = st.columns(4)
    with tuition_col:
        target_tuition = st.slider(
            "Target tuition",
            0,
            150000,
            int(min(max(df["tuition_in_state"].median(skipna=True), 0), 150000)),
            step=2500,
            format="$%d",
        )
    with salary_col:
        target_salary = st.slider(
            "Target salary",
            30000,
            250000,
            int(min(max(df["median_salary"].median(skipna=True), 30000), 250000)),
            step=5000,
            format="$%d",
        )
    with zone_col:
        zone_choices = ["Any preparation level"] + [f"Zone {int(zone)}" for zone in sorted(df["job_zone"].dropna().astype(int).unique())]
        target_zone = st.selectbox("Preparation level", zone_choices)
    with credential_col:
        credential_choices = ["Any credential"] + sorted(df["credential_level"].dropna().unique().tolist())
        target_credential = st.selectbox("Credential", credential_choices)

    matches = program_profile_matches(df, target_tuition, target_salary, target_zone, target_credential)
    if matches.empty:
        st.info("No program profiles match those constraints under the current filters.")
    else:
        table = matches[
            [
                "program_name",
                "credential_level",
                "typical_zone",
                "pathways",
                "median_tuition",
                "median_salary",
                "match_score",
            ]
        ].rename(
            columns={
                "program_name": "Program",
                "credential_level": "Credential",
                "typical_zone": "Prep",
                "pathways": "Pathways",
                "median_tuition": "Median tuition",
                "median_salary": "Median salary",
                "match_score": "Match",
            }
        )
        table["Prep"] = table["Prep"].round().astype(int).map(lambda zone: f"Zone {zone}")
        for col in ["Median tuition", "Median salary"]:
            table[col] = table[col].map(dollars)
        table["Match"] = table["Match"].map(lambda value: f"{value:.0%}")
        st.dataframe(table, width="stretch", height=300, hide_index=True)
        st.caption("Match scores are relative similarity scores for comparison, not acceptance odds or guaranteed returns.")

    left, right = st.columns([1.2, 0.8])
    with left:
        divider_label("High-value pathway examples")
        if top_value.empty:
            st.info("No positive tuition records are available under the current filters.")
        else:
            table = top_value.sort_values(["salary_to_tuition", "median_salary"], ascending=False)[
                [
                    "occupation_title",
                    "program_name",
                    "credential_level",
                    "job_zone_label",
                    "tuition_in_state",
                    "median_salary",
                    "salary_to_tuition",
                    "persona",
                ]
            ].head(10)
            table = table.rename(columns={"job_zone_label": "Job Zone"})
            st.dataframe(format_table(table), width="stretch", hide_index=True)
    with right:
        divider_label("Next data to add")
        next_data = pd.DataFrame(
            [
                {"Feature": "Geography", "Why it matters": "Local labor markets move salary and tuition."},
                {"Feature": "Experience level", "Why it matters": "Advanced-role pay changes sharply by seniority."},
                {"Feature": "Completion and placement", "Why it matters": "Tuition alone misses program outcomes."},
                {"Feature": "Employer and industry", "Why it matters": "The same credential can lead to very different wages."},
            ]
        )
        st.dataframe(next_data, width="stretch", hide_index=True)

    divider_label("Interpretation glossary")
    job_zone_legend(df["job_zone"].dropna().astype(int).unique().tolist())


def main() -> None:
    raw_df = load_data()
    df = enrich_data(raw_df)
    section, filtered_df, search_state = sidebar_filters(df)

    if filtered_df.empty:
        if search_state.get("active"):
            st.warning(
                f"No pathways matched \"{search_state['query']}\" within the current Job Zone and salary filters. Try fewer words or widen the sidebar filters."
            )
        else:
            st.warning("No records match the current filters. Widen the sidebar filters to continue.")
        return

    render_search_status(search_state)

    if section == "Overview":
        overview_page(filtered_df)
    elif section == "Data Foundation":
        data_foundation_page(filtered_df)
    elif section == "Exploration":
        exploration_page(filtered_df)
    elif section == "Model Lab":
        model_lab_page(filtered_df)
    elif section == "Performance":
        performance_page(filtered_df)
    elif section == "Personas":
        personas_page(filtered_df)
    else:
        decision_takeaways_page(filtered_df)


if __name__ == "__main__":
    main()
