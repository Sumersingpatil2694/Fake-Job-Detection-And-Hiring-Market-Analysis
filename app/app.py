import os
import re
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
# TfidfVectorizer is loaded at runtime from the pickled file (see load_models)

# ── Optional: DB / SQL integration ───────────────────────────────────────────
try:
    from db_connection import (
        MYSQL_CONFIG,
        fetch_country_fraud_analysis,
        fetch_employment_type_fraud_analysis,
        fetch_fraud_summary,
        fetch_high_risk_jobs,
        fetch_industry_risk,
        fetch_salary_fraud_analysis,
        run_query,
        test_connection,
    )
    DB_MODULE_AVAILABLE = True
except Exception:
    MYSQL_CONFIG = {}
    DB_MODULE_AVAILABLE = False

    def test_connection():
        return False

    def run_query(*args, **kwargs):
        return pd.DataFrame()

    def fetch_fraud_summary(*args, **kwargs):
        return pd.DataFrame()

    def fetch_high_risk_jobs(*args, **kwargs):
        return pd.DataFrame()

    def fetch_industry_risk(*args, **kwargs):
        return pd.DataFrame()

    def fetch_country_fraud_analysis(*args, **kwargs):
        return pd.DataFrame()

    def fetch_employment_type_fraud_analysis(*args, **kwargs):
        return pd.DataFrame()

    def fetch_salary_fraud_analysis(*args, **kwargs):
        return pd.DataFrame()

# PAGE CONFIG
st.set_page_config(
    page_title            = "🛡️ Fake Job Detector ML",
    page_icon             = "🛡️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# CUSTOM CSS
st.markdown("""
<style>
    .main                { background-color: #0d1117; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#161b22 0%,#0d1117 100%); }
    .kpi-card {
        background: linear-gradient(135deg,#1e2430,#232b3e);
        border-radius: 12px; padding: 20px 16px; text-align: center;
        border-left: 4px solid; margin: 6px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value { font-size: 2rem; font-weight: 800; }
    .kpi-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .fraud-card   { background:#2d1b1b; border-left:5px solid #E74C3C; border-radius:10px; padding:20px; }
    .legit-card   { background:#1b2d1b; border-left:5px solid #2ECC71; border-radius:10px; padding:20px; }
    .warning-card { background:#2d2416; border-left:5px solid #F39C12; border-radius:10px; padding:16px; }
    .info-card    { background:#1b2030; border-left:5px solid #3498DB; border-radius:10px; padding:16px; }
    .limit-card   { background:#1e1e2e; border-left:5px solid #9B59B6; border-radius:10px; padding:16px; margin:8px 0; }
    .fix-card     { background:#1a2d1a; border-left:5px solid #27AE60; border-radius:10px; padding:14px; margin:6px 0; }
    .info-card  { background:linear-gradient(135deg,#1a1a2e,#16213e); border-left:5px solid #4285f4;
                    border-radius:12px; padding:20px; margin:10px 0;
                    box-shadow:0 4px 20px rgba(66,133,244,0.15); }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #e6edf3;
        border-bottom: 2px solid #30363d; padding-bottom: 8px; margin: 20px 0 14px 0;
    }
    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:2px; }
    .badge-red    { background:#3d1a1a; color:#E74C3C; border:1px solid #E74C3C; }
    .badge-green  { background:#1a3d1a; color:#2ECC71; border:1px solid #2ECC71; }
    .badge-yellow { background:#3d2a10; color:#F39C12; border:1px solid #F39C12; }
    .badge-blue   { background:#1a2a3d; color:#3498DB; border:1px solid #3498DB; }
    .badge-purple { background:#2a1a3d; color:#9B59B6; border:1px solid #9B59B6; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
FRAUD_THRESHOLD = 0.35

# IMPORTANT: This list MUST exactly match the notebook (Cell 8) so feature
# engineering at inference time matches training time. 18 words — do NOT add
# or remove without retraining the model.
URGENCY_WORDS = [
    'urgent', 'immediate', 'asap', 'hurry', 'limited',
    'act now', 'no experience', 'work from home',
    'earn money', 'easy money', 'guaranteed',
    'weekly pay', 'quick money', 'high pay',
    'bonus', '100%', 'free training', 'daily pay'
]

# Each entry is a (check_fn, positive_label) tuple.
# check_fn returns True when the flag is a red flag.
# positive_label is shown when the check passes (no red flag).
RED_FLAG_CHECKS = {
    "No salary disclosed"           : (lambda r: r.get("has_salary", 1) == 0,            "Salary disclosed"),
    "No company profile provided"   : (lambda r: r.get("has_company_profile", 1) == 0,   "Company profile available"),
    "No requirements listed"        : (lambda r: r.get("has_requirements", 1) == 0,      "Requirements listed"),
    "Contains urgency language"     : (lambda r: r.get("has_urgency_words", 0) == 1,     "No urgency language"),
    "Very short description (<300)" : (lambda r: r.get("desc_length", 999) < 300,        "Description length OK (≥300 chars)"),
    "No company logo"               : (lambda r: r.get("has_company_logo", 1) == 0,      "Company logo present"),
}


# Cache for 5 min; consistent TTL with load_data() prevents stale empty results
@st.cache_data(show_spinner=False, ttl=300)
def load_sql_views():
    if not DB_MODULE_AVAILABLE or not test_connection():
        return {
            "connected": False,
            "summary": pd.DataFrame(),
            "industry": pd.DataFrame(),
            "high_risk": pd.DataFrame(),
            "country": pd.DataFrame(),
            "employment": pd.DataFrame(),
            "salary": pd.DataFrame(),
        }

    return {
        "connected": True,
        "summary": fetch_fraud_summary(),
        "industry": fetch_industry_risk(15),
        "high_risk": fetch_high_risk_jobs(20),
        "country": fetch_country_fraud_analysis(15),
        "employment": fetch_employment_type_fraud_analysis(12),
        "salary": fetch_salary_fraud_analysis(),
    }


def sql_scalar(df: pd.DataFrame, column: str, default=0):
    if df is None or df.empty or column not in df.columns:
        return default
    val = df.iloc[0][column]
    if pd.isna(val):
        return default
    return val


def extract_model_feature_insights(model, tfidf, top_n: int = 12):
    if model is None or tfidf is None:
        return pd.DataFrame(), pd.DataFrame()

    try:
        feature_names = np.array(tfidf.get_feature_names_out())
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_[0][: len(feature_names)], dtype=float)
        pos_idx = np.argsort(weights)[-top_n:][::-1]
        neg_idx = np.argsort(weights)[:top_n]
        pos_df = pd.DataFrame({
            "keyword": feature_names[pos_idx],
            "weight": weights[pos_idx],
            "direction": "Higher fraud risk",
        })
        neg_df = pd.DataFrame({
            "keyword": feature_names[neg_idx],
            "weight": np.abs(weights[neg_idx]),
            "direction": "Higher legitimacy signal",
        })
        return pos_df, neg_df

    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_[: len(feature_names)], dtype=float)
        top_idx = np.argsort(weights)[-top_n:][::-1]
        imp_df = pd.DataFrame({
            "keyword": feature_names[top_idx],
            "weight": weights[top_idx],
            "direction": "Model importance",
        })
        return imp_df, pd.DataFrame()

    return pd.DataFrame(), pd.DataFrame()


# HELPER FUNCTIONS
def clean_text(text: str) -> str:
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_urgency(text: str) -> int:
    t = str(text).lower()
    return int(any(w in t for w in URGENCY_WORDS))

def make_kpi_card(value, label, color="#3498DB"):
    return f"""<div class="kpi-card" style="border-color:{color}">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div></div>"""

def risk_color(prob: float) -> str:
    if prob >= FRAUD_THRESHOLD: return "#E74C3C"
    if prob >= 0.20:            return "#F39C12"
    return "#2ECC71"

def risk_label(prob: float) -> str:
    if prob >= FRAUD_THRESHOLD: return "🚨 HIGH RISK – LIKELY FRAUD"
    if prob >= 0.20:            return "⚠️  MEDIUM RISK"
    return "✅ LOW RISK – LIKELY LEGITIMATE"

# DATA & MODEL LOADERS
# SQL-first with CSV fallback; 5 min cache matches load_sql_views
@st.cache_data(show_spinner=False, ttl=300)
def load_data():
    if DB_MODULE_AVAILABLE and test_connection():
        # Load full table without a row limit
        df = run_query("SELECT * FROM job_postings")
        if df is not None and not df.empty:
            # Remove duplicate rows in case CSV was uploaded more than once
            if "job_id" in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=["job_id"])
                removed = before - len(df)
                if removed > 0:
                    print(f"[load_data] ⚠ Removed {removed:,} duplicate rows from MySQL.")
            return df, "MySQL"

    paths = [
        "outputs/cleaned_job_postings.csv",
        "cleaned_job_postings.csv",
        "Fake_Job_Postings.csv",
        "Fake Job Postings.csv",
    ]

    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, low_memory=False), p

    return None, "Unavailable"


def prepare_dataset(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    df = df.copy()

    if 'has_salary' not in df.columns:
        df['has_salary'] = df['salary_range'].notna().astype(int) if 'salary_range' in df.columns else 0
    if 'has_company_profile' not in df.columns:
        df['has_company_profile'] = df['company_profile'].notna().astype(int) if 'company_profile' in df.columns else 0
    if 'has_requirements' not in df.columns:
        df['has_requirements'] = df['requirements'].notna().astype(int) if 'requirements' in df.columns else 0
    if 'has_benefits' not in df.columns:
        df['has_benefits'] = df['benefits'].notna().astype(int) if 'benefits' in df.columns else 0
    if 'has_company_logo' not in df.columns:
        df['has_company_logo'] = 0
    if 'telecommuting' not in df.columns:
        df['telecommuting'] = 0
    if 'has_questions' not in df.columns:
        df['has_questions'] = 0

    if 'has_urgency_words' not in df.columns:
        df['has_urgency_words'] = (
            df.get('title', '').fillna('') + ' ' + df.get('description', '').fillna('')
        ).apply(has_urgency)

    if 'desc_length' not in df.columns:
        df['desc_length'] = df.get('description', pd.Series(dtype=str)).fillna('').astype(str).str.len()
    if 'req_length' not in df.columns:
        df['req_length'] = df.get('requirements', pd.Series(dtype=str)).fillna('').astype(str).str.len()
    if 'profile_completeness' not in df.columns:
        # Score 0-6, must match the training data formula exactly
        df['profile_completeness'] = (
            df['has_salary'].fillna(0).astype(int)
            + df['has_company_profile'].fillna(0).astype(int)
            + df['has_requirements'].fillna(0).astype(int)
            + df['has_benefits'].fillna(0).astype(int)
            + df['has_company_logo'].fillna(0).astype(int)
            + df['has_questions'].fillna(0).astype(int)
        )

    # Always re-derive country from location (dataset's country col may contain city names)
    if True:
        US_STATES = {
            'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
            'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV',
            'NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
            'TX','UT','VT','VA','WA','WV','WI','WY','DC','PR','GU',
        }
        COUNTRY_NAMES_INLINE = {
            'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
            'AU': 'Australia',     'IN': 'India',           'DE': 'Germany',
            'FR': 'France',        'NL': 'Netherlands',     'SG': 'Singapore',
            'NZ': 'New Zealand',   'IE': 'Ireland',         'PK': 'Pakistan',
            'PH': 'Philippines',   'MY': 'Malaysia',        'BR': 'Brazil',
            'ZA': 'South Africa',  'GR': 'Greece',          'IT': 'Italy',
            'ES': 'Spain',         'PL': 'Poland',          'AE': 'UAE',
            'QA': 'Qatar',         'NG': 'Nigeria',         'KE': 'Kenya',
            'EG': 'Egypt',         'TR': 'Turkey',          'IL': 'Israel',
            'RO': 'Romania',       'HU': 'Hungary',         'CZ': 'Czech Republic',
            'SE': 'Sweden',        'NO': 'Norway',          'DK': 'Denmark',
            'FI': 'Finland',       'CH': 'Switzerland',     'AT': 'Austria',
            'BE': 'Belgium',       'PT': 'Portugal',        'MX': 'Mexico',
            'AR': 'Argentina',     'CO': 'Colombia',        'CL': 'Chile',
            'JP': 'Japan',         'CN': 'China',           'KR': 'South Korea',
            'HK': 'Hong Kong',     'TW': 'Taiwan',          'TH': 'Thailand',
            'ID': 'Indonesia',     'VN': 'Vietnam',         'BD': 'Bangladesh',
            'LK': 'Sri Lanka',     'UA': 'Ukraine',         'RU': 'Russia',
        }
        def _extract_country(loc):
            if pd.isna(loc) or str(loc).strip() in ('', 'nan'):
                return 'Unknown'
            parts = [p.strip() for p in str(loc).split(',')]

            # Dataset format: "COUNTRY_CODE, STATE, CITY"  → first part is country
            first = parts[0].strip().upper()
            if first in COUNTRY_NAMES_INLINE:
                return COUNTRY_NAMES_INLINE[first]

            # Fallback: check last part (reversed format)
            last = parts[-1].strip().upper()
            if last in COUNTRY_NAMES_INLINE:
                return COUNTRY_NAMES_INLINE[last]

            # US state code in first position → United States
            if first in US_STATES:
                return 'United States'
            # US state code in last position → United States
            if last in US_STATES:
                return 'United States'

            # Unknown 2-letter code → keep raw
            if len(first) == 2 and first.isalpha():
                return first

            return 'Unknown'
        df['country'] = df['location'].apply(_extract_country) if 'location' in df.columns else 'Unknown'
    if 'department' not in df.columns:
        df['department'] = ''
    if 'has_department' not in df.columns:
        df['has_department'] = df['department'].fillna('').astype(str).str.strip().ne('').astype(int)
    if 'title_length' not in df.columns:
        df['title_length'] = df.get('title', pd.Series(dtype=str)).fillna('').astype(str).str.len()

    return df
    

@st.cache_resource(show_spinner=False)
def load_models():
    def _find(fn, alt=None):
        candidates = [fn, os.path.join("models", fn)]
        if alt:
            candidates += [alt, os.path.join("models", alt)]
        for p in candidates:
            if p and os.path.exists(p):
                return p
        raise FileNotFoundError(fn)
    try:
        with open(_find("best_model.pkl"),      "rb") as f: model = pickle.load(f)
        with open(_find("tfidf_vectorizer.pkl"), "rb") as f: tfidf = pickle.load(f)
        with open(_find("numeric_cols.pkl"),     "rb") as f: nc    = pickle.load(f)
        with open(_find("model_info.json","model_info.json.txt"), "r") as f:
            info = json.load(f)
        return model, tfidf, nc, info
    except FileNotFoundError:
        return None, None, None, None


# PREDICT FUNCTION  
def predict_job(title, company, desc, reqs, has_sal, has_logo, model, tfidf, num_cols):
    """

    - profile_completeness correctly computed as sum of available binary flags
    - Numeric features aligned to notebook training order via numeric_cols.pkl
    - Supports both LR (coef_) and RF (feature_importances_)
    """
    raw = f"{title} {company} {desc} {reqs}"
    cleaned = clean_text(raw)

    has_company = int(bool(str(company).strip()))
    has_reqs = int(bool(str(reqs).strip()))
    urgency_flag = has_urgency(raw)
    desc_len = len(str(desc))
    req_len = len(str(reqs))
    title_len = len(str(title))

    # Score 0-6, matching the training data formula; form fields not collected default to 0
    has_benefits_val  = 0
    has_questions_val = 0
    profile_completeness_score = (
        int(has_sal) + has_company + has_reqs
        + has_benefits_val + int(has_logo) + has_questions_val
    )  # max = 6, consistent with notebook

    feature_values = {
        'has_salary': int(has_sal),
        'has_company_profile': has_company,
        'has_requirements': has_reqs,
        'has_benefits': 0,
        'has_company_logo': int(has_logo),
        'has_questions': 0,
        'telecommuting': 0,
        'has_urgency_words': urgency_flag,
        'profile_completeness': profile_completeness_score,
        'desc_length': desc_len,
        'req_length': req_len,
        'title_length': title_len,
        'has_department': 0,
    }

    # 10 numeric features used during training; order must match model input shape
    EXPECTED_NUMERIC_COLS = [
        'has_salary',
        'has_company_profile',
        'has_requirements',
        'has_benefits',
        'has_company_logo',
        'has_questions',
        'telecommuting',
        'has_urgency_words',
        'profile_completeness',
        'desc_length',
    ]

    if num_cols is not None and len(list(num_cols)) == len(EXPECTED_NUMERIC_COLS):
        ordered_numeric_cols = list(num_cols)
    else:
        # Fall back to default list if saved pkl has drifted from training
        ordered_numeric_cols = EXPECTED_NUMERIC_COLS

    # Build feature vector with exactly the columns the model was trained on
    num_feats = np.array([[
        feature_values.get(col, 0) for col in ordered_numeric_cols
    ]], dtype=float)

    # Validate feature count before calling predict_proba
    EXPECTED = len(ordered_numeric_cols)
    if num_feats.shape[1] != EXPECTED:
        raise ValueError(
            f"Numeric feature mismatch: app built {num_feats.shape[1]} features, "
            f"model expects {EXPECTED}. "
            "Solution: Re-run the notebook (Kernel > Restart & Run All) and save artifacts."
        )

    text_vec = tfidf.transform([cleaned])
    full_feat = hstack([text_vec, csr_matrix(num_feats)])
    prob = model.predict_proba(full_feat)[0][1]

    flags = {
        'has_salary': int(has_sal),
        'has_company_profile': has_company,
        'has_requirements': has_reqs,
        'has_urgency_words': urgency_flag,
        'desc_length': desc_len,
        'has_company_logo': int(has_logo),
    }
    # Apply each check function; ignore the positive_label half of the tuple here
    red_flags = {k: check_fn(flags) for k, (check_fn, _pos_label) in RED_FLAG_CHECKS.items()}

    top_words = []
    try:
        feat_names = tfidf.get_feature_names_out()
        n_text = len(feat_names)
        tv = text_vec.toarray()[0]
        if hasattr(model, "coef_"):
            coefs = model.coef_[0][:n_text]
            contribs = coefs * tv
            top_idx = np.argsort(contribs)[-8:][::-1]
            top_words = [
                (feat_names[i], round(float(contribs[i]), 4))
                for i in top_idx if contribs[i] > 0
            ]
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_[:n_text]
            contribs = importances * tv
            top_idx = np.argsort(contribs)[-8:][::-1]
            top_words = [
                (feat_names[i], round(float(contribs[i]) * 1000, 4))
                for i in top_idx if contribs[i] > 0
            ]
    except Exception:
        top_words = []

    return float(prob), red_flags, top_words


# LOAD DATA & MODELS
raw_df, DATA_SOURCE = load_data()
df = prepare_dataset(raw_df)
model, tfidf, num_cols, model_info = load_models()
sql_views = load_sql_views()
data_ok = df is not None and not df.empty
model_ok = model is not None
DB_AVAILABLE = sql_views.get("connected", False)

_mi = model_info or {}
if data_ok:
    _mi.setdefault("total_records", int(len(df)))
    _mi.setdefault("fraud_records", int(df['fraudulent'].sum()) if 'fraudulent' in df.columns else 0)
    if 'fraudulent' in df.columns:
        _mi.setdefault("fraud_rate_pct", round(float(df['fraudulent'].mean() * 100), 2))

MODEL_NAME = _mi.get("best_model", "Logistic Regression" if model_ok else "Model artifacts not found")
MODEL_TYPE = type(model).__name__ if model is not None else "Unknown"


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px'>
        <div style='font-size:3rem'>🛡️</div>
        <div style='font-size:1.2rem; font-weight:700; color:#e6edf3'>Fake Job Detector</div>
        <div style='font-size:0.75rem; color:#2ECC71; margin-top:2px'>Real SQL + ML Workflow</div>
    </div>
    <hr style='border-color:#30363d; margin:10px 0'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Home",
        "📊  EDA Dashboard",
        "🔍  Job Checker",
        "🤖  Model Insights",
        "⚠️  Limitations & Bias",
    ], label_visibility="collapsed")

    # Clear cache so fresh data is loaded after notebook re-run
    if st.button("🔄 Reload Data", help="Clear cache and reload from MySQL / CSV"):
        load_data.clear()
        load_sql_views.clear()
        st.rerun()

    db_status = "✅ Connected" if DB_AVAILABLE else "⚠️ CSV fallback"
    data_source_label = "MySQL table" if DATA_SOURCE == "MySQL" else DATA_SOURCE

    st.markdown(f"""
    <hr style='border-color:#30363d'>
    <div style='font-size:0.78rem; color:#8b949e; padding:8px'>
        <b style='color:#e6edf3'>Data Source</b><br>
        🗄️ {db_status}<br>
        📁 {data_source_label}<br>
        📊 {_mi.get('total_records', 0):,} postings<br>
        🚨 {_mi.get('fraud_records', 0):,} fraud ({_mi.get('fraud_rate_pct', 0.0):.2f}%)<br><br>
        <b style='color:#e6edf3'>Model</b><br>
        🏆 {MODEL_NAME}<br>
        📈 AUC: {f"{_mi.get('auc_roc'):.4f}" if isinstance(_mi.get('auc_roc'), (int, float)) else 'N/A'}<br>
        🎯 Threshold: {_mi.get('threshold', FRAUD_THRESHOLD)}<br><br>
        <b style='color:#e6edf3'>App Focus</b><br>
        🚩 Fraud indicators table<br>
        🗄️ Real SQL KPI cards<br>
        🔑 TF-IDF keyword insights
    </div>""", unsafe_allow_html=True)


#  PAGE 1 – HOME
if page == "🏠  Home":
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px'>
        <h1 style='font-size:2.5rem; font-weight:800; color:#e6edf3'>
            🛡️ Fake Job Detection & Hiring Market Analysis
        </h1>
        <p style='color:#8b949e; font-size:1rem'>
            End-to-End SQL Analytics + Machine Learning Pipeline
        </p>
    </div>""", unsafe_allow_html=True)

    summary_df = sql_views.get("summary", pd.DataFrame())
    total_jobs = int(sql_scalar(summary_df, "total_jobs", _mi.get("total_records", len(df) if data_ok else 0)))
    total_fraud = int(sql_scalar(summary_df, "total_fraud", _mi.get("fraud_records", int(df['fraudulent'].sum()) if data_ok and 'fraudulent' in df.columns else 0)))
    fraud_rate = float(sql_scalar(summary_df, "fraud_rate_pct", _mi.get("fraud_rate_pct", 0.0)))
    avg_fake_desc = sql_scalar(summary_df, "avg_fake_desc_len", np.nan)
    avg_real_desc = sql_scalar(summary_df, "avg_real_desc_len", np.nan)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{total_jobs:,}", "Total Postings", "#3498DB", c1),
        (f"{total_fraud:,}", "Fraudulent Jobs", "#E74C3C", c2),
        (f"{fraud_rate:.2f}%", "Fraud Rate", "#F39C12", c3),
        (f"{_mi.get('auc_roc', 0):.4f}" if _mi.get('auc_roc') is not None else "N/A", "AUC-ROC", "#2ECC71", c4),
        (f"{_mi.get('threshold', FRAUD_THRESHOLD)}", "Deployment Threshold", "#9B59B6", c5),
    ]
    for val, lbl, clr, col in kpis:
        with col:
            st.markdown(make_kpi_card(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1.3, 0.7])

    with col_a:
        st.markdown('<div class="section-header">🏗️ Project Architecture</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Notebook Workflow", "Clean CSV, engineer features, and train the fraud model", "#E74C3C"),
            ("2", "MySQL Storage", "Load cleaned data into fake_job_detection.job_postings", "#E67E22"),
            ("3", "SQL Analytics", "Use vw_fraud_summary, vw_industry_risk, and vw_high_risk_jobs", "#F1C40F"),
            ("4", "ML Inference", f"TF-IDF + numeric features → deployed model: {MODEL_NAME}", "#2ECC71"),
            ("5", "Power BI / Streamlit", "Reuse the same SQL views and model outputs for dashboards", "#1ABC9C"),
        ]
        for num, name, desc_txt, clr in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:flex-start; margin:7px 0;'>
                <div style='background:{clr}; color:white; border-radius:50%;
                    width:26px; height:26px; display:flex; align-items:center;
                    justify-content:center; font-weight:700; flex-shrink:0; font-size:0.82rem'>{num}</div>
                <div style='margin-left:12px;'>
                    <div style='color:#e6edf3; font-weight:600; font-size:0.92rem'>{name}</div>
                    <div style='color:#8b949e; font-size:0.8rem'>{desc_txt}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        if DB_AVAILABLE and not sql_views.get("industry", pd.DataFrame()).empty:
            st.markdown('<div class="section-header">🗄️ SQL Business Insights</div>', unsafe_allow_html=True)
            top_industry = sql_views["industry"].head(8)[["industry", "fraud_rate_pct", "total_jobs"]]
            st.dataframe(top_industry, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-header">📊 Model & Data Status</div>', unsafe_allow_html=True)
        metrics = [
            ("Best Model", MODEL_NAME, "#3498DB"),
            ("Threshold", f"{_mi.get('threshold', FRAUD_THRESHOLD)}", "#9B59B6"),
            ("Precision", f"{_mi.get('precision', 0) * 100:.1f}%" if _mi.get('precision') is not None else "N/A", "#E74C3C"),
            ("Recall", f"{_mi.get('recall', 0) * 100:.1f}%" if _mi.get('recall') is not None else "N/A", "#2ECC71"),
            ("F1-Score", f"{_mi.get('f1_score', 0) * 100:.1f}%" if _mi.get('f1_score') is not None else "N/A", "#F39C12"),
            ("SQL Views", "Ready" if DB_AVAILABLE else "Fallback Mode", "#1ABC9C"),
            ("Avg Fake Desc", f"{avg_fake_desc:.0f}" if pd.notna(avg_fake_desc) else "N/A", "#E67E22"),
            ("Avg Real Desc", f"{avg_real_desc:.0f}" if pd.notna(avg_real_desc) else "N/A", "#2ECC71"),
        ]
        for metric, value, clr in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:8px 12px;
                        background:#161b22; border-radius:8px; margin:4px 0;'>
                <span style='color:#8b949e; font-size:0.85rem'>{metric}</span>
                <span style='color:{clr}; font-weight:700; font-size:0.9rem'>{value}</span>
            </div>""", unsafe_allow_html=True)


#  PAGE 2 – EDA DASHBOARD
elif page == "📊  EDA Dashboard":

    # ── Page CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .eda-header {
        background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0f2027 100%);
        border-radius: 16px;
        padding: 28px 32px 22px;
        margin-bottom: 28px;
        border: 1px solid #1e3a4a;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .eda-header h2 { color: #e6edf3; font-size: 1.7rem; font-weight: 800; margin: 0 0 6px; }
    .eda-header p  { color: #8b949e; font-size: 0.9rem; margin: 0; }

    /* ── Premium KPI Cards (Dashboard Hero Row) ───────────────────────── */
    .kpi-card {
        position: relative;
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 18px 20px 16px;
        text-align: left;
        transition: all 0.25s cubic-bezier(.4,.0,.2,1);
        overflow: hidden;
        min-height: 118px;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent, #388bfd);
        opacity: 0.85;
    }
    .kpi-card::after {
        content: "";
        position: absolute;
        top: -40px; right: -40px;
        width: 110px; height: 110px;
        background: radial-gradient(circle, var(--accent, #388bfd) 0%, transparent 70%);
        opacity: 0.10;
        pointer-events: none;
        transition: opacity 0.3s;
    }
    .kpi-card:hover {
        border-color: var(--accent, #388bfd);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.45), 0 0 0 1px var(--accent, #388bfd);
    }
    .kpi-card:hover::after { opacity: 0.22; }

    .kpi-head {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .kpi-icon {
        font-size: 1.05rem;
        width: 28px; height: 28px;
        display: inline-flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.04);
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #8b949e;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 2px 0 4px;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #6e7681;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .insight-box {
        background: #161b22;
        border-left: 3px solid #388bfd;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin: 8px 0 18px;
        font-size: 0.85rem;
        color: #8b949e;
    }
    .insight-box b { color: #e6edf3; }

    .chart-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #c9d1d9;
        margin-bottom: 4px;
        padding-left: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.warning("⚠️ Dataset not loaded. Please place `Fake Job Postings.csv` in the project directory.")
        st.stop()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="eda-header">
        <h2>📊 Exploratory Data Analysis Dashboard</h2>
        <p>17,880 job postings analysed · Kaggle Fake Job Prediction Dataset · MySQL + Python pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS (Premium Hero Row) ──────────────────────────────────────────
    total      = len(df)
    fraud_n    = int(df['fraudulent'].sum())
    legit_n    = total - fraud_n
    fraud_pct  = round(fraud_n / total * 100, 2)
    avg_desc   = int(df.loc[df['fraudulent']==1, 'desc_length'].mean()) if 'desc_length' in df.columns else 0

    # Prefer city-level fraud hotspot; fall back to country if city not available
    if 'city' in df.columns and df.loc[df['fraudulent']==1, 'city'].notna().any():
        top_fraud_loc = (
            df[df['fraudulent']==1]
            .dropna(subset=['city'])
            .groupby('city').size().idxmax()
        )
        top_fraud_label = "Top Fraud City"
        top_fraud_sub   = "Highest fake job volume"
    elif 'country' in df.columns:
        top_fraud_loc = df[df['fraudulent']==1].groupby('country').size().idxmax()
        top_fraud_label = "Top Fraud Country"
        top_fraud_sub   = "Highest fake job volume"
    else:
        top_fraud_loc = "N/A"
        top_fraud_label = "Top Fraud Region"
        top_fraud_sub   = "No location data"

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        (k1, "📋", f"{total:,}",          "Total Job Postings", "Across all industries",   "#58a6ff"),
        (k2, "✅", f"{legit_n:,}",        "Legitimate Jobs",    "Real verified postings",  "#2ECC71"),
        (k3, "🛡️", f"{fraud_n:,}",        "Fraudulent Jobs",    "Fake / scam postings",    "#E74C3C"),
        (k4, "📈", f"{fraud_pct}%",       "Overall Fraud Rate", "Industry average ~4.84%", "#F39C12"),
        (k5, "🏙️", str(top_fraud_loc),    top_fraud_label,      top_fraud_sub,             "#a371f7"),
    ]
    for col, icon, val, label, sub, color in kpi_data:
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{color}">
            <div class="kpi-head">
                <span class="kpi-icon" style="color:{color}">{icon}</span>
                <span class="kpi-label">{label}</span>
            </div>
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🌍 Geographic", "📝 Text Analysis", "🏭 Industry"])

    DARK_BG   = "#0d1117"
    FONT_CLR  = "#c9d1d9"
    GRID_CLR  = "#21262d"
    FRAUD_CLR = "#E74C3C"
    LEGIT_CLR = "#2ECC71"

    def dark_layout(fig, title="", height=380, xangle=0):
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color=FONT_CLR)) if title else None,
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font=dict(color=FONT_CLR, size=12),
            height=height,
            margin=dict(l=16, r=16, t=40 if title else 20, b=16),
            xaxis=dict(gridcolor=GRID_CLR, tickangle=xangle, linecolor=GRID_CLR),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR, borderwidth=1),
        )
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns([1, 1.6])

        # Donut chart – improved with center annotation
        with col1:
            counts  = df['fraudulent'].value_counts().reindex([0, 1], fill_value=0)
            pie_df  = pd.DataFrame({
                'Type': ['Legitimate', 'Fraudulent'],
                'Count': [int(counts.loc[0]), int(counts.loc[1])],
            })
            fig_pie = px.pie(
                pie_df, values='Count', names='Type',
                color='Type',
                color_discrete_map={"Legitimate": LEGIT_CLR, "Fraudulent": FRAUD_CLR},
                hole=0.55,
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                textfont_size=13,
                marker=dict(line=dict(color=DARK_BG, width=3)),
                pull=[0, 0.06],
            )
            fig_pie.add_annotation(
                text=f"<b>{fraud_pct}%</b><br>Fraud", x=0.5, y=0.5,
                font=dict(size=16, color=FRAUD_CLR), showarrow=False
            )
            dark_layout(fig_pie, "Job Posting Distribution", height=340)
            fig_pie.update_layout(showlegend=True, legend=dict(orientation='h', y=-0.08))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("""<div class="insight-box">
                <b>Key insight:</b> Only <b>4.84%</b> of postings are fraudulent —
                a highly imbalanced dataset. SMOTE oversampling was applied during model training.
            </div>""", unsafe_allow_html=True)

        # Employment Type – horizontal bar
        with col2:
            if 'employment_type' in df.columns:
                emp = df.groupby('employment_type')['fraudulent'].agg(['mean','count']).reset_index()
                emp.columns = ['Employment Type', 'Fraud Rate', 'Count']
                emp = emp[emp['Count'] > 50].sort_values('Fraud Rate')
                emp['Fraud Rate %'] = (emp['Fraud Rate'] * 100).round(2)
                emp['Label'] = emp['Employment Type'].str.replace('_', ' ').str.title()

                fig_emp = px.bar(
                    emp, y='Label', x='Fraud Rate %',
                    orientation='h',
                    color='Fraud Rate %',
                    color_continuous_scale='RdYlGn_r',
                    text='Fraud Rate %',
                )
                fig_emp.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    marker_line_width=0,
                )
                fig_emp.update_coloraxes(showscale=False)
                dark_layout(fig_emp, "Fraud Rate by Employment Type", height=340)
                fig_emp.update_layout(yaxis_title="", xaxis_title="Fraud Rate (%)")
                st.plotly_chart(fig_emp, use_container_width=True)
                st.markdown("""<div class="insight-box">
                    <b>Key insight:</b> Part-time & Other employment types show higher fraud exposure
                    than full-time roles, likely due to lower screening standards.
                </div>""", unsafe_allow_html=True)

        # Feature availability grouped bar
        st.markdown('<div class="chart-title">Feature Availability: Legitimate vs Fraudulent Jobs</div>', unsafe_allow_html=True)
        miss_cols = ['has_salary', 'has_company_profile', 'has_requirements', 'has_benefits', 'has_company_logo']
        miss_cols = [c for c in miss_cols if c in df.columns]
        if miss_cols:
            miss_data = []
            for col_name in miss_cols:
                for label, val in [("Legitimate", 0), ("Fraudulent", 1)]:
                    rate = df[df['fraudulent'] == val][col_name].mean() * 100
                    miss_data.append({
                        'Feature': col_name.replace('has_', '').replace('_', ' ').title(),
                        'Type': label, 'Rate (%)': round(rate, 1)
                    })
            miss_df = pd.DataFrame(miss_data)
            fig_miss = px.bar(
                miss_df, x='Feature', y='Rate (%)', color='Type', barmode='group',
                color_discrete_map={"Legitimate": LEGIT_CLR, "Fraudulent": FRAUD_CLR},
                text='Rate (%)',
            )
            fig_miss.update_traces(texttemplate='%{text:.0f}%', textposition='outside', marker_line_width=0)
            dark_layout(fig_miss, height=360)
            fig_miss.update_layout(legend_title_text='', xaxis_title='', yaxis_title='Availability Rate (%)')
            st.plotly_chart(fig_miss, use_container_width=True)
            st.markdown("""<div class="insight-box">
                <b>Key insight:</b> Fraudulent postings are significantly less likely to include salary,
                company profile, or job requirements — strong red flags for job seekers.
            </div>""", unsafe_allow_html=True)

        # Profile completeness box comparison
        if 'profile_completeness' in df.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_comp = px.box(
                    df, x='fraudulent', y='profile_completeness',
                    color='fraudulent',
                    color_discrete_map={0: LEGIT_CLR, 1: FRAUD_CLR},
                    labels={'fraudulent': 'Job Type', 'profile_completeness': 'Completeness Score'},
                    category_orders={'fraudulent': [0, 1]},
                )
                fig_comp.update_traces(marker_line_width=0)
                fig_comp.update_xaxes(tickvals=[0, 1], ticktext=['Legitimate', 'Fraudulent'])
                dark_layout(fig_comp, "Profile Completeness Score Distribution", height=320)
                st.plotly_chart(fig_comp, use_container_width=True)

            with col_b:
                if 'has_urgency_words' in df.columns:
                    urg_summary = df.groupby('has_urgency_words')['fraudulent'].agg(['mean','count']).reset_index()
                    urg_summary.columns = ['Has Urgency Words', 'Fraud Rate', 'Count']
                    urg_summary['Fraud Rate %'] = (urg_summary['Fraud Rate'] * 100).round(1)
                    urg_summary['Label'] = urg_summary['Has Urgency Words'].map({0: '❌ No Urgency Words', 1: '🚨 Urgency Words Present'})
                    fig_urg = px.bar(
                        urg_summary, x='Label', y='Fraud Rate %',
                        color='Fraud Rate %', color_continuous_scale='RdYlGn_r',
                        text='Fraud Rate %',
                    )
                    fig_urg.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
                    fig_urg.update_coloraxes(showscale=False)
                    dark_layout(fig_urg, "Urgency Words → Fraud Rate Impact", height=320)
                    fig_urg.update_layout(xaxis_title='', yaxis_title='Fraud Rate (%)')
                    st.plotly_chart(fig_urg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — GEOGRAPHIC
    # ══════════════════════════════════════════════════════════════════════════

    # Country ISO code → Full name mapping
    COUNTRY_NAMES = {
        'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
        'AU': 'Australia',     'IN': 'India',           'DE': 'Germany',
        'FR': 'France',        'NL': 'Netherlands',     'SG': 'Singapore',
        'NZ': 'New Zealand',   'IE': 'Ireland',         'PK': 'Pakistan',
        'PH': 'Philippines',   'MY': 'Malaysia',        'BR': 'Brazil',
        'ZA': 'South Africa',  'GR': 'Greece',          'IT': 'Italy',
        'ES': 'Spain',         'PL': 'Poland',          'AE': 'UAE',
        'QA': 'Qatar',         'NG': 'Nigeria',         'KE': 'Kenya',
        'EG': 'Egypt',         'TR': 'Turkey',          'IL': 'Israel',
        'RO': 'Romania',       'HU': 'Hungary',         'CZ': 'Czech Republic',
        'SE': 'Sweden',        'NO': 'Norway',          'DK': 'Denmark',
        'FI': 'Finland',       'CH': 'Switzerland',     'AT': 'Austria',
        'BE': 'Belgium',       'PT': 'Portugal',        'MX': 'Mexico',
        'AR': 'Argentina',     'CO': 'Colombia',        'CL': 'Chile',
        'JP': 'Japan',         'CN': 'China',           'KR': 'South Korea',
        'HK': 'Hong Kong',     'TW': 'Taiwan',          'TH': 'Thailand',
        'ID': 'Indonesia',     'VN': 'Vietnam',         'BD': 'Bangladesh',
        'LK': 'Sri Lanka',     'UA': 'Ukraine',         'RU': 'Russia',
    }

    with tab2:
        if 'country' in df.columns:

            # --- Use pre-processed country column (already cleaned in prepare_dataset) ---
            df_geo = df.copy()

            # --- Aggregate ---
            country_agg = df_geo.groupby('country').agg(
                total=('fraudulent', 'count'),
                fraud=('fraudulent', 'sum')
            ).reset_index()
            country_agg['fraud_rate'] = (
                country_agg['fraud'] / country_agg['total'] * 100
            ).round(2)

            # --- Unknown filter toggle ---
            show_unknown = st.checkbox(
                "Show 'Unknown' country entries", value=False,
                help="Unknown = location was missing or unrecognized in raw data"
            )
            if not show_unknown:
                country_agg = country_agg[country_agg['country'] != 'Unknown']

            top15 = (
                country_agg[country_agg['total'] >= 20]
                .sort_values('fraud', ascending=False)
                .head(15)
            )

            col1, col2 = st.columns([1.8, 1])

            with col1:
                fig_geo = px.bar(
                    top15.sort_values('fraud_rate', ascending=True),
                    y='country', x='fraud_rate',
                    orientation='h',
                    color='fraud_rate',
                    color_continuous_scale='RdYlGn_r',
                    text='fraud_rate',
                    hover_data={'fraud': True, 'total': True},
                )
                fig_geo.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    marker_line_width=0,
                )
                fig_geo.update_coloraxes(showscale=False)
                dark_layout(fig_geo, "Top 15 Countries — Fraud Rate (min. 20 postings)", height=480)
                fig_geo.update_layout(yaxis_title='', xaxis_title='Fraud Rate (%)')
                st.plotly_chart(fig_geo, use_container_width=True)

            with col2:
                st.markdown(
                    '<div class="chart-title" style="margin-top:12px">Country Breakdown</div>',
                    unsafe_allow_html=True
                )
                display_df = (
                    top15[['country', 'total', 'fraud', 'fraud_rate']]
                    .sort_values('fraud', ascending=False)
                    .reset_index(drop=True)
                )
                display_df.columns = ['Country', 'Total', 'Fraud', 'Rate%']
                display_df.index += 1
                st.dataframe(
                    display_df.style
                        .background_gradient(subset=['Rate%'], cmap='RdYlGn_r')
                        .format({'Rate%': '{:.1f}%'}),
                    use_container_width=True, height=460
                )

            st.markdown("""<div class="insight-box">
                <b>Key insight:</b> United States dominates total fraud volume due to
                highest posting count, but smaller countries like Malaysia and Qatar
                show significantly higher fraud <i>rates</i>.
                Always check both absolute numbers and rate together.
            </div>""", unsafe_allow_html=True)
    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — TEXT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if 'desc_length' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                fig_hist = px.histogram(
                    df[df['desc_length'] < 5000],
                    x='desc_length',
                    color='fraudulent',
                    color_discrete_map={0: LEGIT_CLR, 1: FRAUD_CLR},
                    nbins=60,
                    barmode='overlay',
                    opacity=0.75,
                    labels={'desc_length': 'Description Length (chars)', 'fraudulent': 'Job Type'},
                )
                fig_hist.update_traces(marker_line_width=0)
                dark_layout(fig_hist, "Description Length: Legitimate vs Fraudulent", height=340)
                fig_hist.update_layout(legend=dict(
                    orientation='h', y=1.05,
                    itemsizing='constant',
                ))
                # Add mean lines
                for val, color, label in [(0, LEGIT_CLR, 'Legit avg'), (1, FRAUD_CLR, 'Fraud avg')]:
                    mean_val = df[df['fraudulent']==val]['desc_length'].mean()
                    fig_hist.add_vline(x=mean_val, line_dash='dash', line_color=color,
                                       annotation_text=f"{label}: {int(mean_val)}", annotation_position='top right',
                                       annotation_font_color=color)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("""<div class="insight-box">
                    <b>Key insight:</b> Fraudulent job descriptions tend to be <b>shorter</b>
                    — scammers avoid writing detailed, professional-sounding content.
                </div>""", unsafe_allow_html=True)

            with col2:
                # Salary disclosure vs fraud — stacked bar
                if 'has_salary' in df.columns:
                    sal_data = df.groupby(['has_salary', 'fraudulent']).size().reset_index(name='count')
                    sal_data['Salary Status'] = sal_data['has_salary'].map({0: '❌ Not Disclosed', 1: '✅ Disclosed'})
                    sal_data['Job Type'] = sal_data['fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})
                    fig_sal = px.bar(
                        sal_data,
                        x='Salary Status', y='count', color='Job Type',
                        color_discrete_map={'Legitimate': LEGIT_CLR, 'Fraudulent': FRAUD_CLR},
                        barmode='stack',
                        text='count',
                    )
                    fig_sal.update_traces(texttemplate='%{text:,}', textposition='inside', marker_line_width=0)
                    dark_layout(fig_sal, "Salary Disclosure vs Fraud Volume", height=340)
                    fig_sal.update_layout(xaxis_title='', yaxis_title='Job Count', legend_title='')
                    st.plotly_chart(fig_sal, use_container_width=True)
                    st.markdown("""<div class="insight-box">
                        <b>Key insight:</b> The vast majority of fake jobs <b>do not disclose salary</b>.
                        Missing salary info is one of the strongest single fraud predictors.
                    </div>""", unsafe_allow_html=True)

            # Required experience breakdown
            if 'required_experience' in df.columns:
                exp_agg = df.groupby('required_experience')['fraudulent'].agg(['mean','count']).reset_index()
                exp_agg.columns = ['Experience', 'Fraud Rate', 'Count']
                exp_agg = exp_agg[exp_agg['Count'] > 30].sort_values('Fraud Rate', ascending=False)
                exp_agg['Fraud Rate %'] = (exp_agg['Fraud Rate'] * 100).round(1)
                fig_exp = px.bar(
                    exp_agg, x='Experience', y='Fraud Rate %',
                    color='Fraud Rate %', color_continuous_scale='RdYlGn_r',
                    text='Fraud Rate %',
                )
                fig_exp.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
                fig_exp.update_coloraxes(showscale=False)
                dark_layout(fig_exp, "Fraud Rate by Required Experience Level", height=340, xangle=-25)
                fig_exp.update_layout(xaxis_title='', yaxis_title='Fraud Rate (%)')
                st.plotly_chart(fig_exp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — INDUSTRY
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if 'industry' in df.columns:
            ind_agg = df.groupby('industry').agg(
                total=('fraudulent', 'count'),
                fraud=('fraudulent', 'sum')
            ).reset_index()
            ind_agg['fraud_rate'] = (ind_agg['fraud'] / ind_agg['total'] * 100).round(2)
            top_ind = ind_agg[ind_agg['total'] >= 30].sort_values('fraud_rate', ascending=False).head(20)

            col1, col2 = st.columns([1.8, 1])

            with col1:
                fig_ind = px.bar(
                    top_ind.sort_values('fraud_rate', ascending=True),
                    y='industry', x='fraud_rate',
                    orientation='h',
                    color='fraud_rate',
                    color_continuous_scale='RdYlGn_r',
                    text='fraud_rate',
                    hover_data={'fraud': True, 'total': True},
                )
                fig_ind.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside',
                    marker_line_width=0,
                )
                fig_ind.update_coloraxes(showscale=False)
                dark_layout(fig_ind, "Top 20 Industries — Fraud Rate (min. 30 postings)", height=580)
                fig_ind.update_layout(yaxis_title='', xaxis_title='Fraud Rate (%)')
                st.plotly_chart(fig_ind, use_container_width=True)

            with col2:
                # Treemap of fraud volume by industry
                fig_tree = px.treemap(
                    top_ind,
                    path=['industry'],
                    values='fraud',
                    color='fraud_rate',
                    color_continuous_scale='RdYlGn_r',
                    hover_data={'total': True, 'fraud_rate': ':.1f'},
                )
                fig_tree.update_traces(
                    textinfo='label+value',
                    textfont=dict(size=11, color='white'),
                    marker_line_width=1,
                    marker_line_color=DARK_BG,
                )
                fig_tree.update_coloraxes(showscale=False)
                dark_layout(fig_tree, "Fraud Volume by Industry (Treemap)", height=580)
                fig_tree.update_layout(margin=dict(l=4, r=4, t=36, b=4))
                st.plotly_chart(fig_tree, use_container_width=True)

            st.markdown("""<div class="insight-box">
                <b>Key insight:</b> Industries like <b>Oil & Energy, Maritime</b> show high fraud rates
                due to high-pay promises and remote nature. HR recruiters should flag postings from
                these sectors that lack company profiles or salary details.
            </div>""", unsafe_allow_html=True)


#  PAGE 3 – JOB CHECKER
elif page == "🔍  Job Checker":

    st.markdown("""
    <style>
    .jc-hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 18px;
        padding: 36px 30px 28px;
        text-align: center;
        margin-bottom: 28px;
        border: 1px solid #2c5364;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .jc-hero::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at center top, rgba(52,152,219,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .jc-hero h1 { font-size: 2rem; font-weight: 800; color: #e6edf3; margin: 0 0 8px; }
    .jc-hero p  { color: #8b949e; font-size: 0.95rem; margin: 0; }
    .jc-badge-row { display: flex; justify-content: center; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
    .jc-badge {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.78rem; color: #c9d1d9;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="jc-hero">
        <h1>🛡️ Real-Time Job Fraud Checker</h1>
        <p>Powered by the trained ML model, rule-based fraud indicators, and TF-IDF keyword evidence</p>
        <div class="jc-badge-row">
            <span class="jc-badge">⚡ Instant Detection</span>
            <span class="jc-badge">📊 Fraud Probability Gauge</span>
            <span class="jc-badge">🚩 Fraud Indicators Table</span>
            <span class="jc-badge">🔑 Top Fraud Keywords</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error("❌ Model not loaded. Run the notebook first to generate best_model.pkl, tfidf_vectorizer.pkl, numeric_cols.pkl, and model_info.json.")
        st.stop()

    col_form, col_res = st.columns([1, 1])

    with col_form:
        st.markdown("### 📝 Enter Job Details")
        c_title, c_company = st.columns(2)
        with c_title:
            title = st.text_input("JOB TITLE *", placeholder="e.g., Data Analyst")
        with c_company:
            company = st.text_input("COMPANY NAME", placeholder="e.g., Tech Corp")

        desc = st.text_area(
            "JOB DESCRIPTION *",
            height=160,
            placeholder="Paste the complete job description here — role overview, responsibilities, company info..."
        )
        reqs = st.text_area(
            "REQUIREMENTS / QUALIFICATIONS",
            height=100,
            placeholder="e.g., 3+ years Python experience, SQL, statistics, stakeholder communication..."
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            has_sal = st.checkbox("💰 Salary Disclosed", value=False,
                                  help="Check this only if the job posting clearly mentions a salary or pay range.")
        with cc2:
            has_logo = st.checkbox("🏢 Company Logo", value=False,
                                  help="Check this only if the job posting shows a verified company logo.")

        filled_title = bool(str(title).strip())
        filled_company = bool(str(company).strip())
        filled_desc = bool(str(desc).strip())
        filled_reqs = bool(str(reqs).strip())
        completeness_score = (int(filled_title) + int(filled_company) + int(filled_desc) + int(filled_reqs) + int(has_sal) + int(has_logo))
        completeness_pct = int(completeness_score / 6 * 100)

        if completeness_pct >= 80:
            bar_color = "#2ECC71"; bar_label = "Excellent"
        elif completeness_pct >= 50:
            bar_color = "#F39C12"; bar_label = "Good"
        else:
            bar_color = "#E74C3C"; bar_label = "Incomplete"

        chips_html = ""
        for name, ok in [("Title", filled_title), ("Company", filled_company), ("Description", filled_desc), ("Requirements", filled_reqs), ("Salary", has_sal), ("Logo", has_logo)]:
            cls = "badge badge-green" if ok else "badge badge-blue"
            ic = "✔" if ok else "○"
            chips_html += f'<span class="{cls}" style="margin:2px">{ic} {name}</span>'

        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #21262d; border-radius:10px; padding:12px 14px; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:#8b949e; margin-bottom:6px;">
                <span>📊 Form Completeness</span>
                <span style="color:{bar_color}; font-weight:700">{completeness_pct}% — {bar_label}</span>
            </div>
            <div style="background:#21262d; border-radius:8px; height:8px; overflow:hidden;">
                <div style="width:{completeness_pct}%; background:{bar_color}; height:100%; border-radius:8px; transition:width 0.4s ease;"></div>
            </div>
            <div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:4px;">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)

        submitted = st.button("🔍  Analyze Job Posting", type="primary", use_container_width=True, key="jc_submit")

        st.markdown("""
        <div class="info-card" style="margin-top:14px;">
            <b style="color:#3498DB;">💡 Tips for Best Results</b><br>
            📋 Paste the full description — more text improves TF-IDF coverage<br>
            🏢 Company presence and salary disclosure reduce uncertainty<br>
            🚩 Urgency language and thin descriptions are common fraud indicators
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        # ── INPUT VALIDATION ────────────────────────────────────────────────
        MIN_TITLE_LEN = 5
        MIN_DESC_LEN  = 100

        validation_errors = []
        if submitted:
            if not str(title).strip() or len(str(title).strip()) < MIN_TITLE_LEN:
                validation_errors.append(
                    f"❌ **Job Title** is too short — minimum {MIN_TITLE_LEN} characters required. "
                    f"(Entered: {len(str(title).strip())} chars)"
                )
            if not str(desc).strip() or len(str(desc).strip()) < MIN_DESC_LEN:
                validation_errors.append(
                    f"❌ **Job Description** is too short — minimum {MIN_DESC_LEN} characters required for a meaningful prediction. "
                    f"(Entered: {len(str(desc).strip())} chars)"
                )

        if submitted and validation_errors:
            for err in validation_errors:
                st.markdown(f"""
                <div class="warning-card" style="margin-bottom:6px;">
                    {err}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-card" style="margin-top:8px; font-size:0.85rem;">
                💡 <b>Why does this matter?</b> The ML model uses TF-IDF on the description text.
                Very short input has almost zero word contribution — so only binary checkboxes
                influence the score, which creates a <b>misleading result</b>.
            </div>
            """, unsafe_allow_html=True)

        elif submitted and not validation_errors:
            try:
                with st.spinner("🔍 Analyzing job posting with ML model..."):
                    prob, red_flags, top_words = predict_job(title, company, desc, reqs, has_sal, has_logo, model, tfidf, num_cols)
            except ValueError as _feat_err:
                # Handles feature count mismatch between app and saved model
                st.error(f"❌ Model feature mismatch: {_feat_err}")
                st.stop()

            is_fraud = prob >= FRAUD_THRESHOLD
            clr = risk_color(prob)
            lbl = risk_label(prob)
            active_flags = [k for k, v in red_flags.items() if v]
            positive_flags = [k for k, v in red_flags.items() if not v]

            if is_fraud:
                verdict_css = "background:#2d1b1b; border:2px solid #E74C3C; box-shadow:0 0 20px rgba(231,76,60,0.2);"
                verdict_emoji = "🚨"
            elif prob >= 0.20:
                verdict_css = "background:#2d2416; border:2px solid #F39C12; box-shadow:0 0 20px rgba(243,156,18,0.2);"
                verdict_emoji = "⚠️"
            else:
                verdict_css = "background:#1b2d1b; border:2px solid #2ECC71; box-shadow:0 0 20px rgba(46,204,113,0.15);"
                verdict_emoji = "✅"

            st.markdown(f"""
            <div style="{verdict_css} border-radius:14px; padding:20px; text-align:center; margin-bottom:10px;">
                <div style="font-size:2.4rem; font-weight:800; color:{clr}; line-height:1;">{prob*100:.1f}%</div>
                <div style="color:#8b949e; font-size:0.78rem; margin-bottom:8px;">Fraud Probability</div>
                <div style="font-size:1.1rem; font-weight:800; color:{clr}; letter-spacing:0.02em;">{verdict_emoji} {lbl}</div>
                <div style="font-size:0.82rem; color:#c9d1d9; margin-top:4px;">Threshold: {FRAUD_THRESHOLD*100:.0f}% &nbsp;|&nbsp; Red Flags: {len(active_flags)}/6</div>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Fraud Probability (%)", 'font': {'color': '#8b949e', 'size': 13}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b949e', 'tickfont': {'color': '#8b949e', 'size': 10}},
                    'bar': {'color': clr, 'thickness': 0.3},
                    'bgcolor': '#161b22',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 20], 'color': '#1b2d1b'},
                        {'range': [20, 35], 'color': '#2d2416'},
                        {'range': [35, 100], 'color': '#2d1b1b'},
                    ],
                    'threshold': {'line': {'color': '#9B59B6', 'width': 3}, 'thickness': 0.75, 'value': FRAUD_THRESHOLD * 100}
                },
                number={'font': {'color': clr, 'size': 38}, 'suffix': '%'}
            ))
            fig_gauge.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#e6edf3', height=230, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            mk1, mk2, mk3 = st.columns(3)
            with mk1:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{clr}">
                    <div class="kpi-value" style="color:{clr}">{prob*100:.1f}%</div>
                    <div class="kpi-label">Fraud Score</div>
                </div>""", unsafe_allow_html=True)
            with mk2:
                risk_lvl = "HIGH" if is_fraud else ("MED" if prob >= 0.20 else "LOW")
                risk_col2 = "#E74C3C" if is_fraud else ("#F39C12" if prob >= 0.20 else "#2ECC71")
                st.markdown(f"""
                <div class="kpi-card" style="border-color:{risk_col2}">
                    <div class="kpi-value" style="color:{risk_col2}">{risk_lvl}</div>
                    <div class="kpi-label">Risk Level</div>
                </div>""", unsafe_allow_html=True)
            with mk3:
                st.markdown(f"""
                <div class="kpi-card" style="border-color:#E74C3C">
                    <div class="kpi-value" style="color:#E74C3C">{len(active_flags)}/6</div>
                    <div class="kpi-label">Red Flags</div>
                </div>""", unsafe_allow_html=True)

            indicator_df = pd.DataFrame({
                "Indicator": list(red_flags.keys()),
                "Status": ["Triggered" if v else "Clear" for v in red_flags.values()],
            })
            st.markdown("### 🚩 Fraud Indicators Table")
            st.dataframe(indicator_df, use_container_width=True, hide_index=True)

            if active_flags:
                st.markdown(f"""
                <div class="fraud-card" style="margin-top:10px">
                    <div style="color:#E74C3C; font-weight:700; font-size:0.9rem; margin-bottom:8px;">⚠️ Red Flags Detected ({len(active_flags)})</div>
                    {''.join(f'<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;background:rgba(231,76,60,0.08);border-radius:8px;margin:4px 0;border-left:3px solid #E74C3C;font-size:0.85rem;color:#e6edf3;"><span>🚩</span><span>{f}</span></div>' for f in active_flags)}
                </div>
                """, unsafe_allow_html=True)
            if positive_flags:
                st.markdown(f"""
                <div class="legit-card" style="margin-top:8px">
                    <div style="color:#2ECC71; font-weight:700; font-size:0.88rem; margin-bottom:8px;">✅ Positive Signals</div>
                    {''.join(f'<div style="display:flex;align-items:center;gap:8px;padding:6px 8px;background:rgba(46,204,113,0.07);border-radius:8px;margin:4px 0;border-left:3px solid #2ECC71;font-size:0.85rem;color:#e6edf3;"><span>✔</span><span>{RED_FLAG_CHECKS[f][1]}</span></div>' for f in positive_flags[:3])}
                </div>
                """, unsafe_allow_html=True)

            if top_words:
                pills_html = ''.join(f'<span class="badge badge-red">🔑 {w} <span style="opacity:0.6">({v:.3f})</span></span>' for w, v in top_words[:6])
                st.markdown(f"""
                <div class="info-card" style="margin-top:10px; border-left:5px solid #3498DB;">
                    <div style="color:#3498DB; font-weight:700; font-size:0.88rem; margin-bottom:8px;">🔑 Top Fraud Keywords (TF-IDF contribution)</div>
                    {pills_html}
                </div>
                """, unsafe_allow_html=True)

        elif submitted and validation_errors:
            pass  # errors already shown above

        elif submitted:
            st.markdown("""
            <div class="warning-card">
                ⚠️ <b>Job Title</b> and <b>Job Description</b> are required fields.
                Please fill them in and try again.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(145deg,#0d1117,#161b22); border:2px dashed #30363d; border-radius:16px; padding:50px 30px; text-align:center; color:#8b949e;">
                <div style="font-size:3.5rem; margin-bottom:12px;">🔍</div>
                <div style="font-size:1.1rem; font-weight:700; color:#c9d1d9; margin-bottom:6px;">Results will appear here</div>
                <div style="font-size:0.85rem; color:#6e7681;">Fill in the job details on the left<br>and click <b>Analyze Job Posting</b></div>
            </div>
            """, unsafe_allow_html=True)


#  PAGE 5 – MODEL INSIGHTS
elif page == "🤖  Model Insights":
    st.markdown("## 🤖 Model Insights & Performance")

    tab1, tab2, tab3 = st.tabs(["📊 Real Metrics", "🔑 Feature Importance", "🗄️ SQL Integration"])

    with tab1:
        if _mi:
            top_cards = st.columns(5)
            metric_cards = [
                ("Precision", _mi.get("precision"), "#E74C3C"),
                ("Recall", _mi.get("recall"), "#2ECC71"),
                ("F1", _mi.get("f1_score"), "#F39C12"),
                ("AUC-ROC", _mi.get("auc_roc"), "#3498DB"),
                ("Threshold", _mi.get("threshold", FRAUD_THRESHOLD), "#9B59B6"),
            ]
            for col, (label, value, clr) in zip(top_cards, metric_cards):
                with col:
                    display = f"{value*100:.1f}%" if isinstance(value, (int, float)) and label != "Threshold" else str(value)
                    st.markdown(make_kpi_card(display, label, clr), unsafe_allow_html=True)

            metric_rows = [
                ("Best model", MODEL_NAME),
                ("Model family", MODEL_TYPE),
                ("Training records", f"{_mi.get('total_records', 0):,}"),
                ("Fraud rate", f"{_mi.get('fraud_rate_pct', 0):.2f}%"),
                ("TF-IDF features", _mi.get('tfidf_features', 'N/A')),
                ("Numeric features", _mi.get('numeric_features', 'N/A')),
            ]
            metric_df = pd.DataFrame(metric_rows, columns=["Metric", "Value"])
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ model_info.json not found. Save notebook artifacts to display real metrics in this page.")

    with tab2:
        pos_df, neg_df = extract_model_feature_insights(model, tfidf)
        if pos_df.empty and neg_df.empty:
            st.info("Feature importance will appear here once the trained model artifacts are available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if not pos_df.empty:
                    fig_pos = px.bar(pos_df.sort_values("weight"), x="weight", y="keyword", orientation="h", color="weight", color_continuous_scale="Reds", title="Top Fraud-Contributing Words")
                    fig_pos.update_layout(paper_bgcolor='#0d1117', font_color='#e6edf3', showlegend=False)
                    st.plotly_chart(fig_pos, use_container_width=True)
            with col2:
                if not neg_df.empty:
                    fig_neg = px.bar(neg_df.sort_values("weight"), x="weight", y="keyword", orientation="h", color="weight", color_continuous_scale="Greens", title="Top Legitimate-Signal Words")
                    fig_neg.update_layout(paper_bgcolor='#0d1117', font_color='#e6edf3', showlegend=False)
                    st.plotly_chart(fig_neg, use_container_width=True)
            if not pos_df.empty:
                st.dataframe(pd.concat([pos_df.head(10), neg_df.head(10)], ignore_index=True), use_container_width=True, hide_index=True)

    with tab3:
        if DB_AVAILABLE:
            # Clear cached SQL views and reload from MySQL
            col_ref, col_status = st.columns([1, 3])
            with col_ref:
                if st.button("🔄 Refresh SQL Views", key="refresh_sql"):
                    load_sql_views.clear()
                    load_data.clear()
                    st.rerun()
            with col_status:
                st.markdown(
                    "<div style='padding:8px 12px; background:#1b2d1b; border-left:3px solid #2ECC71;"
                    " border-radius:6px; font-size:0.82rem; color:#2ECC71; margin-top:4px;'>"
                    "✅ MySQL Connected — data loaded from <b>vw_fraud_summary</b>, "
                    "<b>vw_industry_risk</b>, <b>vw_high_risk_jobs</b></div>",
                    unsafe_allow_html=True,
                )

            s1, s2, s3 = st.columns(3)
            summary_df = sql_views.get("summary", pd.DataFrame())
            with s1:
                st.markdown(make_kpi_card(f"{int(sql_scalar(summary_df, 'urgency_fraud_count', 0)):,}", "Urgency + Fraud", "#E74C3C"), unsafe_allow_html=True)
            with s2:
                st.markdown(make_kpi_card(f"{int(sql_scalar(summary_df, 'no_salary_fraud_count', 0)):,}", "No Salary + Fraud", "#F39C12"), unsafe_allow_html=True)
            with s3:
                st.markdown(make_kpi_card(f"{sql_scalar(summary_df, 'avg_fraud_completeness', 0)}", "Avg Fraud Completeness", "#3498DB"), unsafe_allow_html=True)

            st.markdown("### 🏭 Industry Risk View")
            st.dataframe(sql_views.get("industry", pd.DataFrame()), use_container_width=True, hide_index=True)
            st.markdown("### 🚩 High Risk Jobs View")
            st.dataframe(sql_views.get("high_risk", pd.DataFrame()).head(15), use_container_width=True, hide_index=True)
        else:
            st.info("MySQL connection unavailable. The app is running on CSV fallback, so SQL view outputs are hidden until the database is available.")


#  PAGE 6 – LIMITATIONS & BIAS
elif page == "⚠️  Limitations & Bias":
    st.markdown("## ⚠️ Known Limitations & Bias")

    limitations = [
        ("🏭 Industry Bias", "#E74C3C", "Some industries are overrepresented in fraudulent examples, so sector-specific false positives can still occur."),
        ("🌐 English-Only Text Features", "#F39C12", "Current TF-IDF features work best on English job descriptions and English resumes."),
        ("📍 Small-Sample Geography", "#3498DB", "Country or city-level fraud rates can be noisy when posting counts are low."),
        ("🔄 Static Model Artifacts", "#9B59B6", "The deployed model depends on saved notebook artifacts. If the data changes, the model should be retrained and versioned."),
        ("📊 Form vs Training Features", "#2ECC71", "The job checker form cannot collect every original dataset field, so some numeric inputs are approximated at inference time."),
        ("🗄️ Local MySQL Dependency", "#E67E22", "SQL KPI cards and views require the fake_job_detection database plus the prepared views to be available locally."),
    ]

    for title_l, clr, desc_l in limitations:
        st.markdown(f"""
        <div class="limit-card" style="border-color:{clr}">
            <b style="color:{clr}">{title_l}</b><br>
            <span style="color:#c9d1d9; font-size:0.9rem">{desc_l}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### ✅ Practical Next Improvements")
    improvements = [
        ("Save validation predictions", "Persist y_true and y_prob from the notebook so the app can render a real threshold curve and confusion matrix."),
        ("Version model artifacts", "Store model_info.json, vectorizer, numeric columns, and training date together to keep deployments reproducible."),
        ("Automate SQL refresh", "Create a repeatable CSV → MySQL load step so Streamlit and Power BI always read the same curated data."),
        ("Expand language coverage", "Train multilingual text features if the project needs non-English job postings or resumes."),
    ]
    for title_i, desc_i in improvements:
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin:6px 0; padding:10px; background:#161b22; border-radius:8px;'>
            <span class='badge badge-purple'>Next</span>
            <span style='color:#e6edf3; font-weight:600; margin:0 10px'>{title_i}</span>
            <span style='color:#8b949e; font-size:0.85rem'>{desc_i}</span>
        </div>""", unsafe_allow_html=True)
