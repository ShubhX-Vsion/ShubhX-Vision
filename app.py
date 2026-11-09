"""
ShubhX Vision — Premium Data Intelligence Studio
"""

import ast
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path  # FIX: Path import missing

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Streamlit may be launched from a different folder than app.py.
# Search the script folder and a few parent folders for the user's .env file.
APP_DIR = Path(__file__).resolve().parent
ENV_CANDIDATES = [APP_DIR / ".env", APP_DIR.parent / ".env",
                  APP_DIR.parent.parent / ".env", Path.cwd() / ".env"]
for env_file in ENV_CANDIDATES:
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=False)
        break
else:
    load_dotenv(override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

st.set_page_config(
    page_title="ShubhX Vision | Data Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- SESSION STATE -----------------------------
DEFAULTS = {
    "theme": "light",
    "data": None,
    "original": None,
    "file_name": None,
    "loaded_signature": None,
    "history": [],
    "chat": [],
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

dark = st.session_state.theme == "dark"
COLORS = {
    "accent": "#7357F6",
    "accent2": "#27B7C5",
    "blue": "#4388F5",
    "success": "#159B72",
    "warning": "#D18B22",
    "danger": "#D65368",
}
if dark:
    BG, FG, MUTED = "#0B1020", "#F4F6FF", "#AAB5CC"
    SURFACE, SURFACE2 = "rgba(20,28,49,.90)", "#151E33"
    LINE, INPUT_BG = "rgba(180,190,220,.16)", "#151E33"
    SHADOW = "0 16px 45px rgba(0,0,0,.28)"
    HERO = "linear-gradient(125deg,#1D2141 0%,#142C3D 100%)"
else:
    BG, FG, MUTED = "#F4F6FC", "#18233B", "#5E6B83"
    SURFACE, SURFACE2 = "rgba(255,255,255,.94)", "#FFFFFF"
    LINE, INPUT_BG = "rgba(43,62,105,.15)", "#FFFFFF"
    SHADOW = "0 14px 38px rgba(36,55,100,.09)"
    HERO = "linear-gradient(125deg,#FFFFFF 0%,#EDF1FF 58%,#E5F8F7 100%)"

# ----------------------------- PREMIUM UI -----------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root {{
 --app-bg:{BG}; --app-fg:{FG}; --app-muted:{MUTED};
 --app-surface:{SURFACE}; --app-surface-2:{SURFACE2}; --app-line:{LINE};
 --app-shadow:{SHADOW}; --accent:{COLORS['accent']}; --accent2:{COLORS['accent2']};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
 background:
   radial-gradient(ellipse at 7% 0%, {'rgba(105,83,220,.19)' if dark else 'rgba(115,87,246,.10)'}, transparent 31%),
   radial-gradient(ellipse at 94% 8%, {'rgba(39,183,197,.13)' if dark else 'rgba(39,183,197,.12)'}, transparent 28%),
   var(--app-bg) !important;
 color:var(--app-fg) !important;
 font-family:'DM Sans',sans-serif;
}}
[data-testid="stHeader"] {{background:transparent !important;}}
.block-container {{max-width:1560px;padding-top:1.35rem;padding-bottom:2.2rem;}}
[data-testid="stSidebar"] {{
 background:var(--app-surface) !important;
 border-right:1px solid var(--app-line);
}}
[data-testid="stSidebar"] > div:first-child {{background:transparent;}}
/* Keep light-theme text readable without forcing chart/canvas internals. */
.stMarkdown, .stText, p, label, [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {{
 color:var(--app-fg);
}}
h1,h2,h3,h4 {{color:var(--app-fg) !important;font-family:'Manrope',sans-serif !important;letter-spacing:-.45px;}}
small, .muted {{color:var(--app-muted) !important;}}
.eyebrow {{
 color:{'#B4A8FF' if dark else '#6750D8'} !important;
 font-size:11px;font-weight:800;letter-spacing:2.1px;text-transform:uppercase;
}}
.brand {{font:800 22px Manrope,sans-serif;letter-spacing:-.8px;color:var(--app-fg);}}
.hero {{
 position:relative;overflow:hidden;padding:clamp(22px,3vw,38px);
 border:1px solid var(--app-line);border-radius:26px;background:{HERO};
 box-shadow:var(--app-shadow);transition:transform .22s ease,box-shadow .22s ease;
}}
.hero:hover {{transform:translateY(-2px);box-shadow:0 22px 55px rgba(55,65,120,.15);}}
.hero h1 {{font-size:clamp(30px,4vw,48px);line-height:1.12;margin:10px 0 8px;color:var(--app-fg) !important;}}
.hero p {{color:var(--app-muted) !important;font-size:15px;margin:0;max-width:720px;}}
.hero-orb {{
 position:absolute;width:210px;height:210px;border-radius:50%;right:5%;top:-80px;
 background:radial-gradient(circle,rgba(115,87,246,.20),rgba(39,183,197,.05) 65%,transparent 70%);
 pointer-events:none;
}}
.metric-card {{
 position:relative;min-height:126px;padding:19px 20px;border:1px solid var(--app-line);
 border-radius:20px;background:var(--app-surface);box-shadow:var(--app-shadow);
 transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
}}
.metric-card:hover {{transform:translateY(-5px);border-color:var(--accent);box-shadow:0 18px 38px rgba(85,75,170,.17);}}
.metric-label {{font-size:11px;font-weight:800;letter-spacing:1px;color:var(--app-muted) !important;}}
.metric-value {{font:800 clamp(24px,2.2vw,31px) Manrope,sans-serif;color:var(--app-fg) !important;margin-top:9px;overflow-wrap:anywhere;}}
.metric-note {{font-size:11px;color:var(--app-muted) !important;margin-top:4px;overflow-wrap:anywhere;}}
.section-kicker {{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;font-weight:800;color:var(--app-muted) !important;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
 border-color:var(--app-line) !important;border-radius:18px;
}}
/* Inputs and select menus: readable in both themes. */
.stTextInput input,.stTextArea textarea,.stNumberInput input,
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea {{
 background:var(--app-surface-2) !important;color:var(--app-fg) !important;
 border:1px solid var(--app-line) !important;border-radius:12px !important;
 -webkit-text-fill-color:var(--app-fg) !important;
}}
[data-baseweb="select"] > div {{
 background:var(--app-surface-2) !important;color:var(--app-fg) !important;
 border-color:var(--app-line) !important;border-radius:12px !important;
}}
[data-baseweb="select"] *,[role="listbox"] *,[role="option"] {{
 color:var(--app-fg) !important;
}}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {{
 background:var(--app-surface-2) !important;
}}
.stCheckbox label,.stRadio label,.stMultiSelect label {{color:var(--app-fg) !important;}}
[data-testid="stFileUploader"] {{
 border:1px dashed var(--accent);border-radius:16px;padding:12px;
 background:var(--app-surface);
}}
[data-testid="stFileUploader"] section {{background:transparent !important;}}
div.stButton > button, div.stDownloadButton > button {{
 min-height:42px;border:1px solid transparent;border-radius:12px;
 background:linear-gradient(120deg,#7357F6,#4388F5);color:white !important;
 font-weight:700;transition:all .2s ease;box-shadow:0 6px 16px rgba(105,83,220,.16);
}}
div.stButton > button:hover,div.stDownloadButton > button:hover {{
 transform:translateY(-2px);box-shadow:0 11px 24px rgba(105,83,220,.25);
 border-color:rgba(255,255,255,.45);color:white !important;
}}
div.stButton > button:disabled {{opacity:.48;transform:none;}}
[data-testid="stDataFrame"],[data-testid="stTable"] {{
 border:1px solid var(--app-line);border-radius:14px;overflow:hidden;
 background:var(--app-surface);
}}
[data-testid="stTabs"] button {{font-weight:700;color:var(--app-muted);}}
[data-testid="stTabs"] button[aria-selected="true"] {{color:var(--accent) !important;}}
[data-testid="stAlert"] {{
 border-radius:14px;
 background:{'rgba(40,52,78,.92)' if dark else '#EAF0FF'} !important;
 border:1px solid var(--app-line) !important;
}}
[data-testid="stAlert"] *, [data-testid="stAlert"] p,
[data-testid="stAlert"] span {{
 color:var(--app-fg) !important;
 -webkit-text-fill-color:var(--app-fg) !important;
}}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section span {{
 color:var(--app-fg) !important;
 -webkit-text-fill-color:var(--app-fg) !important;
}}
[data-testid="stFileUploader"] button {{
 background:{'#26324B' if dark else '#E8ECF5'} !important;
 color:var(--app-fg) !important;
 border:1px solid var(--app-line) !important;
 box-shadow:none !important;
}}
[data-testid="stFileUploader"] button * {{
 color:var(--app-fg) !important;
 -webkit-text-fill-color:var(--app-fg) !important;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
 color:var(--app-fg) !important;
}}
[data-testid="stChatMessage"] {{
 border:1px solid var(--app-line);border-radius:16px;background:var(--app-surface);
}}
[data-testid="stPlotlyChart"] {{border:1px solid var(--app-line);border-radius:16px;padding:7px;background:var(--app-surface);}}
hr {{border-color:var(--app-line) !important;}}
footer,#MainMenu {{visibility:hidden;}}
/* Tooltips appear on hover for metric cards and compact help labels. */
[data-tooltip] {{position:relative;}}
[data-tooltip]:hover::after {{
 content:attr(data-tooltip);position:absolute;z-index:9999;left:50%;bottom:calc(100% + 10px);
 transform:translateX(-50%);width:max-content;max-width:260px;white-space:normal;
 padding:9px 12px;border-radius:10px;background:{'#F4F5FF' if dark else '#18233B'};
 color:{'#18233B' if dark else '#FFFFFF'};font-size:11px;font-weight:600;line-height:1.4;
 box-shadow:0 10px 28px rgba(0,0,0,.22);pointer-events:none;
}}
.footer {{
 margin-top:34px;padding:19px 10px 8px;text-align:center;
 border-top:1px solid var(--app-line);color:var(--app-muted) !important;
 font-size:11px;letter-spacing:.65px;line-height:1.8;
}}
.footer strong {{color:var(--app-fg) !important;font-weight:800;}}
</style>
""",
    unsafe_allow_html=True,
)


def metric(label, value, note="", tooltip=""):
    tip = f' data-tooltip="{tooltip}"' if tooltip else ""
    st.markdown(
        f'<div class="metric-card"{tip}><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def log(message):
    st.session_state.history.insert(0, datetime.now().strftime("%H:%M:%S") + " · " + message)
    st.session_state.history = st.session_state.history[:100]


def style_plot(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FG, family="DM Sans"),
        margin=dict(l=16, r=16, t=42, b=18),
        hoverlabel=dict(
            bgcolor=SURFACE2, font_color=FG, bordercolor=LINE,
            font_family="DM Sans",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FG)),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, tickfont=dict(color=FG),
        title_font=dict(color=FG), automargin=True,
    )
    fig.update_yaxes(
        gridcolor=LINE, zeroline=False, tickfont=dict(color=FG),
        title_font=dict(color=FG), automargin=True,
    )
    return fig


def show_plot(fig, height=350, key=None):
    st.plotly_chart(style_plot(fig, height), use_container_width=True, config={"displaylogo": False}, key=key)


def safe_calculation(expression, frame):
    """Evaluate arithmetic expressions using only numeric columns and safe AST nodes."""
    if not expression.strip():
        raise ValueError("Enter an expression first.")
    tree = ast.parse(expression, mode="eval")
    # Explicit AST validation; calls, attributes, indexing and imports are rejected.
    permitted = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    }
    for node in ast.walk(tree):
        if type(node) not in permitted:
            raise ValueError("Use only column names, numbers, parentheses and + - * / // % ** operators.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(node, ast.Name) and node.id not in frame.columns:
            raise ValueError(f"Unknown column: {node.id}")
        if isinstance(node, ast.Pow) and isinstance(node, ast.BinOp):
            pass
    env = {}
    for column in frame.columns:
        if re.fullmatch(r"[A-Za-z_]\w*", str(column)):
            env[str(column)] = pd.to_numeric(frame[column], errors="coerce")
    return eval(compile(tree, "<calculation>", "eval"), {"__builtins__": {}}, env)


def build_local_summary(frame):
    """Create a useful dataset summary without requiring an external AI API."""
    row_count, column_count = frame.shape
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    other_columns = [column for column in frame.columns if column not in numeric_columns]
    missing_total = int(frame.isna().sum().sum())
    duplicate_total = int(frame.duplicated().sum())
    parts = [
        "## Dataset summary",
        f"- **Rows:** {row_count:,}",
        f"- **Columns:** {column_count:,}",
        f"- **Numeric columns:** {len(numeric_columns)}",
        f"- **Categorical / other columns:** {len(other_columns)}",
        f"- **Missing cells:** {missing_total:,} ({missing_total / max(row_count * column_count, 1) * 100:.1f}%)",
        f"- **Duplicate rows:** {duplicate_total:,}",
    ]
    if numeric_columns:
        stats = frame[numeric_columns].describe().T
        parts += ["", "### Numeric field highlights"]
        for column, values in stats.iterrows():
            mean_value = values.get("mean", np.nan)
            min_value = values.get("min", np.nan)
            max_value = values.get("max", np.nan)
            parts.append(
                f"- **{column}:** mean {mean_value:,.2f}, "
                f"minimum {min_value:,.2f}, maximum {max_value:,.2f}"
            )
    if other_columns:
        parts += ["", "### Categorical field highlights"]
        for column in other_columns[:8]:
            counts = frame[column].fillna("Missing").astype(str).value_counts().head(4)
            top_values = ", ".join(f"{name} ({count:,})" for name, count in counts.items())
            parts.append(f"- **{column}:** {top_values or 'No values'}")
    if missing_total:
        missing_by_column = frame.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
        parts += ["", "### Missing values by column"]
        parts.extend(f"- **{column}:** {int(count):,}" for column, count in missing_by_column.items())
        parts += ["", "### Initial observation",
              "Use the Data Cleaning section to review missing values and duplicates before drawing final conclusions."]

    return "\n".join(parts)


def dataset_profile(frame):
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    missing_count = int(frame.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    return numeric, categorical, missing_count, duplicate_count


# ----------------------------- HEADER / SIDEBAR -----------------------------
top_left, top_right = st.columns([7, 2])
with top_left:
    st.markdown('<div class="eyebrow">✦ DATA INTELLIGENCE STUDIO</div>', unsafe_allow_html=True)
with top_right:
    theme_label = "🌙 Switch to dark" if not dark else "☀️ Switch to light"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "dark" if not dark else "light"
        st.rerun()

with st.sidebar:
    st.markdown('<div class="brand">✦ ShubhX Vision</div>', unsafe_allow_html=True)
    st.caption("Your premium analytics workspace")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Data Cleaning", "Data Analysis", "AI Assistant", "Data Lab", "Export Center"],
        label_visibility="visible",
    )
    st.markdown("#### Add your dataset")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"], help="Supported formats: CSV, XLSX and XLS.")
    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if signature != st.session_state.loaded_signature:
            try:
                uploaded.seek(0)
                loaded = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
                if loaded.empty or len(loaded.columns) == 0:
                    st.warning("The uploaded file has no usable rows or columns.")
                else:
                    st.session_state.data = loaded.copy()
                    st.session_state.original = loaded.copy()
                    st.session_state.file_name = uploaded.name
                    st.session_state.loaded_signature = signature
                    st.session_state.chat = []
                    log(f"Loaded {uploaded.name}")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")
    if st.session_state.data is not None:
        st.caption(f"Current file: **{st.session_state.file_name or 'Dataset'}**")
        if st.button("↺ Reset to original data", use_container_width=True):
            st.session_state.data = st.session_state.original.copy()
            log("Restored original dataset")
            st.rerun()
        if st.button("✕ Clear dataset", use_container_width=True):
            st.session_state.data = None
            st.session_state.original = None
            st.session_state.file_name = None
            st.session_state.loaded_signature = None
            st.session_state.chat = []
            log("Cleared dataset")
            st.rerun()
    st.divider()
    st.caption("PRIVATE WORKSPACE · LOCAL DATA PROCESSING")

st.markdown(
    '<div class="hero"><div class="hero-orb"></div>'
    '<div class="eyebrow">INSIGHT · CLEAN · DISCOVER</div>'
    '<h1>See your data differently.</h1>'
    '<p>Explore, clean, visualize and understand your data in one polished analytics workspace.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------- EMPTY STATE -----------------------------
df = st.session_state.data
if df is None:
    st.markdown("### Your workspace is ready")
    st.caption("Upload your own CSV or Excel file from the sidebar, or explore the built-in demo dataset.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric("SMART CLEANING", "01", "Missing values · duplicates · outliers", "Clean and prepare your dataset.")
    with c2:
        metric("VISUAL ANALYTICS", "02", "Interactive charts · correlations", "Explore patterns and relationships.")
    with c3:
        metric("AI DATA ASSISTANT", "03", "Dataset-aware answers", "Ask questions about the uploaded data.")
    st.write("")
    if st.button("✨ Explore sample dataset", type="primary", use_container_width=True):
        rng = np.random.default_rng(12)
        demo = pd.DataFrame({
            "Customer": [f"C{i:03}" for i in range(1, 151)],
            "Age": rng.integers(18, 70, 150),
            "Monthly_Charge": rng.normal(850, 180, 150).round(2),
            "Tenure": rng.integers(1, 73, 150),
            "Plan": rng.choice(["Basic", "Standard", "Premium"], 150),
            "Region": rng.choice(["North", "South", "East", "West"], 150),
            "Churn": rng.choice(["Yes", "No"], 150, p=[0.27, 0.73]),
        })
        demo.loc[[3, 17, 45], "Age"] = np.nan
        demo.loc[12, "Monthly_Charge"] = 2400
        st.session_state.data = demo
        st.session_state.original = demo.copy()
        st.session_state.file_name = "sample_dataset.csv"
        st.session_state.loaded_signature = None
        log("Loaded sample dataset")
        st.rerun()
    st.info("Tip: The sample dataset includes a few intentional data-quality issues so you can test the cleaning tools.")
    st.stop()

# ----------------------------- DATA SUMMARY -----------------------------
if not isinstance(df, pd.DataFrame):
    st.error("The current dataset is not valid. Please upload it again.")
    st.stop()
rows, cols = df.shape
num, cat, missing, dups = dataset_profile(df)
missing_pct = (missing / max(rows * cols, 1)) * 100

# ----------------------------- OVERVIEW -----------------------------
if page == "Overview":
    st.markdown("### Workspace overview")
    st.caption(f"Currently exploring: **{st.session_state.file_name or 'Dataset'}**")
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        metric("TOTAL ROWS", f"{rows:,}", "Records in current dataset", "Total number of records.")
    with k2:
        metric("TOTAL COLUMNS", f"{cols:,}", f"{len(num)} numeric · {len(cat)} categorical", "Number of fields in your dataset.")
    with k3:
        metric("MISSING CELLS", f"{missing:,}", f"{missing_pct:.1f}% of all cells", "Cells that contain no value.")
    with k4:
        metric("DUPLICATE ROWS", f"{dups:,}", "Exact repeated records", "Rows that are identical across all columns.")
    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(12), use_container_width=True, height=365, hide_index=True)
    with right:
        st.markdown("#### Data composition")
        composition = pd.DataFrame({"Field type": ["Numeric", "Categorical / other"], "Columns": [len(num), len(cat)]})
        fig = px.pie(composition, names="Field type", values="Columns", hole=0.68,
                     color_discrete_sequence=[COLORS["accent"], COLORS["accent2"]])
        fig.update_traces(textposition="outside", textinfo="label+value")
        show_plot(fig, 365)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Missing values by column")
        missing_frame = df.isna().sum().sort_values(ascending=False).head(15).rename("Missing").reset_index()
        missing_frame.columns = ["Column", "Missing"]
        if missing_frame["Missing"].sum() == 0:
            st.success("Great! No missing values found.")
        else:
            fig = px.bar(missing_frame, x="Column", y="Missing", color="Missing",
                         color_continuous_scale="Purples", text="Missing")
            fig.update_layout(coloraxis_showscale=False)
            show_plot(fig, 340)
    with right:
        st.markdown("#### Column profile")
        profile = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique": df.nunique(dropna=True).values,
        })
        st.dataframe(profile, use_container_width=True, height=340, hide_index=True)
    with st.expander("View complete dataset", expanded=False):
        st.dataframe(df, use_container_width=True, height=420, hide_index=True)

# ----------------------------- DATA CLEANING -----------------------------
elif page == "Data Cleaning":
    st.markdown("### Data Cleaning Studio")
    st.caption("Apply changes carefully. Use the sidebar reset button to restore the original upload.")
    missing_tab, duplicate_tab, outlier_tab, text_tab = st.tabs(
        ["🧩 Missing values", "♻️ Duplicates", "📍 Outliers", "🔤 Text & types"]
    )
    with missing_tab:
        affected = df.columns[df.isna().any()].tolist()
        if not affected:
            st.success("No missing values found in the current dataset.")
        else:
            st.write(f"**{len(affected)} columns** contain missing values.")
            selected = st.multiselect("Columns to clean", affected, default=affected)
            method = st.selectbox("Treatment method", ["Drop rows", "Fill mean / mode", "Fill median / mode", "Custom value"])
            custom = st.text_input("Custom replacement value", "Unknown", disabled=method != "Custom value")
            if st.button("Apply missing-value treatment", type="primary", disabled=not selected):
                result = df.copy()
                if method == "Drop rows":
                    result = result.dropna(subset=selected)
                else:
                    for column in selected:
                        if method == "Custom value":
                            value = custom
                            if pd.api.types.is_numeric_dtype(result[column]):
                                try:
                                    value = float(custom)
                                except ValueError:
                                    st.error(f"Enter a number for numeric column '{column}'.")
                                    st.stop()
                        elif pd.api.types.is_numeric_dtype(result[column]):
                            value = result[column].mean() if method.startswith("Fill mean") else result[column].median()
                        else:
                            modes = result[column].mode(dropna=True)
                            value = modes.iloc[0] if not modes.empty else "Unknown"
                        result[column] = result[column].fillna(value)
                st.session_state.data = result
                log("Treated missing values")
                st.success("Missing-value treatment applied.")
                st.rerun()
    with duplicate_tab:
        st.markdown("#### Duplicate record check")
        st.metric("Duplicate rows detected", f"{dups:,}")
        if st.button("Remove duplicate rows", type="primary", disabled=dups == 0):
            st.session_state.data = df.drop_duplicates().copy()
            log("Removed duplicate rows")
            st.rerun()
    with outlier_tab:
        if not num:
            st.info("No numeric columns are available for IQR outlier detection.")
        else:
            selected_num = st.selectbox("Numeric column", num)
            values = df[selected_num].dropna()
            if values.empty:
                st.info("This column has no numeric values to inspect.")
            else:
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (df[selected_num] < lower) | (df[selected_num] > upper)
                outlier_count = int(outlier_mask.sum())
                st.info(f"IQR range: **{lower:,.3f} to {upper:,.3f}** · Potential outliers: **{outlier_count:,}**")
                action = st.radio("Treatment", ["Remove rows", "Cap values"], horizontal=True)
                if st.button("Apply IQR treatment", type="primary", disabled=outlier_count == 0):
                    result = df.copy()
                    if action == "Remove rows":
                        result = result.loc[~outlier_mask].copy()
                    else:
                        result[selected_num] = result[selected_num].clip(lower, upper)
                    st.session_state.data = result
                    log(f"IQR treatment on {selected_num}: {action}")
                    st.rerun()
    with text_tab:
        selected_col = st.selectbox("Choose a column", df.columns, key="text_type_column")
        operation = st.selectbox("Operation", ["Trim spaces", "Lowercase", "Uppercase", "Convert numeric", "Convert datetime"])
        if st.button("Apply column operation", type="primary"):
            result = df.copy()
            if operation == "Trim spaces":
                result[selected_col] = result[selected_col].astype("string").str.strip()
            elif operation == "Lowercase":
                result[selected_col] = result[selected_col].astype("string").str.lower()
            elif operation == "Uppercase":
                result[selected_col] = result[selected_col].astype("string").str.upper()
            elif operation == "Convert numeric":
                result[selected_col] = pd.to_numeric(result[selected_col], errors="coerce")
            else:
                result[selected_col] = pd.to_datetime(result[selected_col], errors="coerce")
            st.session_state.data = result
            log(f"{operation}: {selected_col}")
            st.rerun()
    with st.expander("Recent activity", expanded=True):
        if st.session_state.history:
            for entry in st.session_state.history[:12]:
                st.caption("• " + entry)
        else:
            st.caption("Your actions will appear here.")

# ----------------------------- DATA ANALYSIS -----------------------------
elif page == "Data Analysis":
    st.markdown("### Interactive Data Analysis")
    st.caption("Choose a chart and fields. Hover over chart elements to inspect their values.")
    chart_types = ["Histogram", "Box plot", "Violin plot", "Scatter plot", "Bar chart", "Line chart", "Pie chart", "Area chart"]
    chart_kind = st.selectbox("Visualization", chart_types)
    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        x_col = st.selectbox("X axis / Category", df.columns)
    with col_b:
        y_options = num if num else df.columns.tolist()
        y_col = st.selectbox("Y axis / Measure", y_options)
    with col_c:
        color_choice = st.selectbox("Color grouping", ["None"] + df.columns.tolist())
    color_col = None if color_choice == "None" else color_choice
    try:
        if chart_kind == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col, marginal="box")
        elif chart_kind == "Box plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col)
        elif chart_kind == "Violin plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True)
        elif chart_kind == "Scatter plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data=list(df.columns[:5]))
        elif chart_kind == "Bar chart":
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "median", "count"])
            if aggregation == "count":
                summary = df.groupby(x_col, dropna=False).size().reset_index(name="Value")
            else:
                if not pd.api.types.is_numeric_dtype(df[y_col]):
                    st.warning("Choose a numeric Y-axis field for this aggregation.")
                    st.stop()
                summary = df.groupby(x_col, dropna=False)[y_col].agg(aggregation).reset_index(name="Value")
            fig = px.bar(summary, x=x_col, y="Value", color=color_col if color_col in summary.columns else None, text_auto=".3s")
        elif chart_kind == "Line chart":
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                st.warning("Choose a numeric Y-axis field for a line chart.")
                st.stop()
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, color=color_col, markers=True)
        elif chart_kind == "Pie chart":
            pie_data = df[x_col].fillna("Missing").astype(str).value_counts().head(18).rename_axis(x_col).reset_index(name="Count")
            fig = px.pie(pie_data, names=x_col, values="Count", hole=0.45)
        else:
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                st.warning("Choose a numeric Y-axis field for an area chart.")
                st.stop()
            fig = px.area(df.sort_values(x_col), x=x_col, y=y_col, color=color_col)
        show_plot(fig, 480)
    except Exception as exc:
        st.warning(f"Unable to create this chart with the selected fields: {exc}")

    st.markdown("#### Correlation matrix")
    if len(num) > 1:
        corr = df[num].corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        show_plot(heat, max(360, min(620, 50 * len(num))))
    else:
        st.info("At least two numeric columns are required for a correlation matrix.")

    st.markdown("#### Group-by summary")
    g1, g2, g3 = st.columns(3)
    with g1:
        group_col = st.selectbox("Group field", df.columns, key="group_field")
    with g2:
        measure_col = st.selectbox("Measure field", num if num else df.columns, key="measure_field")
    with g3:
        agg = st.selectbox("Aggregation", ["mean", "sum", "min", "max", "count"], key="group_aggregation")
    try:
        if agg != "count" and not pd.api.types.is_numeric_dtype(df[measure_col]):
            st.info("Select a numeric measure for mean, sum, min or max.")
        else:
            summary = df.groupby(group_col, dropna=False)[measure_col].agg(agg).reset_index()
            st.dataframe(summary, use_container_width=True, height=300, hide_index=True)
            st.download_button("⬇ Download summary CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                               "group_summary.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Could not calculate this summary: {exc}")

# ----------------------------- AI ASSISTANT -----------------------------
elif page == "AI Assistant":
    st.markdown("### AI Data Assistant")
    api_key = GEMINI_API_KEY
    if api_key:
        st.caption("🟢 Gemini is configured. Local dataset summaries are also available.")
    else:
        st.caption("🟡 Local AI mode is active. Dataset summaries and basic analysis work without a Gemini key.")
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about your data: summarize, missing values, trends...")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = ""
        question = prompt.lower().strip()
        wants_summary = any(term in question for term in (
            "summar", "all data", "whole data", "entire data", "dataset overview",
            "overview of data", "describe data", "describe the data", "full data"
        ))

        with st.chat_message("assistant"):
            # Use Gemini for open-ended questions when a key is configured.
            if api_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key, http_options={"timeout": 120000})
                    numeric_stats = df[num].describe().round(3).to_dict() if num else {}
                    context = {
                        "shape": [rows, cols],
                        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
                        "missing_by_column": df.isna().sum().to_dict(),
                        "duplicate_rows": dups,
                        "numeric_summary": numeric_stats,
                        "categorical_top_values": {
                            str(c): df[c].fillna("Missing").astype(str).value_counts().head(8).to_dict()
                            for c in cat[:12]
                        },
                        "sample_rows": df.head(8).replace({np.nan: None}).to_dict(orient="records"),
                    }
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=(
                            "You are a careful data analyst. Answer in clear, simple English. "
                            "Use only the supplied dataset context; do not invent values. "
                            "For a request to summarize all data, provide dataset size, data types, "
                            "missing values, duplicates, numeric highlights and categorical highlights. "
                            "Mention limitations where appropriate.\n"
                            f"DATASET CONTEXT:\n{json.dumps(context, default=str)}\n"
                            f"USER QUESTION: {prompt}"
                        ),
                    )
                    answer = response.text or build_local_summary(df)
                except Exception as exc:
                    answer = build_local_summary(df)
                    answer += f"\n\n*Gemini could not respond ({type(exc).__name__}). Showing the local dataset summary instead.*"
            elif wants_summary:
                answer = build_local_summary(df)
            elif "missing" in question or "null" in question or "empty" in question:
                answer = "## Missing values by column\n\n" + df.isna().sum().to_string()
            elif "duplicate" in question:
                answer = f"There are **{dups:,} duplicate rows** in the current dataset."
            elif "column" in question or "data type" in question or "field" in question:
                answer = pd.DataFrame({
                    "Column": df.columns,
                    "Data type": df.dtypes.astype(str).values,
                    "Missing": df.isna().sum().values,
                    "Unique": df.nunique(dropna=True).values,
                }).to_string(index=False)
            elif "statistics" in question or "describe" in question or "numeric" in question:
                answer = df.describe(include="all").transpose().to_string()
            elif "shape" in question or "size" in question or "how many rows" in question:
                answer = f"The dataset has **{rows:,} rows** and **{cols:,} columns**."
            elif "chart" in question or "visual" in question or "graph" in question:
                answer = (
                    f"**Numeric columns:** {', '.join(map(str, num)) or 'None'}\n\n"
                    f"**Categorical / other columns:** {', '.join(map(str, cat)) or 'None'}\n\n"
                    "Try a histogram for distributions, a bar chart for categories, "
                    "a scatter plot for relationships, or a correlation matrix for numeric fields."
                )
            else:
                answer = (
                    f"The dataset contains **{rows:,} rows** and **{cols:,} columns**, "
                    f"with **{missing:,} missing cells** and **{dups:,} duplicate rows**.\n\n"
                    "Try asking: **Summarize all data**, **Show missing values**, "
                    "**Find duplicate rows**, or **List all columns**."
                )
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ----------------------------- DATA LAB -----------------------------
elif page == "Data Lab":
    st.markdown("### Data Lab")
    stats_tab, filter_tab, calc_tab, sample_tab = st.tabs(
        ["📊 Statistics", "🔎 Filter rows", "🧮 Calculated column", "🎯 Sampling"]
    )
    with stats_tab:
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True, height=440)
    with filter_tab:
        filter_col = st.selectbox("Filter column", df.columns, key="filter_column")
        series = df[filter_col]
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if valid.empty:
                st.info("This column has no numeric values.")
                filtered = df.iloc[0:0]
            else:
                minimum, maximum = float(valid.min()), float(valid.max())
                if minimum == maximum:
                    st.caption(f"All non-empty values equal {minimum:g}.")
                    filtered = df[series == minimum]
                else:
                    low, high = st.slider("Select value range", minimum, maximum, (minimum, maximum), key="numeric_filter_range")
                    filtered = df[series.between(low, high)]
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            chosen = st.multiselect("Keep values", options, default=options, key="category_filter_values")
            filtered = df[series.astype(str).isin(chosen)]
        st.caption(f"Showing **{len(filtered):,}** of **{rows:,}** rows")
        st.dataframe(filtered, use_container_width=True, height=350, hide_index=True)
        st.download_button("⬇ Download filtered CSV", filtered.to_csv(index=False).encode("utf-8-sig"),
                           "filtered_data.csv", "text/csv")
    with calc_tab:
        st.caption("Create a new numeric field with arithmetic expressions, for example: Monthly_Charge * 12")
        new_name = st.text_input("New column name", "Calculated")
        expression = st.text_input("Formula", placeholder="Monthly_Charge * 12")
        st.caption("Allowed: column names, numbers, parentheses and +  -  *  /  //  %  **")
        if st.button("Create calculated column", type="primary"):
            try:
                if not re.fullmatch(r"[A-Za-z_]\w*", new_name.strip()):
                    raise ValueError("Use a simple column name (letters, numbers and underscores; do not start with a number).")
                if new_name in df.columns:
                    raise ValueError("That column name already exists.")
                result = df.copy()
                result[new_name] = safe_calculation(expression, df)
                st.session_state.data = result
                log(f"Created calculated column: {new_name}")
                st.success(f"Created '{new_name}'.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with sample_tab:
        max_rows = max(1, len(df))
        sample_count = st.number_input("Number of rows", min_value=1, max_value=max_rows,
                                       value=min(20, max_rows), step=1)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
        sample = df.sample(n=int(sample_count), random_state=int(seed))
        st.dataframe(sample, use_container_width=True, height=350, hide_index=True)
        st.download_button("⬇ Download sample CSV", sample.to_csv(index=False).encode("utf-8-sig"),
                           "sample_rows.csv", "text/csv")

# ----------------------------- EXPORT CENTER -----------------------------
elif page == "Export Center":
    st.markdown("### Export Center")
    st.caption(f"Ready to export **{rows:,} rows** and **{cols:,} columns**.")
    exp1, exp2, exp3 = st.columns(3, gap="medium")
    with exp1:
        st.download_button("⬇ Export CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "shubhx_cleaned.csv", "text/csv", use_container_width=True)
    with exp2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        st.download_button("⬇ Export Excel", excel_buffer.getvalue(),
                           "shubhx_cleaned.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with exp3:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "file_name": st.session_state.file_name,
            "rows": rows,
            "columns": cols,
            "missing_cells": missing,
            "missing_percentage": round(missing_pct, 2),
            "duplicate_rows": dups,
            "column_types": {str(column): str(df[column].dtype) for column in df.columns},
        }
        st.download_button("⬇ Export quality report", json.dumps(report, indent=2, default=str),
                           "shubhx_quality_report.json", "application/json",
                           use_container_width=True)
    st.markdown("#### Export data profile")
    export_profile = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Unique values": df.nunique(dropna=True).values,
    })
    st.dataframe(export_profile, use_container_width=True, height=360, hide_index=True)

# ----------------------------- FOOTER -----------------------------
st.markdown(
    '<div class="footer"><strong>DEVELOPED BY-SHUBHAM SHARMA (DATA ANALYTICS AND PROMPT ENGINEER)</strong>'
    '<br>SHUBHX VISION · DATA INTELLIGENCE STUDIO · ALL RIGHTS RESERVED</div>',
    unsafe_allow_html=True,
)
