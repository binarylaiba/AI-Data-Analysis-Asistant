"""
app.py
------
Dynamic, Secure Streamlit UI for the AI Data Analysis Assistant.
Works with the default dataset.csv and any newly uploaded CSV file.

Features:
  1. Strict Security & API Protection: Loads GROQ_API_KEY exclusively from .env
     using os.getenv('GROQ_API_KEY'). Provides password-masked sidebar input fallback
     if environment key is missing. NEVER hardcodes or logs keys.
  2. Dynamic CSV Upload & Data Summary: st.file_uploader for CSVs (defaults to dataset.csv).
     Displays row/col counts, column dtypes, missing values count, and interactive st.dataframe.
  3. Dynamic Multi-Graph Visualizations: Auto-detects categorical & numeric columns.
     Interactive dropdown (st.selectbox) for Bar Chart, Pie Chart, Line Chart, Histogram,
     and Scatter Plot using Seaborn/Matplotlib with clean labels, titles, and PNG export.
  4. AI Explanation & Interactive Q&A: Groq API (llama-3.3-70b-versatile) generates
     2-sentence chart insights and instant grounded responses to user questions.

Run:
  py -m streamlit run app.py
"""

import io
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Glassmorphism Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1e1b4b 50%, #24243e 100%);
    color: #e0e7ff;
}

/* Hero Banner */
.hero-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.15) 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
    text-align: center;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1.05rem;
    margin: 0;
}

/* Metric Cards Grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-box {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}

.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(129, 140, 248, 0.6);
}

.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-lbl {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.82rem;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section Header */
.section-title {
    color: #e0e7ff;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 1.8rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(129, 140, 248, 0.4), transparent);
    margin-left: 0.5rem;
}

/* AI Card */
.ai-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    backdrop-filter: blur(10px);
}

.ai-card-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
}

.ai-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c7d2fe;
    margin-bottom: 0.6rem;
}

.ai-card-body {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #e0e7ff;
}

/* Q&A Chat elements */
.qa-user {
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(129, 140, 248, 0.4);
    border-radius: 12px 12px 2px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0 0.6rem 2rem;
    color: #e0e7ff;
}

.qa-ai {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px 12px 12px 2px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 2rem 0.6rem 0;
    color: #c7d2fe;
}

.qa-role {
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    opacity: 0.7;
    text-transform: uppercase;
}

/* Custom Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
}

[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(129, 140, 248, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Security & API Protection
# ---------------------------------------------------------------------------
def get_groq_client(user_key_input=None):
    """
    Retrieve Groq API key from Streamlit Cloud Secrets (st.secrets) or .env via os.getenv.
    Fallback to password-masked sidebar input if key is absent.
    NEVER hardcode keys or expose them in the UI.
    """
    key = ""
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GROQ_API_KEY", "").strip()

    if not key and user_key_input:
        key = user_key_input.strip()

    if not key or key.startswith("gsk_your"):
        return None

    try:
        return Groq(api_key=key)
    except Exception:
        return None


def query_groq_ai(client, prompt, system_prompt=None):
    """Query Groq API trying llama-3.3-70b-versatile with fallback handling."""
    if client is None:
        return "Groq API key not configured. Please set GROQ_API_KEY in your .env file or enter your API key in the sidebar."

    models_to_try = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "groq/compound",
    ]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for model in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as err:
            err_msg = str(err).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
                continue
            return f"Groq API error ({model}): {err}"

    return "Error: Unable to connect to Groq API. Please verify your API key."


# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(file_source):
    """Load CSV file safely into pandas DataFrame."""
    try:
        return pd.read_csv(file_source)
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        return None


def detect_columns(df):
    """Auto-detect numeric and categorical columns from the dataset."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string", "boolean"]).columns.tolist()

    # Fallback for low-cardinality numeric columns
    if not cat_cols:
        for col in df.columns:
            if df[col].nunique() <= 15:
                cat_cols.append(col)

    # Fallback for text columns convertable to numeric
    if not num_cols:
        for col in df.columns:
            if col not in cat_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                    num_cols.append(col)
                except Exception:
                    pass

    return cat_cols, num_cols


# ---------------------------------------------------------------------------
# Chart Generation (Seaborn & Matplotlib)
# ---------------------------------------------------------------------------
def render_seaborn_chart(df, chart_type, cat_col=None, num_col=None, agg_func="sum", x_col=None, y_col=None, hue_col=None, bins=20):
    """
    Generate Seaborn/Matplotlib chart, returning figure, bytes buffer, and summary string.
    """
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))

    fig.patch.set_facecolor("#161626")
    ax.set_facecolor("#1b1b2f")
    ax.tick_params(colors="#e0e7ff", labelsize=9.5)
    ax.xaxis.label.set_color("#e0e7ff")
    ax.yaxis.label.set_color("#e0e7ff")
    ax.title.set_color("#c7d2fe")

    summary_text = ""

    if chart_type == "Bar Chart":
        if not cat_col:
            ax.text(0.5, 0.5, "Please select a categorical column.", color="white", ha="center", va="center")
            summary_text = "No categorical column selected."
        else:
            if num_col and num_col != "Count of Records":
                if agg_func == "mean":
                    agg_df = df.groupby(cat_col, as_index=False)[num_col].mean()
                else:
                    agg_df = df.groupby(cat_col, as_index=False)[num_col].sum()
                val_title = f"{agg_func.capitalize()} of {num_col}"
            else:
                agg_df = df[cat_col].value_counts().reset_index()
                agg_df.columns = [cat_col, "Count"]
                val_title = "Count"

            agg_df = agg_df.sort_values(by=agg_df.columns[1], ascending=False).head(15)
            palette = sns.color_palette("mako", n_colors=len(agg_df))

            sns.barplot(data=agg_df, x=cat_col, y=agg_df.columns[1], palette=palette, ax=ax, edgecolor="none")
            ax.set_title(f"Bar Chart: {val_title} by {cat_col}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(cat_col, fontsize=10.5, labelpad=8)
            ax.set_ylabel(val_title, fontsize=10.5, labelpad=8)
            plt.xticks(rotation=35, ha="right")

            top_cat = agg_df.iloc[0][cat_col]
            top_val = agg_df.iloc[0][agg_df.columns[1]]
            summary_text = f"Highest category is '{top_cat}' with {val_title} of {top_val:,.2f}. Total categories displayed: {len(agg_df)}."

    elif chart_type == "Pie Chart":
        if not cat_col:
            ax.text(0.5, 0.5, "Please select a categorical column.", color="white", ha="center", va="center")
            summary_text = "No categorical column selected."
        else:
            if num_col and num_col != "Count of Records":
                agg_s = df.groupby(cat_col)[num_col].sum()
                val_title = f"Sum of {num_col}"
            else:
                agg_s = df[cat_col].value_counts()
                val_title = "Record Count"

            if len(agg_s) > 8:
                top7 = agg_s.nlargest(7)
                other_val = agg_s.iloc[7:].sum()
                top7["Other"] = other_val
                agg_s = top7

            colors = sns.color_palette("pastel", len(agg_s))
            wedges, texts, autotexts = ax.pie(
                agg_s.values,
                labels=agg_s.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                textprops=dict(color="#e0e7ff", fontsize=9.5),
            )
            for autotext in autotexts:
                autotext.set_color("#1e1b4b")
                autotext.set_weight("bold")

            ax.set_title(f"Pie Chart: {val_title} Distribution by {cat_col}", fontsize=13, fontweight="bold", pad=12)
            top_cat = agg_s.idxmax()
            top_pct = (agg_s.max() / agg_s.sum()) * 100
            summary_text = f"Top slice is '{top_cat}' accounting for {top_pct:.1f}% of total {val_title} across {len(agg_s)} segments."

    elif chart_type == "Line Chart":
        if not x_col or not y_col:
            ax.text(0.5, 0.5, "Please select X and Y columns.", color="white", ha="center", va="center")
            summary_text = "Missing X or Y columns."
        else:
            sns.lineplot(data=df, x=x_col, y=y_col, marker="o", color="#818cf8", linewidth=2.2, ax=ax)
            ax.set_title(f"Line Chart: {y_col} vs {x_col}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(x_col, fontsize=10.5, labelpad=8)
            ax.set_ylabel(y_col, fontsize=10.5, labelpad=8)
            if df[x_col].dtype == "object":
                plt.xticks(rotation=35, ha="right")

            min_y, max_y = df[y_col].min(), df[y_col].max()
            summary_text = f"Line chart shows progression of {y_col} over {x_col}. Values range between {min_y:,.2f} and {max_y:,.2f} across {len(df)} records."

    elif chart_type == "Histogram":
        if not num_col:
            ax.text(0.5, 0.5, "Please select a numeric column.", color="white", ha="center", va="center")
            summary_text = "No numeric column selected."
        else:
            sns.histplot(data=df, x=num_col, bins=bins, kde=True, color="#818cf8", edgecolor="#312e81", ax=ax)
            ax.set_title(f"Histogram: Distribution of {num_col}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(num_col, fontsize=10.5, labelpad=8)
            ax.set_ylabel("Frequency", fontsize=10.5, labelpad=8)

            mean_val = df[num_col].mean()
            median_val = df[num_col].median()
            summary_text = f"Histogram displays distribution of {num_col} across {bins} bins. Mean value is {mean_val:,.2f} and median is {median_val:,.2f}."

    elif chart_type == "Scatter Plot":
        if not x_col or not y_col:
            ax.text(0.5, 0.5, "Please select X and Y numeric columns.", color="white", ha="center", va="center")
            summary_text = "Missing X or Y columns."
        else:
            hue = hue_col if (hue_col and hue_col != "None") else None
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette="viridis" if hue else None, s=60, alpha=0.85, ax=ax)
            ax.set_title(f"Scatter Plot: {y_col} vs {x_col}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(x_col, fontsize=10.5, labelpad=8)
            ax.set_ylabel(y_col, fontsize=10.5, labelpad=8)
            if hue:
                ax.legend(title=hue, facecolor="#1b1b2f", edgecolor="none", labelcolor="#e0e7ff")

            corr = df[[x_col, y_col]].corr().iloc[0, 1] if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]) else 0
            summary_text = f"Scatter plot correlates {y_col} against {x_col} across {len(df)} data points. Calculated correlation coefficient is {corr:.2f}."

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return fig, buf, summary_text


# ---------------------------------------------------------------------------
# Sidebar - Security & CSV Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 0.8rem 0;'>
            <span style='font-size: 2.4rem;'>📊</span><br>
            <span style='font-size: 1.15rem; font-weight: 700; color: #818cf8;'>AI Data Assistant</span><br>
            <span style='font-size: 0.75rem; color: rgba(255,255,255,0.5);'>Groq API • Seaborn Charts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Security check: load key from st.secrets or environment
    env_key = ""
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            env_key = str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        env_key = ""

    if not env_key:
        env_key = os.getenv("GROQ_API_KEY", "").strip()

    user_key_input = None

    if not env_key or env_key.startswith("gsk_your"):
        user_key_input = st.sidebar.text_input(
            "Enter Groq API Key",
            type="password",
            help="Password-masked fallback key input (never stored or logged).",
        )

    groq_client = get_groq_client(user_key_input=user_key_input)

    if groq_client:
        st.success("Groq API Connected", icon="✅")
    else:
        st.info("Groq API Key not set. Enter key above or add to Streamlit Secrets.", icon="🔑")

    st.divider()
    st.subheader("Data Loading & Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    st.caption("If no file is uploaded, defaults to `dataset.csv`.")
    st.divider()

    st.markdown("<small style='color: rgba(255,255,255,0.4);'>AI Data Analysis Assistant v2.0</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data (Default: dataset.csv)
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    data_source = uploaded_file.name
else:
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    if os.path.exists(default_path):
        df = load_csv(default_path)
        data_source = "dataset.csv (Default)"
    else:
        df = None
        st.error("dataset.csv not found. Please upload a CSV file.")

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI Data Analysis Assistant</div>
        <div class="hero-subtitle">Dynamic CSV Insights &nbsp;•&nbsp; Multi-Graph Visualizations &nbsp;•&nbsp; Groq AI Analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None:
    st.stop()

# ---------------------------------------------------------------------------
# Data Summary & Statistics
# ---------------------------------------------------------------------------
cat_cols, num_cols = detect_columns(df)
total_rows, total_cols = df.shape
missing_count = int(df.isnull().sum().sum())

st.markdown("<div class='section-title'>Dataset Overview & Summary Statistics</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val">{total_rows:,}</div>
            <div class="metric-lbl">Total Rows</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{total_cols}</div>
            <div class="metric-lbl">Total Columns</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{missing_count}</div>
            <div class="metric-lbl">Missing Values</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{len(cat_cols)} / {len(num_cols)}</div>
            <div class="metric-lbl">Categorical / Numeric</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🔍 Interactive Data Preview & Data Types", expanded=True):
    col_tbl, col_info = st.columns([3, 1])
    with col_tbl:
        st.dataframe(df, use_container_width=True, height=270)
    with col_info:
        st.markdown(f"**Data Source:** `{data_source}`")
        st.markdown("**Column Types:**")
        for col, dtype in df.dtypes.items():
            ctype = "Numeric" if col in num_cols else "Categorical"
            st.markdown(f"- `{col}` <small>({ctype}, {dtype})</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dynamic Multi-Graph Visualizations
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Dynamic Multi-Graph Visualizations</div>", unsafe_allow_html=True)

col_chart_select, _ = st.columns([2, 1])
with col_chart_select:
    chart_type = st.selectbox(
        "Select Graph Type",
        ["Bar Chart", "Pie Chart", "Line Chart", "Histogram", "Scatter Plot"],
        index=0,
    )

c_cat, c_num, c_opt = st.columns(3)

sel_cat, sel_num, sel_x, sel_y, sel_hue = None, None, None, None, None
agg_func = "sum"
bins = 20
all_cols = df.columns.tolist()

if chart_type in ["Bar Chart", "Pie Chart"]:
    with c_cat:
        cat_opts = cat_cols if cat_cols else all_cols
        sel_cat = st.selectbox("Categorical Column", cat_opts, index=0)
    with c_num:
        num_opts = ["Count of Records"] + num_cols
        sel_num = st.selectbox("Values Column", num_opts, index=0 if len(num_opts) > 1 else 0)
    if chart_type == "Bar Chart" and sel_num != "Count of Records":
        with c_opt:
            agg_func = st.selectbox("Aggregation", ["sum", "mean"], index=0)

elif chart_type == "Line Chart":
    with c_cat:
        sel_x = st.selectbox("X-Axis Column", all_cols, index=0)
    with c_num:
        y_opts = num_cols if num_cols else all_cols
        sel_y = st.selectbox("Y-Axis Column (Numeric)", y_opts, index=0)

elif chart_type == "Histogram":
    with c_num:
        h_opts = num_cols if num_cols else all_cols
        sel_num = st.selectbox("Numeric Column", h_opts, index=0)
    with c_opt:
        bins = st.slider("Bins Count", min_value=5, max_value=50, value=20, step=5)

elif chart_type == "Scatter Plot":
    with c_cat:
        sx_opts = num_cols if num_cols else all_cols
        sel_x = st.selectbox("X-Axis (Numeric)", sx_opts, index=0)
    with c_num:
        sy_opts = [c for c in num_cols if c != sel_x] if len(num_cols) > 1 else (num_cols if num_cols else all_cols)
        sel_y = st.selectbox("Y-Axis (Numeric)", sy_opts, index=0)
    with c_opt:
        hue_opts = ["None"] + cat_cols + num_cols
        sel_hue = st.selectbox("Group Color (Hue)", hue_opts, index=0)

# Render Seaborn Chart
fig, img_buf, summary_text = render_seaborn_chart(
    df=df,
    chart_type=chart_type,
    cat_col=sel_cat,
    num_col=sel_num,
    agg_func=agg_func,
    x_col=sel_x,
    y_col=sel_y,
    hue_col=sel_hue,
    bins=bins,
)

chart_col, ai_col = st.columns([3, 2])

with chart_col:
    st.image(img_buf, use_container_width=True)
    st.download_button(
        label="📥 Export Chart as PNG",
        data=img_buf.getvalue(),
        file_name=f"{chart_type.lower().replace(' ', '_')}.png",
        mime="image/png",
        use_container_width=True,
    )

with ai_col:
    sys_prompt = "You are a data analyst. Provide a concise, 2-sentence explanation of what the chart illustrates based on the summary context."
    user_prompt = (
        f"Data Source: '{data_source}' ({total_rows} rows, {total_cols} columns).\n"
        f"Chart Type: {chart_type}\n"
        f"Chart Summary: {summary_text}\n"
        "Provide a clear, 2-sentence explanation of the key finding shown in this chart."
    )

    with st.spinner("Generating AI explanation via Groq (llama-3.3-70b-versatile)..."):
        ai_exp = query_groq_ai(groq_client, user_prompt, system_prompt=sys_prompt)

    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-card-badge">GROQ AI INSIGHT</div>
            <div class="ai-card-title">{chart_type} Analysis</div>
            <div class="ai-card-body">{ai_exp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# AI CSV Q&A Assistant
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Interactive AI Q&A Assistant</div>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:
    st.markdown(f'<div class="qa-user"><div class="qa-role">You</div>{item["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="qa-ai"><div class="qa-role">Groq AI (llama-3.3-70b-versatile)</div>{item["answer"]}</div>', unsafe_allow_html=True)

with st.form("qa_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_q = st.text_input("Ask any question about your CSV data:", placeholder="e.g. What is the total sales or average age in the dataset?", label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("Ask AI", use_container_width=True)

if submitted and user_q.strip():
    desc_summary = df.describe(include="all").round(2).to_string() if not df.empty else ""
    head_summary = df.head(5).to_string()

    qa_sys_prompt = (
        f"You are an expert AI data assistant inspecting '{data_source}'. "
        f"Dataset shape: {total_rows} rows x {total_cols} columns. "
        f"Columns: {', '.join(df.columns)}. "
        f"Summary stats:\n{desc_summary[:1500]}\n"
        f"Sample rows:\n{head_summary}\n"
        "Answer the user's question accurately, concisely, and professionally based strictly on this dataset."
    )

    with st.spinner("Analyzing dataset & formulating AI answer..."):
        answer = query_groq_ai(groq_client, user_q.strip(), system_prompt=qa_sys_prompt)

    st.session_state.chat_history.append({"question": user_q.strip(), "answer": answer})
    st.rerun()

if st.session_state.chat_history:
    if st.button("Clear Q&A History"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem; padding: 2rem 0 1rem 0;'>
        AI Data Analysis Assistant • Built with Streamlit, Seaborn & Groq (llama-3.3-70b-versatile)
    </div>
    """,
    unsafe_allow_html=True,
)
