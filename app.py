"""
app.py
------
Modern Streamlit web interface for the AI Data Analysis Assistant.

Features: CSV upload, metric cards, Q&A callouts, Plotly chart,
saved PNG preview, AI chat via Groq openai/gpt-oss-120b.

Run:  py -m streamlit run app.py
"""

import os
import warnings
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS -- dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.25) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px; padding: 2.2rem 2.5rem; margin-bottom: 2rem;
    backdrop-filter: blur(14px); text-align: center;
}
.hero-banner h1 {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-banner p { color: rgba(255,255,255,0.65); font-size: 1.05rem; margin: 0; }
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    backdrop-filter: blur(10px); text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(129,140,248,0.6); }
.metric-icon  { font-size: 1.8rem; margin-bottom: 0.3rem; }
.metric-value {
    font-size: 1.75rem; font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { color: rgba(255,255,255,0.55); font-size: 0.82rem; margin-top: 0.25rem; }
.qa-callout {
    background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(168,85,247,0.12) 100%);
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0; padding: 1rem 1.4rem;
    margin-bottom: 0.85rem; backdrop-filter: blur(8px);
}
.qa-question { color: rgba(255,255,255,0.55); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.07em; }
.qa-answer   { color: #e0e7ff; font-size: 1.12rem; font-weight: 600; margin-top: 0.25rem; }
.qa-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white; font-size: 0.7rem; font-weight: 700;
    padding: 0.2rem 0.6rem; border-radius: 999px;
    margin-bottom: 0.4rem; letter-spacing: 0.06em;
}
.section-header {
    color: #e0e7ff; font-size: 1.22rem; font-weight: 700;
    margin: 1.5rem 0 0.9rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(129,140,248,0.5), transparent);
    margin-left: 0.5rem;
}
.chat-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(168,85,247,0.25));
    border: 1px solid rgba(129,140,248,0.4);
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0 0.5rem 2rem; color: #e0e7ff;
}
.chat-ai {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.13);
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 2rem 0.5rem 0; color: #c7d2fe;
}
.chat-label { font-size: 0.72rem; font-weight: 600; margin-bottom: 0.3rem; opacity: 0.6; text-transform: uppercase; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.9) !important;
    border-right: 1px solid rgba(129,140,248,0.2) !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.5); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
load_dotenv()


@st.cache_resource
def get_groq_client():
    """Return an authenticated Groq client, or None if key is absent.
    
    Checks in order:
    1. Streamlit Cloud secrets (st.secrets)
    2. Local .env file (os.getenv)
    """
    # 1. Try Streamlit Cloud secrets first
    key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    # 2. Fall back to local .env
    if not key:
        key = os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_your"):
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None


@st.cache_data
def load_dataframe(source):
    """Load a DataFrame from a file path or uploaded file object."""
    try:
        return pd.read_csv(source)
    except Exception as err:
        st.error(f"Failed to read CSV: {err}")
        return None


def compute_metrics(df):
    """Derive key metrics from the retail dataset."""
    m = {
        "total_records": len(df), "total_sales": 0.0,
        "avg_age": 0.0, "top_product": "N/A",
        "top_category": "N/A", "top_product_sales": 0.0,
    }
    if "Sales" in df.columns and "Product" in df.columns:
        ps = df.groupby("Product")["Sales"].sum()
        m["total_sales"]       = df["Sales"].sum()
        m["top_product"]       = ps.idxmax()
        m["top_product_sales"] = ps.max()
    if "Age" in df.columns:
        m["avg_age"] = round(df["Age"].mean(), 2)
    if "Category" in df.columns:
        m["top_category"] = df["Category"].mode()[0]
    return m


def ask_groq(client, df, question):
    """Send a question to Groq with the dataset as context."""
    rows, cols   = df.shape
    col_list     = ", ".join(df.columns.tolist())
    numeric_desc = df.describe().round(2).to_string()
    cat_summary  = ""
    for col in df.select_dtypes(include=["object", "category"]).columns:
        top5 = df[col].value_counts().head(5).to_dict()
        cat_summary += f"  {col}: {top5}"
    system_msg = (
        f"You are a helpful data analyst. Dataset: {rows} rows, columns: {col_list}. "
        f"Numeric summary: {numeric_desc}. Top categorical: {cat_summary}. "
        "Answer concisely and accurately."
    )
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": question}],
            temperature=0.4, max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as err:
        return f"Groq API error: {err}"


def make_plotly_bar(df):
    """Create a styled Plotly horizontal bar chart of sales by category."""
    if "Category" not in df.columns or "Sales" not in df.columns:
        return None
    agg = (
        df.groupby("Category")["Sales"].sum()
        .sort_values(ascending=True).reset_index()
        .rename(columns={"Sales": "Total Sales"})
    )
    fig = px.bar(
        agg, x="Total Sales", y="Category", orientation="h",
        color="Total Sales",
        color_continuous_scale=["#6366f1", "#a855f7", "#ec4899"],
        text=agg["Total Sales"].apply(lambda v: f"${v:,.0f}"),
        labels={"Total Sales": "Total Sales (USD)", "Category": ""},
        title="Total Sales by Product Category",
    )
    fig.update_traces(
        textposition="outside",
        marker_line_color="rgba(255,255,255,0.15)",
        marker_line_width=1,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e7ff", family="Inter"),
        title_font=dict(size=17, color="#c7d2fe"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                   tickprefix="$", tickformat=",.0f", color="#94a3b8"),
        yaxis=dict(color="#e0e7ff"),
        margin=dict(l=0, r=20, t=50, b=20), height=360,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0'>"
        "<span style='font-size:2.5rem'>&#x1F4CA;</span><br>"
        "<span style='font-size:1.1rem;font-weight:700;color:#818cf8'>AI Data Assistant</span><br>"
        "<span style='font-size:0.75rem;color:rgba(255,255,255,0.45)'>Powered by Groq + LLaMA 3.3</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Data Source**")
    upload      = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    use_default = st.checkbox("Use dataset.csv (default)", value=(upload is None))
    st.divider()
    groq_client = get_groq_client()
    if groq_client:
        st.success("Groq API connected")
    else:
        st.warning("Groq API key missing\nAdd GROQ_API_KEY to .env (local) or Streamlit Secrets (cloud)")
    st.divider()
    st.markdown(
        "<small style='color:rgba(255,255,255,0.35)'>AI Data Analysis Assistant v1.0<br>Competition Edition 2026</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = None
if upload is not None:
    df = load_dataframe(upload)
elif use_default:
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
    if os.path.exists(default_path):
        df = load_dataframe(default_path)
    else:
        st.warning("dataset.csv not found. Please upload a CSV file.")

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='hero-banner'><h1>AI Data Analysis Assistant</h1>"
    "<p>Upload a retail CSV &nbsp;&#183;&nbsp; Explore insights &nbsp;&#183;&nbsp; Chat with your data via Groq AI</p></div>",
    unsafe_allow_html=True,
)

if df is None:
    st.info("Upload a CSV or enable 'Use dataset.csv' in the sidebar to begin.")
    st.stop()

metrics = compute_metrics(df)

# ---- Metric cards ----------------------------------------------------------
st.markdown(
    f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-icon">&#x1F4E6;</div>
            <div class="metric-value">{metrics['total_records']:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">&#x1F4B0;</div>
            <div class="metric-value">${metrics['total_sales']:,.0f}</div>
            <div class="metric-label">Total Sales</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">&#x1F382;</div>
            <div class="metric-value">{metrics['avg_age']}</div>
            <div class="metric-label">Avg. Customer Age</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">&#x1F3C6;</div>
            <div class="metric-value" style="font-size:1.2rem">{metrics['top_category']}</div>
            <div class="metric-label">Top Category</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ---- Dataset preview -------------------------------------------------------
st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)
col_l, col_r = st.columns([3, 1])
with col_l:
    st.dataframe(df.head(10), use_container_width=True, height=280)
with col_r:
    st.markdown(f"**Shape:** {df.shape[0]} x {df.shape[1]}")
    st.markdown("**Columns:**")
    for c, d in df.dtypes.items():
        st.markdown(f"- `{c}` - *{d}*")
    miss = df.isnull().sum().sum()
    st.success("No missing values") if miss == 0 else st.warning(f"{miss} missing cell(s)")

# ---- Q&A callouts ----------------------------------------------------------
st.markdown("<div class='section-header'>Competition Q&amp;A - 3 Key Questions</div>", unsafe_allow_html=True)

q1_product, q1_sales, q2_age, q3_cat, q3_freq = "N/A", 0.0, "N/A", "N/A", 0
if "Product" in df.columns and "Sales" in df.columns:
    ps = df.groupby("Product")["Sales"].sum()
    q1_product, q1_sales = ps.idxmax(), ps.max()
if "Age" in df.columns:
    q2_age = round(df["Age"].mean(), 2)
if "Category" in df.columns:
    q3_cat  = df["Category"].mode()[0]
    q3_freq = int((df["Category"] == q3_cat).sum())

st.markdown(
    f"""
    <div class="qa-callout">
        <div class="qa-badge">Q1</div>
        <div class="qa-question">Which product has the highest total sales?</div>
        <div class="qa-answer">{q1_product} &nbsp;&#8212;&nbsp; <span style="color:#a5b4fc;">${q1_sales:,.2f}</span></div>
    </div>
    <div class="qa-callout">
        <div class="qa-badge">Q2</div>
        <div class="qa-question">What is the average customer age?</div>
        <div class="qa-answer">{q2_age} years old</div>
    </div>
    <div class="qa-callout">
        <div class="qa-badge">Q3</div>
        <div class="qa-question">Which product category appears most frequently?</div>
        <div class="qa-answer">{q3_cat} &nbsp;&#8212;&nbsp; <span style="color:#a5b4fc;">{q3_freq} orders</span></div>
    </div>""",
    unsafe_allow_html=True,
)

# ---- Chart -----------------------------------------------------------------
st.markdown("<div class='section-header'>Sales by Category</div>", unsafe_allow_html=True)
chart_tab, saved_tab = st.tabs(["Interactive Chart", "Saved PNG Preview"])

with chart_tab:
    fig = make_plotly_bar(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Chart requires 'Category' and 'Sales' columns.")

with saved_tab:
    saved_png = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "charts", "sales_by_category.png"
    )
    if os.path.exists(saved_png):
        st.image(saved_png, caption="Generated by visualization.py - Seaborn/Matplotlib", use_container_width=True)
    else:
        st.info("No saved chart found. Run  py main.py dataset.csv  to generate one.")

# ---- AI Chat ---------------------------------------------------------------
st.markdown("<div class='section-header'>AI Chat - Ask Anything About Your Data</div>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for turn in st.session_state.chat_history:
    st.markdown(
        f'<div class="chat-user"><div class="chat-label">You</div>{turn["question"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="chat-ai"><div class="chat-label">AI - GPT-OSS 120B</div>{turn["answer"]}</div>',
        unsafe_allow_html=True,
    )

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_q = st.text_input(
            "Ask", placeholder="e.g. Which city has the highest average sales?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send", use_container_width=True)

if submitted and user_q.strip():
    if groq_client is None:
        answer = "Groq API key not configured. Add GROQ_API_KEY to your .env file."
    else:
        with st.spinner("Thinking..."):
            answer = ask_groq(groq_client, df, user_q.strip())
    st.session_state.chat_history.append({"question": user_q.strip(), "answer": answer})
    st.rerun()

if st.session_state.chat_history:
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---- Footer ----------------------------------------------------------------
st.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.78rem;padding:1rem 0'>"
    "AI Data Analysis Assistant - Built with Streamlit + Groq + GPT-OSS 120B - 2026"
    "</div>",
    unsafe_allow_html=True,
)
