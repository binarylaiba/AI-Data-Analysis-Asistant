<div align="center">

# 📊 AI Data Analysis Assistant

### *Intelligent CSV Analysis powered by Groq LLaMA 3.3 & Streamlit*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F54F2B?style=for-the-badge&logo=meta&logoColor=white)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> Upload any CSV, get instant statistics, beautiful charts, and AI-powered answers — all in one sleek dark-themed web app.

![App Preview](https://img.shields.io/badge/Status-Live%20%26%20Ready-brightgreen?style=flat-square)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **CSV Upload** | Upload any CSV or use the built-in retail dataset |
| 📈 **Auto Visualizations** | Bar, Pie, Line, Histogram & Scatter charts via Plotly + Seaborn |
| 🤖 **AI Q&A Chat** | Ask anything about your data — powered by Groq LLaMA 3.3 70B |
| 📊 **Smart Statistics** | Min, max, mean, std, quartiles, frequency distributions |
| 🔒 **Secure API Handling** | Keys loaded automatically from `.env` / Streamlit Secrets — no manual input needed |
| 🌙 **Dark Glassmorphism UI** | Premium dark-themed Streamlit interface |

---

## 🗂️ Project Structure

```
📦 AI Data Analysis Assistant/
├── 📄 app.py               → Streamlit web interface (main app)
├── 📄 main.py              → CLI pipeline (terminal mode)
├── 📄 analysis.py          → Data loading + statistics + Q&A logic
├── 📄 visualization.py     → Seaborn/Matplotlib chart generator
├── 📄 dataset.csv          → 50-row realistic retail dataset
├── 📄 requirements.txt     → Python dependencies
├── 📄 .env.example         → API key template (safe to commit)
├── 📄 .env                 → Your actual API key (NEVER commit this)
├── 📁 charts/              → Auto-generated PNG charts saved here
└── 📄 README.md            → You are here!
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/binarylaiba/AI-Data-Analysis-Asistant.git
cd AI-Data-Analysis-Asistant
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Your Groq API Key

```bash
# Copy the example file
copy .env.example .env
```

Open `.env` and add your key:

```env
GROQ_API_KEY=gsk_your_real_key_here
```

🔑 Get a **free** Groq API key at → https://console.groq.com/keys

> **Note:** The app works without a key — AI chat is simply skipped gracefully.

---

## ▶️ Running the App

### 🌐 Streamlit Web App *(Recommended)*

```bash
py -m streamlit run app.py
```

Opens at **http://localhost:8501**

**What you get:**
- 📌 **Sidebar** — Upload any CSV or use default `dataset.csv`
- 📊 **Metric Cards** — Total records, total sales, avg age, top category
- 🔍 **Dataset Preview** — First 10 rows + column types + missing values
- ❓ **Competition Q&A** — Styled answers to all 3 analysis questions
- 📉 **Interactive Plotly Chart** — Dark-themed, hover-enabled charts
- 🖼️ **Chart Export** — Saved PNG preview (Seaborn/Matplotlib)
- 💬 **AI Chat** — Ask anything about your data in real time

### 💻 CLI Pipeline *(Terminal Mode)*

```bash
py main.py dataset.csv
```

**Steps it runs:**
1. Loads & inspects the CSV
2. Prints full descriptive statistics
3. Answers the 3 competition questions
4. Generates `charts/sales_by_category.png`
5. Sends chart to Groq for a 2-sentence AI explanation
6. Launches **interactive Q&A** — type questions, get instant answers. Type `exit` to quit.

---

## 📋 Dataset Overview

`dataset.csv` — 50 rows of realistic retail transactions:

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | int | Unique order ID (1001–1050) |
| `Product` | string | Product name |
| `Category` | string | Electronics / Furniture / Clothing / Home & Kitchen |
| `Sales` | float | Revenue per order (USD) |
| `Age` | int | Customer age |
| `City` | string | US city |

---

## 🏆 Competition Q&A Results

| # | Question | Answer |
|---|----------|--------|
| 1 | Which product has the **highest total sales**? | 🥇 **Laptop** |
| 2 | What is the **average customer age**? | 📅 **37.14 years** |
| 3 | Which **product category** appears most frequently? | 🏷️ **Electronics** |

> Exact figures are computed at runtime directly from `dataset.csv`.

---

## 📦 Dependencies

```txt
pandas
numpy
matplotlib
seaborn
groq
streamlit
python-dotenv
plotly
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## ✅ Deliverables Checklist

- [x] `dataset.csv` — 50-row retail dataset with required headers
- [x] `analysis.py` — `load_data`, `analyze_data`, `answer_questions`
- [x] `visualization.py` — `generate_chart` (Seaborn/Matplotlib PNG)
- [x] `main.py` — CLI pipeline + interactive Q&A (Groq)
- [x] `app.py` — Streamlit web interface with all required panels
- [x] `requirements.txt` — all dependencies listed
- [x] `.env.example` — placeholder API key (safe for version control)
- [x] `README.md` — this file

---

## 🔐 Security

- API key is loaded **automatically** from `.env` or Streamlit Cloud Secrets
- No manual API key input in the UI — clean & secure by design
- `.env` is listed in `.gitignore` — your key is **never** committed

---

## 📝 Notes

- AI model used: **`llama-3.3-70b-versatile`** via Groq (with fallback models)
- All Python files are commented with clean section dividers
- Charts are auto-saved to `charts/sales_by_category.png`
- Streamlit app and CLI pipeline run **independently** of each other

---

<div align="center">

Made with ❤️ by **Laiba** | Powered by **Groq** + **Streamlit**

</div>
