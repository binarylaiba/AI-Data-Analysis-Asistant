# AI Data Analysis Assistant

A complete Python project that loads a retail CSV dataset, performs
statistical analysis, generates visualisations, and answers questions
via a Groq-powered LLM (llama-3.3-70b-versatile).

---

## Project Structure

```
Ai Data Analysis Assistant/
|-- dataset.csv          # 50-row realistic retail dataset
|-- analysis.py          # Data loading + descriptive statistics + Q&A
|-- visualization.py     # Seaborn/Matplotlib bar chart generator
|-- main.py              # CLI pipeline (all steps + interactive Q&A)
|-- app.py               # Streamlit web interface
|-- requirements.txt     # Python dependencies
|-- .env                 # Your API key (never commit this)
|-- .env.example         # Placeholder for version control
|-- charts/              # Auto-created; stores generated PNG charts
|-- README.md            # This file
```

---

## Setup

### 1. Clone / download the project

```bash
# or just open the folder in VS Code / your editor
cd "Ai Data Analysis Assistant"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Groq API key

Copy `.env.example` to `.env` and replace the placeholder:

```bash
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_real_key_here
```

Get a free key at <https://console.groq.com/keys>.

> The project works without a key -- AI explanation and chat are
> simply skipped.

---

## Execution

### CLI Pipeline

Runs the full analysis pipeline in your terminal:

```bash
py main.py dataset.csv
```

**What it does:**

1. Loads and inspects `dataset.csv`
2. Prints descriptive statistics (min, max, mean, std, quartiles,
   frequency distributions)
3. Answers the 3 competition questions:
   - Which product has the highest total sales?
   - What is the average customer age?
   - Which product category appears most frequently?
4. Generates `charts/sales_by_category.png`
5. Sends chart findings to Groq for a 2-sentence AI explanation
6. Starts an **interactive CLI Q&A** session -- type any question
   about the dataset; Groq answers in real time. Type `exit` to quit.

### Streamlit Web App

```bash
py -m streamlit run app.py
```

Opens at <http://localhost:8501> with:

- **Sidebar** -- upload any CSV or use the default `dataset.csv`
- **Metric cards** -- total records, total sales, avg age, top category
- **Dataset preview** -- first 10 rows + column types + missing-value check
- **Competition Q&A callouts** -- styled answers to all 3 questions
- **Interactive Plotly chart** -- sales by category (dark-themed)
- **Saved PNG preview** -- shows the Seaborn/Matplotlib chart if already generated
- **AI Chat** -- powered by Groq llama-3.3-70b-versatile; ask anything about
  your data in a sleek chat interface

---

## Dataset

`dataset.csv` -- 50 rows of realistic retail transactions:

| Column   | Type    | Description                    |
|----------|---------|-------------------------------|
| Order_ID | int     | Unique order identifier (1001-1050) |
| Product  | string  | Product name                  |
| Category | string  | Electronics / Furniture / Clothing / Home & Kitchen |
| Sales    | float   | Revenue for this order (USD)  |
| Age      | int     | Customer age                  |
| City     | string  | US city                       |

---

## Competition Q&A Answers

| # | Question                                       | Answer         |
|---|------------------------------------------------|----------------|
| 1 | Which product has the highest total sales?     | Laptop         |
| 2 | What is the average customer age?              | 37.14 years    |
| 3 | Which product category appears most frequently?| Electronics    |

> Exact figures are computed at runtime from `dataset.csv`.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
groq
streamlit
python-dotenv
plotly
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Deliverables Checklist

- [x] `dataset.csv` -- 50-row retail dataset with required headers
- [x] `analysis.py` -- `load_data`, `analyze_data`, `answer_questions`
- [x] `visualization.py` -- `generate_chart` (Seaborn/Matplotlib PNG)
- [x] `main.py` -- CLI pipeline + interactive Q&A (Groq)
- [x] `app.py` -- Streamlit web interface with all required panels
- [x] `requirements.txt` -- all dependencies listed
- [x] `.env.example` -- placeholder API key
- [x] `README.md` -- this file

---

## Notes

- The Groq model used is **llama-3.3-70b-versatile** (as specified).
- All Python files are commented and use clean dividers for readability.
- Charts are auto-saved to `charts/sales_by_category.png` (directory
  is created automatically).
- The Streamlit app and CLI can be run independently of each other.
