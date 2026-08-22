"""
main.py
-------
End-to-end CLI pipeline for the AI Data Analysis Assistant.

Pipeline
--------
1. Load dataset                    (analysis.load_data)
2. Compute descriptive statistics  (analysis.analyze_data)
3. Answer 3 evaluation questions   (analysis.answer_questions)
4. Generate sales chart            (visualization.generate_chart)
5. AI natural-language explanation (Groq  openai/gpt-oss-120b)
6. Interactive CLI — answer ad-hoc user / judge questions via Groq

Usage
-----
    py main.py <path_to_csv>
    py main.py dataset.csv
"""

import os
import sys
import warnings

# Suppress pandas / seaborn FutureWarnings that clutter the terminal
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from groq import Groq

from analysis import load_data, analyze_data, answer_questions
from visualization import generate_chart


# ===========================================================================
# Console formatting helpers
# ===========================================================================

WIDTH = 65


def _banner(title: str) -> None:
    """Print a bold section header."""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def _divider() -> None:
    print("-" * WIDTH)


# ===========================================================================
# Step 5 — AI chart explanation via Groq
# ===========================================================================

def generate_ai_explanation(chart_summary: dict, client: Groq) -> str:
    """
    Call the Groq API (openai/gpt-oss-120b) to produce a clear,
    natural 2-sentence summary of the chart findings.

    Parameters
    ----------
    chart_summary : dict
        Output of visualization.generate_chart — must contain:
            top_category, top_share_pct, total_sales
    client : groq.Groq
        Authenticated Groq client.

    Returns
    -------
    str  — the AI-generated explanation, or an error message.
    """
    top_cat   = chart_summary.get("top_category",  "Unknown")
    top_share = chart_summary.get("top_share_pct", 0)
    total     = chart_summary.get("total_sales",   0)

    prompt = (
        f"You are a concise data analyst. "
        f"Based on a retail sales chart, the top-performing product category is "
        f"'{top_cat}', which accounts for {top_share}% of the total sales "
        f"of ${total:,.2f}. "
        f"Write exactly 2 clear, professional sentences summarising this "
        f"finding and what it might mean for the business. "
        f"Do not use bullet points or lists."
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as err:
        return f"[Groq API error: {err}]"


# ===========================================================================
# Step 6 — Interactive CLI Q&A session
# ===========================================================================

def interactive_qa_session(df, client) -> None:
    """
    Start an interactive loop where the user (or a judge) can ask
    free-form questions about the dataset. Groq LLM answers each
    question using the full dataset summary as context.

    Type 'exit' or 'quit' to end the session.

    Parameters
    ----------
    df     : pd.DataFrame — the loaded dataset
    client : groq.Groq   — authenticated Groq client (may be None)
    """
    _banner("INTERACTIVE Q&A  (type 'exit' to quit)")

    if client is None:
        print("\n  [WARN]  Groq client not available.")
        print("  Set GROQ_API_KEY in your .env file to enable AI answers.")
        print()
        return

    # Build a compact dataset summary to inject into every system prompt
    rows, cols   = df.shape
    col_list     = ", ".join(df.columns.tolist())
    numeric_desc = df.describe().to_string()
    cat_summary  = ""
    for col in df.select_dtypes(include=["object", "category"]).columns:
        top5 = df[col].value_counts().head(5).to_dict()
        cat_summary += f"\n  {col}: {top5}"

    system_ctx = (
        f"You are an expert data analyst. "
        f"You have access to a retail sales dataset with {rows} rows and "
        f"{cols} columns: {col_list}.\n\n"
        f"Numeric summary:\n{numeric_desc}\n\n"
        f"Top categorical values:{cat_summary}\n\n"
        f"Answer the user's questions concisely and accurately based on this data."
    )

    print("\n  Ask anything about the dataset. Type 'exit' to stop.\n")

    while True:
        try:
            user_input = input("  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  [Session ended]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("\n  [Session ended]")
            break

        # Call Groq with conversation context
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_ctx},
                    {"role": "user",   "content": user_input},
                ],
                temperature=0.4,
                max_tokens=400,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as err:
            answer = f"[Error calling Groq API: {err}]"

        print(f"\n  AI  > {answer}\n")

    print()


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(csv_path: str) -> None:
    """Execute the full analysis pipeline for the given CSV file."""

    # -- 0. Environment & Groq client ----------------------------------------
    _banner("AI Data Analysis Assistant  —  CLI Pipeline")
    print(f"\n  Dataset  :  {csv_path}")

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print(
            "\n  [WARN]  GROQ_API_KEY not found in .env — "
            "AI explanation & interactive Q&A will be skipped.\n"
        )
        groq_client = None
    else:
        groq_client = Groq(api_key=api_key)
        print("  Groq     :  client initialised (openai/gpt-oss-120b)")

    # -- 1. Load data --------------------------------------------------------
    df = load_data(csv_path)
    if df is None:
        print("\n  Pipeline aborted — could not load dataset.")
        sys.exit(1)

    # -- 2. Descriptive statistics -------------------------------------------
    analyze_data(df)

    # -- 3. Answer evaluation questions -------------------------------------
    qa_results = answer_questions(df)

    # -- 4. Generate chart --------------------------------------------------
    chart_out     = "charts/sales_by_category.png"
    chart_summary = generate_chart(df, output_path=chart_out)

    # -- 5. AI explanation --------------------------------------------------
    _banner("AI EXPLANATION  (Groq · openai/gpt-oss-120b)")

    if groq_client and chart_summary:
        explanation = generate_ai_explanation(chart_summary, groq_client)
        print()
        print(f"  {explanation}")
    elif not groq_client:
        print("\n  Skipped — no GROQ_API_KEY configured.")
    else:
        print("\n  Skipped — chart generation did not produce a summary.")

    # -- Final summary banner -----------------------------------------------
    _banner("PIPELINE COMPLETE")
    _divider()
    print("  Competition Q&A Results:")
    for k, v in qa_results.items():
        print(f"    {k:<30} {v}")
    _divider()
    if chart_summary:
        print(f"  Chart saved  ->  {chart_summary.get('output_path', '')}")
        print(
            f"  Top category :  {chart_summary.get('top_category', '')} "
            f"({chart_summary.get('top_share_pct', '')}% of total sales)"
        )
    _divider()
    print()

    # -- 6. Interactive Q&A session -----------------------------------------
    interactive_qa_session(df, groq_client)


# ===========================================================================
# Entry-point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to dataset.csv in the same directory
        default_csv = os.path.join(os.path.dirname(__file__), "dataset.csv")
        if os.path.exists(default_csv):
            print(f"  No CSV path given — using default: {default_csv}")
            run_pipeline(default_csv)
        else:
            print("Usage: py main.py <path_to_csv>")
            print("  Example: py main.py dataset.csv")
            sys.exit(1)
    else:
        run_pipeline(sys.argv[1])
