"""
analysis.py
-----------
Core data analysis utilities for the AI Data Analysis Assistant.
Two public functions:
    load_data(filepath)  -> pd.DataFrame | None
    analyze_data(df)     -> None
"""

import os
import pandas as pd


# ===========================================================================
# Internal helpers — clean console formatting
# ===========================================================================

WIDTH = 65

def _line(char: str = "=") -> None:
    print(char * WIDTH)

def _header(title: str) -> None:
    print()
    _line("=")
    print(f"  {title}")
    _line("=")

def _sub(title: str) -> None:
    print()
    print(f"  {'-' * (WIDTH - 4)}")
    print(f"  >  {title}")
    print(f"  {'-' * (WIDTH - 4)}")


# ===========================================================================
# Function 1 — load_data
# ===========================================================================

def load_data(filepath: str):
    """
    Read a CSV file into a DataFrame and print a structural overview.

    Prints
    ------
    - Row and column counts
    - Each column name with its data type
    - Missing-value report (count + % per column)

    Returns
    -------
    pd.DataFrame on success, None on failure.
    """
    _header("  LOAD DATA")

    # --- validate -----------------------------------------------------------
    if not os.path.exists(filepath):
        print(f"\n  [ERR]  File not found: {filepath!r}")
        return None
    if not filepath.lower().endswith(".csv"):
        print(f"  [WARN]  '{filepath}' may not be a CSV — attempting anyway.")

    # --- read ---------------------------------------------------------------
    try:
        df = pd.read_csv(filepath)
    except Exception as err:
        print(f"\n  [ERR]  Could not read file: {err}")
        return None

    print(f"\n  [OK]  Loaded: {filepath}")

    # --- shape --------------------------------------------------------------
    _sub("Shape")
    rows, cols = df.shape
    print(f"  {'Rows':<22} {rows:>10,}")
    print(f"  {'Columns':<22} {cols:>10}")

    # --- data types ---------------------------------------------------------
    _sub("Columns & Data Types")
    print(f"  {'#':<4}  {'Column Name':<28}  {'Dtype'}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*12}")
    for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
        print(f"  {i:<4}  {col:<28}  {dtype}")

    # --- missing values -----------------------------------------------------
    _sub("Missing Values")
    miss      = df.isnull().sum()
    miss_pct  = (miss / rows * 100).round(2)
    has_miss  = miss[miss > 0]

    if has_miss.empty:
        print("  [OK]  No missing values found.")
    else:
        print(f"  {'Column':<28}  {'Missing':>8}  {'% Rows':>8}")
        print(f"  {'-'*28}  {'-'*8}  {'-'*8}")
        for col in has_miss.index:
            print(f"  {col:<28}  {miss[col]:>8,}  {miss_pct[col]:>7.2f}%")
        total_miss  = int(miss.sum())
        total_cells = rows * cols
        print(f"\n  Total missing cells: {total_miss:,} / {total_cells:,}"
              f"  ({total_miss / total_cells * 100:.2f}%)")

    print(f"\n{'=' * WIDTH}\n")
    return df


# ===========================================================================
# Function 2 — analyze_data
# ===========================================================================

def analyze_data(df) -> None:
    """
    Compute and print descriptive statistics for every column.

    Covers
    ------
    - Record / duplicate counts
    - Numeric: min, max, mean, median, std dev, Q1, Q3, IQR
    - Categorical: top-10 value frequencies with ASCII bar chart
    """
    if df is None or df.empty:
        print("\n  [ERR]  DataFrame is empty or None — nothing to analyse.")
        return

    _header("  DESCRIPTIVE STATISTICS")

    rows, cols = df.shape

    # --- general summary ----------------------------------------------------
    _sub("General Summary")
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    print(f"  {'Total Records':<28} {rows:>10,}")
    print(f"  {'Total Columns':<28} {cols:>10}")
    print(f"  {'Numeric Columns':<28} {len(num_cols):>10}")
    print(f"  {'Categorical Columns':<28} {len(cat_cols):>10}")
    print(f"  {'Duplicate Rows':<28} {df.duplicated().sum():>10,}")

    # --- numeric statistics -------------------------------------------------
    if len(num_cols):
        _sub("Numeric Statistics")
        hdr = (f"  {'Column':<20}  {'Min':>10}  {'Max':>10}"
               f"  {'Mean':>10}  {'Median':>10}  {'Std Dev':>10}")
        print(hdr)
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                print(f"  {col:<20}  {'N/A':>10}  {'N/A':>10}"
                      f"  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
            else:
                print(f"  {col:<20}  {s.min():>10.2f}  {s.max():>10.2f}"
                      f"  {s.mean():>10.2f}  {s.median():>10.2f}  {s.std():>10.2f}")

        # quartiles
        _sub("Quartiles  (Q1 · Q3 · IQR)")
        print(f"  {'Column':<20}  {'Q1 (25%)':>10}  {'Q3 (75%)':>10}  {'IQR':>10}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            print(f"  {col:<20}  {q1:>10.2f}  {q3:>10.2f}  {q3 - q1:>10.2f}")
    else:
        print("\n  [INFO]  No numeric columns found.")

    # --- categorical frequency distributions --------------------------------
    if len(cat_cols):
        _sub("Categorical Frequency Distributions  (top 10 per column)")
        for col in cat_cols:
            s      = df[col].dropna()
            total  = len(s)
            counts = s.value_counts().head(10)
            print(f"\n  Column: '{col}'  —  {s.nunique()} unique value(s)")
            print(f"  {'Value':<30}  {'Count':>7}  {'Share':>7}")
            print(f"  {'-'*30}  {'-'*7}  {'-'*7}")
            for val, cnt in counts.items():
                pct = cnt / total * 100
                bar = "█" * min(int(pct / 4), 20)
                print(f"  {str(val):<30}  {cnt:>7,}  {pct:>6.1f}%  {bar}")
    else:
        print("\n  [INFO]  No categorical columns found.")

    print(f"\n{'=' * WIDTH}\n")


# ===========================================================================
# Function 3 — answer_questions
# ===========================================================================

def answer_questions(df) -> dict:
    """
    Answer 3 evaluation questions programmatically from the DataFrame.

    Questions
    ---------
    1. Which product has the highest total sales?
    2. What is the average customer age?
    3. Which product category appears most frequently?

    Returns
    -------
    dict with keys:
        'highest_sales_product'   : str   — product name with max total sales
        'average_customer_age'    : float — mean of the 'Age' column (2 d.p.)
        'most_frequent_category'  : str   — mode of the 'Category' column
    Any question that cannot be answered (missing column / empty data)
    will have the value None and a warning is printed.
    """
    if df is None or df.empty:
        print("\n  [ERR]  DataFrame is empty or None — cannot answer questions.")
        return {}

    _header("?  EVALUATION QUESTIONS")

    results = {}

    # -- Q1: Highest sales product -----------------------------------------
    _sub("Q1 · Highest Total Sales Product")
    if "Product" in df.columns and "Sales" in df.columns:
        product_sales = df.groupby("Product")["Sales"].sum()
        top_product   = product_sales.idxmax()
        top_sales     = product_sales.max()
        results["highest_sales_product"] = top_product
        print(f"  Product  :  {top_product}")
        print(f"  Total Sales  :  {top_sales:,.2f}")
    else:
        missing = [c for c in ("Product", "Sales") if c not in df.columns]
        print(f"  [WARN]  Missing column(s): {missing} — skipping Q1.")
        results["highest_sales_product"] = None

    # -- Q2: Average customer age ------------------------------------------
    _sub("Q2 · Average Customer Age")
    if "Age" in df.columns:
        avg_age = round(df["Age"].mean(), 2)
        results["average_customer_age"] = avg_age
        print(f"  Average Age  :  {avg_age}")
    else:
        print("  [WARN]  Column 'Age' not found — skipping Q2.")
        results["average_customer_age"] = None

    # -- Q3: Most frequent category ----------------------------------------
    _sub("Q3 · Most Frequent Product Category")
    if "Category" in df.columns:
        top_category = df["Category"].mode()[0]
        freq         = (df["Category"] == top_category).sum()
        results["most_frequent_category"] = top_category
        print(f"  Category  :  {top_category}")
        print(f"  Frequency  :  {freq:,} row(s)")
    else:
        print("  [WARN]  Column 'Category' not found — skipping Q3.")
        results["most_frequent_category"] = None

    # -- Summary dict ------------------------------------------------------
    _sub("Results Dictionary")
    for key, val in results.items():
        print(f"  {key:<30}  {val}")

    print(f"\n{'=' * WIDTH}\n")
    return results


# ===========================================================================
# CLI entry-point:  python analysis.py <file.csv>
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: py analysis.py <path_to_csv>")
        sys.exit(1)

    dataframe = load_data(sys.argv[1])
    if dataframe is not None:
        analyze_data(dataframe)
        answer_questions(dataframe)
