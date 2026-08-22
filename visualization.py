"""
visualization.py
----------------
Chart generation utilities for the AI Data Analysis Assistant.

Public function
---------------
    generate_chart(df, output_path) -> dict
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")                      # headless / file-only backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# generate_chart
# ===========================================================================

def generate_chart(
    df,
    output_path: str = "charts/sales_by_category.png",
) -> dict:
    """
    Aggregate total sales by Category, plot a professional horizontal bar
    chart with Seaborn + Matplotlib, save it as a high-DPI PNG, and
    return a summary dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'Category' and 'Sales'.
    output_path : str
        Destination file path for the PNG.
        Parent directories are created automatically.
        Default: 'charts/sales_by_category.png'

    Returns
    -------
    dict
        {
            'top_category'  : str   -- category with the highest total sales
            'top_share_pct' : float -- its % share of all sales (2 d.p.)
            'output_path'   : str   -- absolute path of the saved file
            'total_sales'   : float -- grand total across all categories
        }
        Returns an empty dict if required columns are missing or data is empty.
    """

    # -- Validate -----------------------------------------------------------
    if df is None or df.empty:
        print("  [ERROR]  generate_chart: DataFrame is empty or None.")
        return {}

    missing = [c for c in ("Category", "Sales") if c not in df.columns]
    if missing:
        print(f"  [ERROR]  generate_chart: Missing column(s): {missing}")
        return {}

    # -- Aggregate ----------------------------------------------------------
    agg = (
        df.groupby("Category", sort=False)["Sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"Sales": "Total Sales"})
    )

    grand_total  = agg["Total Sales"].sum()
    top_category = agg.loc[0, "Category"]
    top_share    = round(agg.loc[0, "Total Sales"] / grand_total * 100, 2)

    # -- Chart setup --------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.05)
    palette = sns.color_palette("mako_r", n_colors=len(agg))

    n      = len(agg)
    fig_h  = max(5, n * 0.6 + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # -- Horizontal bar chart -----------------------------------------------
    bars = sns.barplot(
        data      = agg,
        x         = "Total Sales",
        y         = "Category",
        palette   = palette,
        edgecolor = "white",
        linewidth = 0.7,
        ax        = ax,
    )

    # -- Data labels --------------------------------------------------------
    for patch, (_, row) in zip(bars.patches, agg.iterrows()):
        pct   = row["Total Sales"] / grand_total * 100
        label = f"  ${row['Total Sales']:,.0f}  ({pct:.1f}%)"
        ax.text(
            patch.get_width(),
            patch.get_y() + patch.get_height() / 2,
            label,
            va        = "center",
            ha        = "left",
            fontsize  = 9,
            color     = "#333333",
        )

    # Highlight top bar with a gold border
    bars.patches[0].set_edgecolor("#f5a623")
    bars.patches[0].set_linewidth(2.5)

    # -- Titles & axes ------------------------------------------------------
    ax.set_title(
        "Total Sales by Product Category",
        fontsize   = 15,
        fontweight = "bold",
        pad        = 16,
        color      = "#1a1a2e",
    )
    ax.set_xlabel("Total Sales (USD)", fontsize=11, labelpad=9)
    ax.set_ylabel("Product Category",  fontsize=11, labelpad=9)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.set_xlim(0, agg["Total Sales"].max() * 1.30)

    # Footnote annotation
    ax.annotate(
        f"Top: {top_category}  ({top_share}% of total sales)",
        xy         = (0.99, 0.02),
        xycoords   = "axes fraction",
        ha         = "right",
        fontsize   = 9,
        color      = "#f5a623",
        style      = "italic",
    )

    sns.despine(left=True, bottom=False)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout(pad=1.6)

    # -- Save --------------------------------------------------------------
    out_abs = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    fig.savefig(out_abs, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n  [OK]  Chart saved  ->  {out_abs}")
    print(f"  [OK]  Top category :  {top_category}  ({top_share}% of total)")

    # -- Return summary ----------------------------------------------------
    return {
        "top_category"  : top_category,
        "top_share_pct" : top_share,
        "output_path"   : out_abs,
        "total_sales"   : round(grand_total, 2),
    }


# ===========================================================================
# CLI entry-point:  py visualization.py <file.csv> [output_path]
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: py visualization.py <path_to_csv> [output_path]")
        sys.exit(1)

    csv_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "charts/sales_by_category.png"

    data    = pd.read_csv(csv_file)
    summary = generate_chart(data, out_file)

    if summary:
        print("\n  Summary:")
        for k, v in summary.items():
            print(f"    {k:<18} {v}")
