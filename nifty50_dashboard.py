"""
Nifty 50 Financial Stock Dashboard
====================================
Replicates the interactive dashboard in Python using:
  - pandas  : data wrangling
  - matplotlib / seaborn : static charts
  - plotly  : interactive charts (optional – comment-out if not installed)

Run:
    pip install pandas matplotlib seaborn plotly
    python nifty50_dashboard.py

The script produces:
  1. Risk-Return scatter (sector-coloured, annotated)
  2. Sector YTD bar chart
  3. Beta distribution histogram
  4. Sector heatmap (colour-graded by YTD return)
  5. Summary metrics printed to console
  6. Sharpe-ratio ranked screener table (console + CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  DATASET  (indicative / illustrative data)
# ─────────────────────────────────────────────
STOCKS_DATA = [
    {"sym": "RELIANCE",   "name": "Reliance Industries",         "sector": "Energy",     "price": 2847,  "d1": 0.82,  "ytd": 12.4, "mcap_cr": 1920000, "pe": 28.4, "beta": 0.92, "vol": 22.1, "ret": 14.2},
    {"sym": "TCS",        "name": "Tata Consultancy Services",    "sector": "IT",         "price": 3621,  "d1": -0.34, "ytd": 8.2,  "mcap_cr": 1310000, "pe": 31.2, "beta": 0.68, "vol": 18.4, "ret": 10.1},
    {"sym": "HDFCBANK",   "name": "HDFC Bank",                    "sector": "Financials", "price": 1892,  "d1": 1.12,  "ytd": 15.8, "mcap_cr": 1440000, "pe": 19.8, "beta": 0.88, "vol": 20.6, "ret": 17.2},
    {"sym": "ICICIBANK",  "name": "ICICI Bank",                   "sector": "Financials", "price": 1324,  "d1": 0.65,  "ytd": 22.3, "mcap_cr": 930000,  "pe": 17.2, "beta": 0.94, "vol": 21.8, "ret": 24.1},
    {"sym": "INFY",       "name": "Infosys",                      "sector": "IT",         "price": 1654,  "d1": -0.78, "ytd": 5.1,  "mcap_cr": 690000,  "pe": 27.6, "beta": 0.72, "vol": 20.2, "ret": 6.8},
    {"sym": "ITC",        "name": "ITC",                          "sector": "FMCG",       "price": 482,   "d1": 0.44,  "ytd": 9.8,  "mcap_cr": 600000,  "pe": 28.1, "beta": 0.58, "vol": 16.2, "ret": 11.4},
    {"sym": "HINDUNILVR", "name": "Hindustan Unilever",           "sector": "FMCG",       "price": 2734,  "d1": -0.21, "ytd": 3.2,  "mcap_cr": 640000,  "pe": 56.8, "beta": 0.52, "vol": 14.8, "ret": 4.1},
    {"sym": "SBIN",       "name": "State Bank of India",          "sector": "Financials", "price": 812,   "d1": 1.44,  "ytd": 18.6, "mcap_cr": 720000,  "pe": 11.4, "beta": 1.14, "vol": 26.4, "ret": 20.8},
    {"sym": "BHARTIARTL", "name": "Bharti Airtel",                "sector": "Telecom",    "price": 1742,  "d1": 0.92,  "ytd": 24.1, "mcap_cr": 1040000, "pe": 82.4, "beta": 0.84, "vol": 19.6, "ret": 25.8},
    {"sym": "KOTAKBANK",  "name": "Kotak Mahindra Bank",          "sector": "Financials", "price": 2108,  "d1": -0.48, "ytd": 11.4, "mcap_cr": 420000,  "pe": 22.6, "beta": 0.78, "vol": 19.2, "ret": 12.8},
    {"sym": "WIPRO",      "name": "Wipro",                        "sector": "IT",         "price": 584,   "d1": -1.12, "ytd": -2.4, "mcap_cr": 300000,  "pe": 22.4, "beta": 0.74, "vol": 22.4, "ret": -1.2},
    {"sym": "HCLTECH",    "name": "HCL Technologies",             "sector": "IT",         "price": 1624,  "d1": 0.34,  "ytd": 14.8, "mcap_cr": 440000,  "pe": 24.8, "beta": 0.76, "vol": 21.2, "ret": 16.4},
    {"sym": "BAJFINANCE", "name": "Bajaj Finance",                "sector": "Financials", "price": 7842,  "d1": 1.82,  "ytd": 28.4, "mcap_cr": 470000,  "pe": 32.8, "beta": 1.18, "vol": 28.6, "ret": 30.2},
    {"sym": "ASIANPAINT", "name": "Asian Paints",                 "sector": "FMCG",       "price": 2924,  "d1": -0.64, "ytd": -4.2, "mcap_cr": 280000,  "pe": 62.4, "beta": 0.54, "vol": 15.6, "ret": -3.1},
    {"sym": "MARUTI",     "name": "Maruti Suzuki",                "sector": "Auto",       "price": 12840, "d1": 0.72,  "ytd": 16.8, "mcap_cr": 390000,  "pe": 29.4, "beta": 0.98, "vol": 23.4, "ret": 18.2},
    {"sym": "LTIM",       "name": "LTIMindtree",                  "sector": "IT",         "price": 5282,  "d1": -0.92, "ytd": 6.4,  "mcap_cr": 160000,  "pe": 34.2, "beta": 0.82, "vol": 24.8, "ret": 7.8},
    {"sym": "AXISBANK",   "name": "Axis Bank",                    "sector": "Financials", "price": 1184,  "d1": 0.88,  "ytd": 20.4, "mcap_cr": 380000,  "pe": 14.8, "beta": 1.04, "vol": 24.2, "ret": 22.1},
    {"sym": "TATAMOTORS", "name": "Tata Motors",                  "sector": "Auto",       "price": 924,   "d1": 1.24,  "ytd": 8.4,  "mcap_cr": 340000,  "pe": 8.4,  "beta": 1.34, "vol": 34.8, "ret": 10.2},
    {"sym": "ONGC",       "name": "Oil & Natural Gas Corp",       "sector": "Energy",     "price": 248,   "d1": -0.84, "ytd": 4.8,  "mcap_cr": 310000,  "pe": 8.2,  "beta": 1.04, "vol": 24.8, "ret": 6.1},
    {"sym": "NTPC",       "name": "NTPC",                         "sector": "Energy",     "price": 348,   "d1": 0.48,  "ytd": 11.2, "mcap_cr": 340000,  "pe": 15.8, "beta": 0.82, "vol": 18.8, "ret": 12.4},
    {"sym": "POWERGRID",  "name": "Power Grid Corp",              "sector": "Infra",      "price": 312,   "d1": 0.28,  "ytd": 6.8,  "mcap_cr": 290000,  "pe": 18.4, "beta": 0.74, "vol": 16.4, "ret": 8.2},
    {"sym": "ULTRACEMCO", "name": "UltraTech Cement",             "sector": "Infra",      "price": 11240, "d1": -0.34, "ytd": 7.4,  "mcap_cr": 320000,  "pe": 42.8, "beta": 0.88, "vol": 21.4, "ret": 9.1},
    {"sym": "BAJAJFINSV", "name": "Bajaj Finserv",                "sector": "Financials", "price": 1824,  "d1": 0.92,  "ytd": 14.8, "mcap_cr": 290000,  "pe": 28.4, "beta": 1.08, "vol": 26.2, "ret": 16.4},
    {"sym": "ADANIENT",   "name": "Adani Enterprises",            "sector": "Energy",     "price": 2784,  "d1": 2.14,  "ytd": 32.4, "mcap_cr": 320000,  "pe": 82.4, "beta": 1.48, "vol": 44.2, "ret": 35.8},
    {"sym": "ADANIPORTS", "name": "Adani Ports",                  "sector": "Infra",      "price": 1184,  "d1": 1.04,  "ytd": 18.4, "mcap_cr": 260000,  "pe": 32.4, "beta": 1.12, "vol": 28.4, "ret": 20.2},
    {"sym": "TATASTEEL",  "name": "Tata Steel",                   "sector": "Metals",     "price": 148,   "d1": 1.84,  "ytd": 14.8, "mcap_cr": 180000,  "pe": 14.8, "beta": 1.24, "vol": 32.4, "ret": 16.8},
    {"sym": "JSWSTEEL",   "name": "JSW Steel",                    "sector": "Metals",     "price": 842,   "d1": 0.84,  "ytd": 12.4, "mcap_cr": 210000,  "pe": 22.4, "beta": 1.18, "vol": 30.2, "ret": 14.2},
    {"sym": "SUNPHARMA",  "name": "Sun Pharmaceutical",           "sector": "Pharma",     "price": 1724,  "d1": -0.44, "ytd": 18.8, "mcap_cr": 410000,  "pe": 38.4, "beta": 0.68, "vol": 18.4, "ret": 20.4},
    {"sym": "DRREDDY",    "name": "Dr. Reddy's Labs",             "sector": "Pharma",     "price": 6284,  "d1": -0.28, "ytd": 12.4, "mcap_cr": 110000,  "pe": 24.8, "beta": 0.64, "vol": 17.2, "ret": 14.2},
    {"sym": "CIPLA",      "name": "Cipla",                        "sector": "Pharma",     "price": 1482,  "d1": 0.48,  "ytd": 16.4, "mcap_cr": 120000,  "pe": 32.4, "beta": 0.62, "vol": 16.8, "ret": 18.2},
    {"sym": "M&M",        "name": "Mahindra & Mahindra",          "sector": "Auto",       "price": 3124,  "d1": 1.42,  "ytd": 22.4, "mcap_cr": 390000,  "pe": 28.4, "beta": 1.02, "vol": 24.8, "ret": 24.2},
    {"sym": "EICHERMOT",  "name": "Eicher Motors",                "sector": "Auto",       "price": 5284,  "d1": 0.64,  "ytd": 14.8, "mcap_cr": 150000,  "pe": 38.2, "beta": 0.84, "vol": 22.4, "ret": 16.4},
    {"sym": "HEROMOTOCO", "name": "Hero MotoCorp",                "sector": "Auto",       "price": 4824,  "d1": -0.24, "ytd": 8.4,  "mcap_cr": 97000,   "pe": 22.8, "beta": 0.78, "vol": 20.4, "ret": 10.1},
    {"sym": "BPCL",       "name": "BPCL",                         "sector": "Energy",     "price": 348,   "d1": -1.24, "ytd": -8.4, "mcap_cr": 150000,  "pe": 7.4,  "beta": 1.14, "vol": 26.8, "ret": -6.8},
    {"sym": "COALINDIA",  "name": "Coal India",                   "sector": "Energy",     "price": 428,   "d1": -0.48, "ytd": 2.4,  "mcap_cr": 260000,  "pe": 8.4,  "beta": 0.88, "vol": 20.4, "ret": 3.8},
    {"sym": "NESTLEIND",  "name": "Nestle India",                 "sector": "FMCG",       "price": 24840, "d1": 0.14,  "ytd": 4.8,  "mcap_cr": 240000,  "pe": 74.8, "beta": 0.42, "vol": 13.4, "ret": 5.8},
    {"sym": "BRITANNIA",  "name": "Britannia Industries",         "sector": "FMCG",       "price": 5624,  "d1": -0.34, "ytd": 2.8,  "mcap_cr": 140000,  "pe": 58.4, "beta": 0.48, "vol": 14.2, "ret": 3.8},
    {"sym": "TATACONSUM", "name": "Tata Consumer Products",       "sector": "FMCG",       "price": 1124,  "d1": 0.48,  "ytd": 8.4,  "mcap_cr": 100000,  "pe": 68.4, "beta": 0.58, "vol": 16.8, "ret": 9.8},
    {"sym": "GRASIM",     "name": "Grasim Industries",            "sector": "Infra",      "price": 2724,  "d1": 0.34,  "ytd": 9.8,  "mcap_cr": 180000,  "pe": 24.8, "beta": 0.92, "vol": 22.4, "ret": 11.4},
    {"sym": "HINDALCO",   "name": "Hindalco Industries",          "sector": "Metals",     "price": 684,   "d1": 1.24,  "ytd": 18.4, "mcap_cr": 150000,  "pe": 18.4, "beta": 1.14, "vol": 28.4, "ret": 20.2},
    {"sym": "TECHM",      "name": "Tech Mahindra",                "sector": "IT",         "price": 1624,  "d1": -1.84, "ytd": -6.4, "mcap_cr": 160000,  "pe": 42.4, "beta": 0.88, "vol": 24.8, "ret": -4.8},
    {"sym": "INDUSINDBK", "name": "IndusInd Bank",                "sector": "Financials", "price": 924,   "d1": 2.84,  "ytd": 12.4, "mcap_cr": 72000,   "pe": 12.4, "beta": 1.28, "vol": 30.4, "ret": 14.2},
    {"sym": "DIVISLAB",   "name": "Divi's Laboratories",          "sector": "Pharma",     "price": 4824,  "d1": 0.14,  "ytd": 14.4, "mcap_cr": 130000,  "pe": 58.4, "beta": 0.54, "vol": 16.4, "ret": 16.2},
    {"sym": "LTTS",       "name": "L&T Technology Services",      "sector": "IT",         "price": 4284,  "d1": -0.64, "ytd": 2.4,  "mcap_cr": 45000,   "pe": 32.4, "beta": 0.84, "vol": 23.4, "ret": 3.8},
    {"sym": "APOLLOHOSP", "name": "Apollo Hospitals",             "sector": "Pharma",     "price": 7284,  "d1": 0.84,  "ytd": 28.4, "mcap_cr": 100000,  "pe": 84.8, "beta": 0.74, "vol": 20.4, "ret": 30.2},
    {"sym": "LT",         "name": "Larsen & Toubro",              "sector": "Infra",      "price": 3724,  "d1": 0.44,  "ytd": 14.4, "mcap_cr": 510000,  "pe": 32.8, "beta": 0.94, "vol": 21.8, "ret": 16.2},
    {"sym": "TRENT",      "name": "Trent",                        "sector": "FMCG",       "price": 6284,  "d1": 1.84,  "ytd": 38.4, "mcap_cr": 220000,  "pe": 148.4,"beta": 1.08, "vol": 28.4, "ret": 40.8},
    {"sym": "BEL",        "name": "Bharat Electronics",           "sector": "Infra",      "price": 284,   "d1": 0.84,  "ytd": 22.4, "mcap_cr": 210000,  "pe": 42.4, "beta": 0.94, "vol": 22.4, "ret": 24.2},
    {"sym": "SHRIRAMFIN", "name": "Shriram Finance",              "sector": "Financials", "price": 3724,  "d1": 0.64,  "ytd": 16.4, "mcap_cr": 140000,  "pe": 16.4, "beta": 1.06, "vol": 24.4, "ret": 18.2},
    {"sym": "VEDL",       "name": "Vedanta",                      "sector": "Metals",     "price": 484,   "d1": 1.44,  "ytd": 22.4, "mcap_cr": 180000,  "pe": 12.4, "beta": 1.28, "vol": 34.4, "ret": 24.8},
]

RISK_FREE_RATE = 6.0   # % annualised (approx 91-day T-bill)

SECTOR_COLORS = {
    "IT":         "#185fa5",
    "Financials": "#534ab7",
    "Energy":     "#ba7517",
    "FMCG":       "#3b6d11",
    "Auto":       "#993c1d",
    "Pharma":     "#993556",
    "Metals":     "#5f5e5a",
    "Infra":      "#0f6e56",
    "Telecom":    "#639922",
}

# ─────────────────────────────────────────────
# 2.  BUILD DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame(STOCKS_DATA)
df["sharpe"] = ((df["ret"] - RISK_FREE_RATE) / df["vol"]).round(2)
df["mcap_lakh_cr"] = (df["mcap_cr"] / 100000).round(2)

# ─────────────────────────────────────────────
# 3.  CONSOLE SUMMARY
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  NSE NIFTY 50 — DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"  Stocks tracked       : {len(df)}")
    print(f"  Avg YTD Return       : {df['ytd'].mean():.1f}%")
    print(f"  Advancers (today)    : {(df['d1'] > 0).sum()} / {len(df)}")
    print(f"  Avg Beta             : {df['beta'].mean():.2f}")
    print(f"  Avg Volatility       : {df['vol'].mean():.1f}%")
    print(f"  Avg Sharpe Ratio     : {df['sharpe'].mean():.2f}")
    top = df.nlargest(3, "ytd")[["sym", "ytd"]]
    bot = df.nsmallest(3, "ytd")[["sym", "ytd"]]
    print(f"\n  Top 3 YTD gainers    : {', '.join(f'{r.sym}(+{r.ytd}%)' for _, r in top.iterrows())}")
    print(f"  Top 3 YTD laggards   : {', '.join(f'{r.sym}({r.ytd}%)' for _, r in bot.iterrows())}")
    print("=" * 60 + "\n")

print_summary(df)

# ─────────────────────────────────────────────
# 4.  SCREENER TABLE  →  console + CSV
# ─────────────────────────────────────────────
screener_cols = ["sym", "sector", "price", "d1", "ytd", "pe", "beta", "vol", "sharpe"]
screener = df[screener_cols].sort_values("sharpe", ascending=False).reset_index(drop=True)
screener.columns = ["Symbol", "Sector", "Price(₹)", "1D%", "YTD%", "P/E", "Beta", "Volatility%", "Sharpe"]

print(screener.to_string(index=False))

screener.to_csv("nifty50_screener.csv", index=False)
print("\n[Saved] nifty50_screener.csv\n")

# ─────────────────────────────────────────────
# 5.  PLOT SETTINGS
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linewidth":   0.6,
    "figure.facecolor": "#fafafa",
    "axes.facecolor":   "#fafafa",
})

# ─────────────────────────────────────────────
# CHART 1 — RISK-RETURN SCATTER
# ─────────────────────────────────────────────
def plot_risk_return(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#fafafa")

    for sector, grp in df.groupby("sector"):
        color = SECTOR_COLORS.get(sector, "#888888")
        ax.scatter(grp["vol"], grp["ret"],
                   color=color, s=grp["mcap_lakh_cr"] * 60 + 40,
                   alpha=0.85, edgecolors="white", linewidths=0.8,
                   label=sector, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row["sym"],
                        (row["vol"], row["ret"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="#444444")

    # Quadrant reference lines
    ax.axhline(0,  color="#aaaaaa", lw=0.8, ls="--")
    ax.axhline(df["ret"].mean(), color="#888888", lw=0.6, ls=":", label=f"Avg return ({df['ret'].mean():.1f}%)")
    ax.axvline(df["vol"].mean(), color="#888888", lw=0.6, ls=":", label=f"Avg volatility ({df['vol'].mean():.1f}%)")

    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualised Return (%)",     fontsize=11)
    ax.set_title("Nifty 50 — Risk-Return Scatter\n(bubble size ∝ market cap)",
                 fontsize=13, fontweight="bold", pad=14)

    legend_patches = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#888"), label=s)
                      for s in df["sector"].unique()]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              framealpha=0.8, title="Sector", title_fontsize=9)

    plt.tight_layout()
    plt.savefig("chart1_risk_return.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart1_risk_return.png")
    plt.show()


# ─────────────────────────────────────────────
# CHART 2 — SECTOR YTD BAR
# ─────────────────────────────────────────────
def plot_sector_ytd(df: pd.DataFrame) -> None:
    sec = (df.groupby("sector")["ytd"]
             .mean()
             .sort_values(ascending=True))

    colors = [SECTOR_COLORS.get(s, "#888888") for s in sec.index]
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(sec.index, sec.values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    for bar, val in zip(bars, sec.values):
        ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color="#333333")

    ax.axvline(0, color="#aaaaaa", lw=0.8)
    ax.set_xlabel("Average YTD Return (%)", fontsize=10)
    ax.set_title("Sector Performance — Average YTD Return",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig("chart2_sector_ytd.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart2_sector_ytd.png")
    plt.show()


# ─────────────────────────────────────────────
# CHART 3 — BETA DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────
def plot_beta_dist(df: pd.DataFrame) -> None:
    bins = [0, 0.5, 0.7, 0.9, 1.1, 1.3, 2.0]
    labels = ["<0.5", "0.5–0.7", "0.7–0.9", "0.9–1.1", "1.1–1.3", ">1.3"]
    counts = pd.cut(df["beta"], bins=bins, labels=labels).value_counts().sort_index()

    palette = ["#b5d4f4", "#85b7eb", "#378add", "#185fa5", "#0c447c", "#042c53"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=palette,
                  edgecolor="white", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                str(val), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Beta Bucket", fontsize=10)
    ax.set_ylabel("Number of Stocks", fontsize=10)
    ax.set_title("Beta Distribution — Nifty 50 Stocks",
                 fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("chart3_beta_dist.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart3_beta_dist.png")
    plt.show()


# ─────────────────────────────────────────────
# CHART 4 — SECTOR HEATMAP (YTD % per stock)
# ─────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="sector", columns="sym", values="ytd", aggfunc="first")

    # Pad with NaN so grid is uniform
    max_cols = pivot.shape[1]
    fig, ax = plt.subplots(figsize=(20, 6))

    cmap = sns.diverging_palette(10, 130, s=80, l=45, as_cmap=True)
    sns.heatmap(pivot,
                annot=True, fmt=".1f", linewidths=0.4,
                linecolor="white", cmap=cmap,
                center=0, vmin=-15, vmax=40,
                cbar_kws={"label": "YTD Return (%)", "shrink": 0.6},
                ax=ax)
    ax.set_title("Nifty 50 — YTD Return Heatmap by Sector",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig("chart4_heatmap.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart4_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────
# CHART 5 — SHARPE RATIO RANKING (Top 20)
# ─────────────────────────────────────────────
def plot_sharpe_ranking(df: pd.DataFrame, top_n: int = 20) -> None:
    ranked = df.nlargest(top_n, "sharpe")
    colors = [SECTOR_COLORS.get(s, "#888888") for s in ranked["sector"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ranked["sym"][::-1], ranked["sharpe"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5, height=0.7)

    for bar, val in zip(bars, ranked["sharpe"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    ax.set_xlabel("Sharpe Ratio  (risk-free rate = 6%)", fontsize=10)
    ax.set_title(f"Top {top_n} Nifty 50 Stocks by Sharpe Ratio",
                 fontsize=12, fontweight="bold", pad=10)

    legend_patches = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#888"), label=s)
                      for s in ranked["sector"].unique()]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right", title="Sector")

    plt.tight_layout()
    plt.savefig("chart5_sharpe.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart5_sharpe.png")
    plt.show()


# ─────────────────────────────────────────────
# CHART 6 — P/E vs RETURN SCATTER
# ─────────────────────────────────────────────
def plot_pe_vs_return(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))

    for sector, grp in df.groupby("sector"):
        color = SECTOR_COLORS.get(sector, "#888888")
        ax.scatter(grp["pe"], grp["ytd"],
                   color=color, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.6, label=sector, zorder=3)
        for _, row in grp.iterrows():
            ax.annotate(row["sym"], (row["pe"], row["ytd"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=7, color="#444444")

    ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_xlabel("P/E Ratio", fontsize=11)
    ax.set_ylabel("YTD Return (%)", fontsize=11)
    ax.set_title("Nifty 50 — Valuation (P/E) vs YTD Return",
                 fontsize=13, fontweight="bold", pad=14)

    legend_patches = [mpatches.Patch(color=SECTOR_COLORS.get(s, "#888"), label=s)
                      for s in df["sector"].unique()]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              framealpha=0.8, title="Sector")

    plt.tight_layout()
    plt.savefig("chart6_pe_vs_return.png", dpi=150, bbox_inches="tight")
    print("[Saved] chart6_pe_vs_return.png")
    plt.show()


# ─────────────────────────────────────────────
# RUN ALL CHARTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    plot_risk_return(df)
    plot_sector_ytd(df)
    plot_beta_dist(df)
    plot_heatmap(df)
    plot_sharpe_ranking(df)
    plot_pe_vs_return(df)
    print("\nAll charts saved. Dashboard complete.")
