# ============================================================
# CE480 - HW4 Extreme Wave Analysis Using Gumbel Distribution
# Lecture-corrected orientation:
#   y_g = -ln(-ln(P))
#   Fit Hs = a*y_g + b
#
# Based on HW3 annual maximum significant wave heights Hs.
# ============================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

START_YEAR = 2005
END_YEAR = 2025

INPUT_CANDIDATES = [
    Path("HW3_outputs_2005_2025") / "HW3_summary_results_2005_2025.csv",
    Path("HW3_summary_results_2005_2025.csv"),
]

OUT_DIR = Path("HW4_gumbel_outputs_lecture_corrected")
OUT_DIR.mkdir(exist_ok=True)

RETURN_PERIODS_YEARS = [10, 50, 100]

C1 = 0.44
C2 = 0.12

EMBEDDED_HW3_ANNUAL_HS = [
    {"year": 2005, "Hs_m": 2.923191},
    {"year": 2006, "Hs_m": 0.834905},
    {"year": 2007, "Hs_m": 4.183988},
    {"year": 2008, "Hs_m": 3.368033},
    {"year": 2009, "Hs_m": 1.866054},
    {"year": 2010, "Hs_m": 3.366278},
    {"year": 2011, "Hs_m": 3.702909},
    {"year": 2012, "Hs_m": 1.215482},
    {"year": 2013, "Hs_m": 3.303181},
    {"year": 2014, "Hs_m": 5.059716},
    {"year": 2015, "Hs_m": 2.279769},
    {"year": 2016, "Hs_m": 3.225712},
    {"year": 2017, "Hs_m": 3.110653},
    {"year": 2018, "Hs_m": 7.074669},
    {"year": 2019, "Hs_m": 1.192844},
    {"year": 2020, "Hs_m": 3.138614},
    {"year": 2021, "Hs_m": 1.550646},
    {"year": 2022, "Hs_m": 0.512366},
    {"year": 2023, "Hs_m": 0.668832},
    {"year": 2024, "Hs_m": 6.235831},
    {"year": 2025, "Hs_m": 5.779957},
]


def find_input_file() -> Path | None:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    return None


def load_annual_hs() -> pd.DataFrame:
    path = find_input_file()
    if path is None:
        df = pd.DataFrame(EMBEDDED_HW3_ANNUAL_HS)
        print("HW3 CSV not found; using embedded corrected HW3 Hs values.")
    else:
        raw = pd.read_csv(path)
        if "year" not in raw.columns:
            raise ValueError("Input file must contain a 'year' column.")
        if "Hm0_Hs_m" in raw.columns:
            hcol = "Hm0_Hs_m"
        elif "Hs_m" in raw.columns:
            hcol = "Hs_m"
        else:
            raise ValueError("Input file must contain 'Hm0_Hs_m' or 'Hs_m'.")
        df = raw[["year", hcol]].rename(columns={hcol: "Hs_m"})

    df["year"] = df["year"].astype(int)
    df["Hs_m"] = df["Hs_m"].astype(float)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    df = df.dropna().sort_values("year").reset_index(drop=True)

    if df.empty:
        raise ValueError("No annual Hs data found.")

    return df


def gumbel_reduced_variate(P):
    P = np.asarray(P, dtype=float)
    if np.any(P <= 0) or np.any(P >= 1):
        raise ValueError("P must satisfy 0 < P < 1.")
    return -np.log(-np.log(P))


def main():
    annual = load_annual_hs()
    N = len(annual)

    ordered = annual.sort_values("Hs_m", ascending=False).reset_index(drop=True)
    ordered["rank_m"] = np.arange(1, N + 1)

    # Gringorten plotting position for Gumbel, as used in the lecture:
    ordered["P_non_exceedance"] = 1 - (ordered["rank_m"] - C1) / (N + C2)
    ordered["Q_exceedance"] = 1 - ordered["P_non_exceedance"]
    ordered["gumbel_y"] = gumbel_reduced_variate(ordered["P_non_exceedance"])

    # Lecture form:
    # P = exp[-exp(-(H-b)/a)]
    # -ln[-ln(P)] = (H-b)/a
    # H = a*y + b
    x = ordered["gumbel_y"].values
    y = ordered["Hs_m"].values
    a, b = np.polyfit(x, y, 1)

    ordered["Hs_fit_m"] = a * x + b
    ordered["fit_residual_m"] = ordered["Hs_m"] - ordered["Hs_fit_m"]

    ss_res = float(np.sum(ordered["fit_residual_m"] ** 2))
    ss_tot = float(np.sum((ordered["Hs_m"] - ordered["Hs_m"].mean()) ** 2))
    R2 = 1 - ss_res / ss_tot

    return_rows = []
    for Tr in RETURN_PERIODS_YEARS:
        Q = 1 / Tr
        P = 1 - Q
        yg = float(gumbel_reduced_variate(P))
        Hs_Tr = a * yg + b
        return_rows.append({
            "return_period_Tr_years": Tr,
            "P_non_exceedance": P,
            "Q_exceedance": Q,
            "gumbel_y_Tr": yg,
            "Hs_return_m": Hs_Tr,
        })

    return_df = pd.DataFrame(return_rows)

    ordered.to_csv(OUT_DIR / "HW4_gumbel_ordered_data_lecture_corrected.csv", index=False)
    return_df.to_csv(OUT_DIR / "HW4_gumbel_return_wave_heights_lecture_corrected.csv", index=False)

    x_line = np.linspace(
        min(ordered["gumbel_y"].min(), return_df["gumbel_y_Tr"].min()) - 0.3,
        max(ordered["gumbel_y"].max(), return_df["gumbel_y_Tr"].max()) + 0.3,
        300,
    )
    y_line = a * x_line + b

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ordered["gumbel_y"], ordered["Hs_m"], s=55, label="Annual maxima", zorder=3)
    ax.plot(x_line, y_line, linewidth=2, label=f"Fit: Hs = {a:.4f} y + {b:.4f}, R² = {R2:.4f}")
    ax.scatter(return_df["gumbel_y_Tr"], return_df["Hs_return_m"], marker="x", s=90, label="Return-period heights", zorder=4)

    for _, row in return_df.iterrows():
        ax.annotate(
            f"Tr={int(row['return_period_Tr_years'])} yr\nHs={row['Hs_return_m']:.2f} m",
            xy=(row["gumbel_y_Tr"], row["Hs_return_m"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Gumbel reduced variate, y = -ln(-ln(P))")
    ax.set_ylabel("Significant wave height, Hs (m)")
    ax.set_title("Gumbel Extreme Wave Analysis")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gumbel_Hs_vs_reduced_variate_lecture_corrected.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n================ HW4 GUMBEL RESULT ================")
    print(f"N = {N}")
    print("Fitted lecture form: Hs = a*y + b")
    print(f"a = {a:.8f}")
    print(f"b = {b:.8f}")
    print(f"R^2 = {R2:.8f}")
    print("\nReturn-period wave heights:")
    print(return_df[["return_period_Tr_years", "P_non_exceedance", "gumbel_y_Tr", "Hs_return_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
