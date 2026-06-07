# ============================================================
# CE480 - HW3 Wave Prediction from ECMWF/ERA5 Wind Data
# Corrected version:
#   - includes year 2025
#   - uses corrected FAS threshold logic with max()
#   - keeps the official HW2 effective fetch table
#   - produces wind rose, annual results, and storm wind data
#
# INPUT:
#   combined_2005_2025.nc
#
# Expected variables:
#   wind = 10 m wind speed, m/s
#   dwi  = converted wind direction, degrees FROM which wind blows
#
# OUTPUT FOLDER:
#   HW3_outputs_2005_2025
# ============================================================

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

NC_FILE = Path("combined_2005_2025.nc")

# User requested to include 2025.
START_YEAR = 2005
END_YEAR = 2025

ORIGIN = (41.934003, 28.069026)  # (latitude, longitude)

# Ray coordinates from HW2. These are kept for reference only.
# The final submitted HW2 fetch table is used by default below.
RAY_COORDS = [
    (352.5,  42.845649, 27.895897),
    (0.0,    43.339963, 28.078544),
    (7.5,    43.409006, 28.340631),
    (15.0,   44.754456, 29.135816),
    (22.5,   46.540415, 30.908103),
    (30.0,   46.213721, 31.684169),
    (37.5,   46.018012, 32.683854),
    (45.0,   45.292607, 33.001442),
    (52.5,   44.761736, 33.541665),
    (60.0,   45.080096, 45.080096),
    (67.5,   44.465643, 38.129395),
    (75.0,   43.607073, 39.739132),
    (82.5,   42.474583, 41.532527),
    (90.0,   41.819372, 32.646569),
    (97.5,   41.486146, 31.860025),
    (105.0,  41.198339, 31.382485),
    (112.5,  41.183836, 30.386227),
    (120.0,  41.160665, 29.756132),
    (127.5,  41.234781, 29.234569),
    (135.0,  41.254633, 28.959466),
    (142.5,  41.347997, 28.645114),
    (150.0,  41.418274, 28.460106),
    (157.5,  41.476751, 28.316214),
    (165.0,  41.550431, 28.191785),
    (172.5,  41.615920, 28.122759),
    (180.0,  41.700600, 28.066728),
    (187.5,  41.753531, 28.040335),
    (330.0,  42.552445, 27.585892),
    (337.5,  42.633179, 27.677957),
    (345.0,  42.695080, 27.794461),
]

# Use the final HW2 fetch table. This is recommended because it exactly
# matches the values submitted in HW2.
USE_HW2_FETCH_TABLE = True

HW2_FETCH_KM = {
    0.0: 140.68,
    22.5: 478.25,
    45.0: 554.31,
    67.5: 1074.12,
    90.0: 597.69,
    112.5: 219.42,
    135.0: 103.06,
    157.5: 54.55,
    180.0: 27.11,
    202.5: 0.00,
    225.0: 0.00,
    247.5: 0.00,
    270.0: 0.00,
    292.5: 0.00,
    315.0: 0.00,
    337.5: 83.25,
}

FETCH_VALID_THRESHOLD_KM = 5.0

# HW2 fetch rays use main -7.5, main, main +7.5 degrees.
FETCH_RAY_HALF_WIDTH_DEG = 7.5

# 16-sector wind direction bin half-width:
# 360/16 = 22.5 degrees, so half-width is 11.25 degrees.
WIND_DIRECTION_HALF_WIDTH_DEG = 11.25

# Storm definition:
# continuous hourly records around the yearly peak wind where:
#   1) wind remains in the critical direction sector
#   2) wind speed remains above threshold
# If STORM_SPEED_THRESHOLD is None, threshold = 0.75 * yearly peak wind.
STORM_SPEED_THRESHOLD = None
STORM_FRACTION_OF_UMAX = 0.75

# Lecture wind-stress factor: UA = 0.71 * U10^1.23
USE_WIND_STRESS_FACTOR = True

# Keep False if 'dwi' is already direction FROM which wind blows.
# Set True only if your file direction is direction TOWARDS which wind blows.
CONVERT_TOWARD_TO_FROM = False

OUT_DIR = Path("HW3_outputs_2005_2025")
OUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. BASIC FUNCTIONS
# ------------------------------------------------------------

def angular_difference_deg(angle: np.ndarray | float, center: float) -> np.ndarray | float:
    """Smallest signed angular difference angle-center, range [-180, 180)."""
    return (np.asarray(angle) - center + 180.0) % 360.0 - 180.0


def in_direction_sector(direction_deg: pd.Series | np.ndarray, center_deg: float, half_width_deg: float) -> np.ndarray:
    """True if direction is inside a circular sector centered at center_deg."""
    diff = np.abs(angular_difference_deg(np.asarray(direction_deg), center_deg))
    return diff <= half_width_deg


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


def wind_stress_factor_uA(u10: float) -> float:
    """Adjusted wind speed UA in m/s."""
    if USE_WIND_STRESS_FACTOR:
        return 0.71 * (u10 ** 1.23)
    return u10


# ------------------------------------------------------------
# 3. LOAD NETCDF WIND DATA
# ------------------------------------------------------------

def load_wind_data(nc_file: Path) -> pd.DataFrame:
    """
    Loads wind speed and direction from a NetCDF file.
    First tries xarray. If unavailable, uses h5py fallback.
    """
    if not nc_file.exists():
        raise FileNotFoundError(
            f"Cannot find {nc_file}. Put this script in the same folder as the .nc file, "
            "or update NC_FILE to the correct path."
        )

    try:
        import xarray as xr
        ds = xr.open_dataset(nc_file)

        if "wind" not in ds or "dwi" not in ds:
            raise KeyError(f"Expected variables 'wind' and 'dwi'. Found: {list(ds.data_vars)}")

        if "valid_time" in ds:
            time_values = ds["valid_time"].values
        elif "time" in ds:
            time_values = ds["time"].values
        else:
            raise KeyError("Expected time coordinate named 'valid_time' or 'time'.")

        time = pd.to_datetime(time_values)
        speed = np.asarray(ds["wind"].squeeze().values, dtype=float)
        direction = np.asarray(ds["dwi"].squeeze().values, dtype=float)

    except Exception:
        import h5py
        with h5py.File(nc_file, "r") as f:
            if "valid_time" in f:
                time = pd.to_datetime(f["valid_time"][:], unit="s", utc=True).tz_convert(None)
            elif "time" in f:
                time = pd.to_datetime(f["time"][:], unit="s", utc=True).tz_convert(None)
            else:
                raise KeyError("Expected time variable named 'valid_time' or 'time'.")

            speed_arr = np.asarray(f["wind"][:], dtype=float)
            direction_arr = np.asarray(f["dwi"][:], dtype=float)

            speed = np.squeeze(speed_arr)
            direction = np.squeeze(direction_arr)

    if CONVERT_TOWARD_TO_FROM:
        direction = (direction + 180.0) % 360.0

    df = pd.DataFrame({
        "time": pd.to_datetime(time),
        "wind_speed_m_s": np.ravel(speed),
        "wind_dir_from_deg": np.ravel(direction) % 360.0,
    })

    df = df.dropna().sort_values("time").reset_index(drop=True)
    df["year"] = df["time"].dt.year
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    if df.empty:
        raise ValueError(f"No data found between {START_YEAR} and {END_YEAR}.")

    return df


# ------------------------------------------------------------
# 4. FETCH TABLE AND CRITICAL DIRECTION
# ------------------------------------------------------------

def compute_effective_fetch_table() -> pd.DataFrame:
    """Computes effective fetch for each 22.5-degree main direction from RAY_COORDS."""
    main_dirs = np.arange(0.0, 360.0, 22.5)
    rows = []
    lat0, lon0 = ORIGIN

    for main in main_dirs:
        sector_rays = []
        for bearing, lat, lon in RAY_COORDS:
            if abs(float(angular_difference_deg(bearing, main))) <= FETCH_RAY_HALF_WIDTH_DEG + 1e-9:
                F_i = haversine_km(lat0, lon0, lat, lon)
                alpha_rad = math.radians(abs(float(angular_difference_deg(bearing, main))))
                sector_rays.append((bearing, F_i, alpha_rad))

        if not sector_rays:
            feff = 0.0
        else:
            numerator = sum(F_i * (math.cos(alpha) ** 2) for _, F_i, alpha in sector_rays)
            denominator = sum(math.cos(alpha) for _, _, alpha in sector_rays)
            feff = numerator / denominator

        rows.append({
            "bearing_deg": main,
            "effective_fetch_km": feff,
            "status": "VALID" if feff >= FETCH_VALID_THRESHOLD_KM else "EXCLUDED",
            "number_of_rays_used": len(sector_rays),
        })

    return pd.DataFrame(rows)


def get_fetch_table() -> pd.DataFrame:
    """Returns the HW2 fetch table or recomputes fetch from ray coordinates."""
    if USE_HW2_FETCH_TABLE:
        rows = []
        for bearing, fetch in HW2_FETCH_KM.items():
            rows.append({
                "bearing_deg": float(bearing),
                "effective_fetch_km": float(fetch),
                "status": "VALID" if fetch >= FETCH_VALID_THRESHOLD_KM else "EXCLUDED",
                "number_of_rays_used": np.nan,
            })
        return pd.DataFrame(rows).sort_values("bearing_deg").reset_index(drop=True)

    return compute_effective_fetch_table()


# ------------------------------------------------------------
# 5. WIND ROSE
# ------------------------------------------------------------

def plot_wind_rose(df: pd.DataFrame, output_png: Path) -> None:
    """Creates a wind rose as a polar stacked bar chart."""
    direction_centers = np.arange(0.0, 360.0, 22.5)
    speed_bins = [0, 2, 4, 6, 8, 10, 12, 15, 20, np.inf]
    speed_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-15", "15-20", ">20"]

    counts = np.zeros((len(speed_bins) - 1, len(direction_centers)))

    for j, center in enumerate(direction_centers):
        dir_mask = in_direction_sector(df["wind_dir_from_deg"], center, 11.25)
        dir_speeds = df.loc[dir_mask, "wind_speed_m_s"].values
        hist, _ = np.histogram(dir_speeds, bins=speed_bins)
        counts[:, j] = hist

    percent = counts / len(df) * 100.0

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    theta = np.deg2rad(direction_centers)
    width = np.deg2rad(22.5 * 0.90)
    bottom = np.zeros(len(direction_centers))

    cmap = plt.get_cmap("viridis", len(speed_labels))
    for i, label in enumerate(speed_labels):
        ax.bar(
            theta,
            percent[i],
            width=width,
            bottom=bottom,
            align="center",
            label=f"{label} m/s",
            color=cmap(i),
            edgecolor="black",
            linewidth=0.2,
        )
        bottom += percent[i]

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(direction_centers))
    ax.set_xticklabels([f"{d:.1f}°" if d % 1 else f"{int(d)}°" for d in direction_centers])
    ax.set_title("Wind Rose - Wind Speed Distribution by Direction", pad=30, fontsize=14)
    ax.set_rlabel_position(225)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), title="Wind speed")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 6. STORM DETECTION
# ------------------------------------------------------------

def find_storm_covering_peak(
    year_df: pd.DataFrame,
    peak_time: pd.Timestamp,
    critical_dir: float,
    annual_umax: float,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Finds the continuous hourly storm block covering the annual maximum wind.

    The storm block is defined as consecutive hourly records around the peak where:
        direction is inside the critical direction sector
        wind speed >= threshold

    If STORM_SPEED_THRESHOLD is None:
        threshold = STORM_FRACTION_OF_UMAX * annual_umax
    """
    year_df = year_df.sort_values("time").reset_index(drop=True)

    peak_pos_list = np.where(year_df["time"].values == np.datetime64(peak_time))[0]
    if len(peak_pos_list) == 0:
        raise ValueError("Peak time was not found inside yearly dataframe.")
    peak_pos = int(peak_pos_list[0])

    if STORM_SPEED_THRESHOLD is None:
        threshold = STORM_FRACTION_OF_UMAX * annual_umax
    else:
        threshold = float(STORM_SPEED_THRESHOLD)

    direction_mask = in_direction_sector(
        year_df["wind_dir_from_deg"],
        critical_dir,
        WIND_DIRECTION_HALF_WIDTH_DEG,
    )
    speed_mask = year_df["wind_speed_m_s"].values >= threshold
    storm_mask = direction_mask & speed_mask

    # The selected peak should already satisfy the storm condition.
    storm_mask[peak_pos] = True

    start_pos = peak_pos
    while start_pos > 0:
        time_gap_h = (
            year_df.loc[start_pos, "time"] - year_df.loc[start_pos - 1, "time"]
        ).total_seconds() / 3600.0
        if storm_mask[start_pos - 1] and time_gap_h <= 1.5:
            start_pos -= 1
        else:
            break

    end_pos = peak_pos
    while end_pos < len(year_df) - 1:
        time_gap_h = (
            year_df.loc[end_pos + 1, "time"] - year_df.loc[end_pos, "time"]
        ).total_seconds() / 3600.0
        if storm_mask[end_pos + 1] and time_gap_h <= 1.5:
            end_pos += 1
        else:
            break

    storm_df = year_df.loc[start_pos:end_pos].copy()
    storm_df["storm_threshold_m_s"] = threshold
    storm_df["critical_direction_deg"] = critical_dir
    storm_df["inside_critical_sector"] = in_direction_sector(
        storm_df["wind_dir_from_deg"],
        critical_dir,
        WIND_DIRECTION_HALF_WIDTH_DEG,
    )

    # With hourly records, duration in hours is the number of consecutive hourly records.
    # This is equivalent to end_time - start_time + 1 hour.
    duration_hours = float(len(storm_df))

    return storm_df, duration_hours, threshold


# ------------------------------------------------------------
# 7. JONSWAP CALCULATION
# ------------------------------------------------------------

def jonswap_wave_prediction(u10: float, fetch_km: float, duration_hours: float) -> Dict[str, float | str]:
    """
    JONSWAP manual wave-growth calculation.

    Dimensionless parameters:
        F*  = gF / UA^2
        t*  = gt / UA
        H*  = gHm0 / UA^2
        Tp* = gTp / UA

    Fully developed sea:
        Hm0* = 0.243
        Tp*  = 8.13

    Developing sea:
        Hm0* = 0.0016(F*)^0.5
        Tp*  = 0.286(F*)^(1/3)

    Duration-limited sea:
        use Feff* = (t*/68.8)^(3/2) instead of F*.
    """
    g = 9.81
    uA = wind_stress_factor_uA(u10)

    F_m = fetch_km * 1000.0
    t_s = duration_hours * 3600.0

    F_star = g * F_m / (uA ** 2)
    t_star = g * t_s / uA
    F_eff_star_duration = (t_star / 68.8) ** 1.5

    # Fully developed threshold based on both FAS height and period criteria.
    # Corrected: use max(), because both Hm0* and Tp* should be able to reach FAS.
    F_star_fas_by_height = (0.243 / 0.0016) ** 2
    F_star_fas_by_period = (8.13 / 0.286) ** 3
    F_star_fas = max(F_star_fas_by_height, F_star_fas_by_period)

    if F_star >= F_star_fas and F_eff_star_duration >= F_star_fas:
        sea_state = "FAS"
        limiting_star = F_star_fas
        H_star = 0.243
        Tp_star = 8.13
    else:
        if F_star < F_eff_star_duration:
            sea_state = "Fetch limited"
            limiting_star = F_star
        else:
            sea_state = "Duration limited"
            limiting_star = F_eff_star_duration

        H_star = 0.0016 * (limiting_star ** 0.5)
        Tp_star = 0.286 * (limiting_star ** (1.0 / 3.0))

        H_star = min(H_star, 0.243)
        Tp_star = min(Tp_star, 8.13)

    Hm0_m = H_star * (uA ** 2) / g
    Tp_s = Tp_star * uA / g

    return {
        "U10_m_s": u10,
        "UA_m_s": uA,
        "fetch_km": fetch_km,
        "duration_h": duration_hours,
        "F_star": F_star,
        "t_star": t_star,
        "F_eff_star_from_duration": F_eff_star_duration,
        "F_star_FAS_by_height": F_star_fas_by_height,
        "F_star_FAS_by_period": F_star_fas_by_period,
        "F_star_FAS_limit": F_star_fas,
        "limiting_dimensionless_fetch": limiting_star,
        "sea_state": sea_state,
        "Hm0_Hs_m": Hm0_m,
        "Tp_s": Tp_s,
    }


# ------------------------------------------------------------
# 8. MAIN PROGRAM
# ------------------------------------------------------------

def main() -> None:
    print("Loading wind data...")
    df = load_wind_data(NC_FILE)
    print(f"Loaded {len(df):,} hourly records from {df['time'].min()} to {df['time'].max()}.")
    print(f"Years included in analysis: {df['year'].min()} - {df['year'].max()}")

    fetch_table = get_fetch_table()
    fetch_table.to_csv(OUT_DIR / "effective_fetch_table.csv", index=False)

    valid_fetch = fetch_table[fetch_table["effective_fetch_km"] >= FETCH_VALID_THRESHOLD_KM]
    critical_row = valid_fetch.loc[valid_fetch["effective_fetch_km"].idxmax()]
    critical_dir = float(critical_row["bearing_deg"])
    critical_fetch_km = float(critical_row["effective_fetch_km"])

    print("\nCritical fetch direction:")
    print(f"  Direction = {critical_dir:.1f} degrees")
    print(f"  Effective fetch = {critical_fetch_km:.2f} km")

    print("\nPlotting wind rose...")
    plot_wind_rose(df, OUT_DIR / "wind_rose.png")
    print(f"Saved: {OUT_DIR / 'wind_rose.png'}")

    summary_rows = []
    storm_rows_all = []

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year].copy().sort_values("time")

        crit_mask = in_direction_sector(
            year_df["wind_dir_from_deg"],
            critical_dir,
            WIND_DIRECTION_HALF_WIDTH_DEG,
        )
        year_critical_df = year_df[crit_mask].copy()

        if year_critical_df.empty:
            print(f"WARNING: No critical-direction wind data found for {year}.")
            continue

        peak_row = year_critical_df.loc[year_critical_df["wind_speed_m_s"].idxmax()]
        umax = float(peak_row["wind_speed_m_s"])
        peak_time = peak_row["time"]
        peak_dir = float(peak_row["wind_dir_from_deg"])

        storm_df, duration_h, threshold_used = find_storm_covering_peak(
            year_df=year_df,
            peak_time=peak_time,
            critical_dir=critical_dir,
            annual_umax=umax,
        )

        storm_df["year_of_storm_peak"] = year
        storm_df["annual_peak_time"] = peak_time
        storm_rows_all.append(storm_df)

        wave = jonswap_wave_prediction(
            u10=umax,
            fetch_km=critical_fetch_km,
            duration_hours=duration_h,
        )

        summary_rows.append({
            "year": year,
            "critical_direction_deg": critical_dir,
            "critical_fetch_km": critical_fetch_km,
            "max_wind_time": peak_time,
            "max_wind_speed_U10_m_s": umax,
            "max_wind_direction_from_deg": peak_dir,
            "wind_sector_half_width_deg": WIND_DIRECTION_HALF_WIDTH_DEG,
            "storm_start": storm_df["time"].iloc[0],
            "storm_end": storm_df["time"].iloc[-1],
            "storm_duration_h": duration_h,
            "storm_threshold_m_s": threshold_used,
            "UA_m_s": wave["UA_m_s"],
            "F_star": wave["F_star"],
            "t_star": wave["t_star"],
            "F_eff_star_from_duration": wave["F_eff_star_from_duration"],
            "F_star_FAS_by_height": wave["F_star_FAS_by_height"],
            "F_star_FAS_by_period": wave["F_star_FAS_by_period"],
            "F_star_FAS_limit": wave["F_star_FAS_limit"],
            "limiting_dimensionless_fetch": wave["limiting_dimensionless_fetch"],
            "sea_state": wave["sea_state"],
            "Hm0_Hs_m": wave["Hm0_Hs_m"],
            "Tp_s": wave["Tp_s"],
        })

    summary_df = pd.DataFrame(summary_rows)
    storm_data_df = pd.concat(storm_rows_all, ignore_index=True) if storm_rows_all else pd.DataFrame()

    summary_csv = OUT_DIR / "HW3_summary_results_2005_2025.csv"
    storm_csv = OUT_DIR / "HW3_storm_wind_data_2005_2025.csv"
    excel_file = OUT_DIR / "HW3_results_2005_2025.xlsx"

    summary_df.to_csv(summary_csv, index=False)
    storm_data_df.to_csv(storm_csv, index=False)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        fetch_table.to_excel(writer, sheet_name="effective_fetch", index=False)
        summary_df.to_excel(writer, sheet_name="annual_JONSWAP_results", index=False)
        storm_data_df.to_excel(writer, sheet_name="storm_wind_data", index=False)

    print("\nSaved output files:")
    print(f"  {summary_csv}")
    print(f"  {storm_csv}")
    print(f"  {excel_file}")
    print(f"  {OUT_DIR / 'wind_rose.png'}")

    print("\nAnnual JONSWAP summary:")
    print(summary_df[[
        "year",
        "max_wind_time",
        "max_wind_speed_U10_m_s",
        "storm_duration_h",
        "sea_state",
        "Hm0_Hs_m",
        "Tp_s",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
