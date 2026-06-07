

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
OUT_DIR = Path('/mnt/data/CE480_HW5_improved_outputs')
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'
for d in [OUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SITE_LAT = 41.934003
SITE_LON = 28.069026
SITE_NAME = 'Small Commercial Harbor Site (same as HW2/HW3)'
HARBOR_TYPE = 'Small commercial harbor'

# Vessel assumptions
MEAN_SHIP_LENGTH_M = 50.0
MAX_SHIP_LENGTH_M = 80.0
MEAN_DRAFT_M = 4.0
MAX_DRAFT_M = 5.5
MEAN_BEAM_M = 9.0
MAX_BEAM_M = 14.0

# Harbor layout assumptions
QUAY_LENGTH_M = 150.0
TURNING_BASIN_FACTOR = 2.0
ENTRANCE_WIDTH_M = 100.0
CHANNEL_WIDTH_FACTOR = 5.0
PARALLEL_PROTECTION_MARGIN_M = 80.0
PERP_EXTENSION_AFTER_REQ_DEPTH_M = 50.0
ROUND_BW_TO_NEAREST_M = 25.0

# Navigation depth allowances
UNDER_KEEL_CLEARANCE_FACTOR = 0.10
WAVE_ALLOWANCE_M = 0.50
DREDGING_TOLERANCE_M = 0.30
SEDIMENTATION_ALLOWANCE_M = 0.20
DEPTH_ROUNDING_M = 0.10

# Bathymetry points (distance from shoreline/profile origin, positive depth)
BATHY_POINT_1 = (457.2, 5.79)    # 1500 ft, -19 ft
BATHY_POINT_2 = (2027.8, 36.58)  # 1.26 mi, -120 ft

# Wave input from previous HWs
CRITICAL_DIRECTION_DEG = 67.5
EFFECTIVE_FETCH_KM = 1074.12
DEEP_WATER_STEEPNESS = 0.04
RETURN_PERIODS_YEARS = [10, 50, 100]
C1_GRINGORTEN = 0.44
C2_GRINGORTEN = 0.12

EMBEDDED_HW3_ANNUAL_HS = [
    {"year": 2005, "Hs_m": 2.923191}, {"year": 2006, "Hs_m": 0.834905},
    {"year": 2007, "Hs_m": 4.183988}, {"year": 2008, "Hs_m": 3.368033},
    {"year": 2009, "Hs_m": 1.866054}, {"year": 2010, "Hs_m": 3.366278},
    {"year": 2011, "Hs_m": 3.702909}, {"year": 2012, "Hs_m": 1.215482},
    {"year": 2013, "Hs_m": 3.303181}, {"year": 2014, "Hs_m": 5.059716},
    {"year": 2015, "Hs_m": 2.279769}, {"year": 2016, "Hs_m": 3.225712},
    {"year": 2017, "Hs_m": 3.110653}, {"year": 2018, "Hs_m": 7.074669},
    {"year": 2019, "Hs_m": 1.192844}, {"year": 2020, "Hs_m": 3.138614},
    {"year": 2021, "Hs_m": 1.550646}, {"year": 2022, "Hs_m": 0.512366},
    {"year": 2023, "Hs_m": 0.668832}, {"year": 2024, "Hs_m": 6.235831},
    {"year": 2025, "Hs_m": 5.779957},
]

# -----------------------------------------------------------------------------
# CALCULATION HELPERS
# -----------------------------------------------------------------------------
def round_up_to(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def gumbel_reduced_variate(P: float) -> float:
    if not 0 < P < 1:
        raise ValueError('P must be between 0 and 1')
    return -math.log(-math.log(P))


def compute_gumbel_return_heights() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    annual = pd.DataFrame(EMBEDDED_HW3_ANNUAL_HS).sort_values('year').reset_index(drop=True)
    ordered = annual.sort_values('Hs_m', ascending=False).reset_index(drop=True)
    N = len(ordered)
    ordered['rank_m'] = np.arange(1, N + 1)
    ordered['P_non_exceedance'] = 1 - (ordered['rank_m'] - C1_GRINGORTEN) / (N + C2_GRINGORTEN)
    ordered['Q_exceedance'] = 1 - ordered['P_non_exceedance']
    ordered['gumbel_y'] = [gumbel_reduced_variate(P) for P in ordered['P_non_exceedance']]
    a, b = np.polyfit(ordered['gumbel_y'], ordered['Hs_m'], 1)
    ordered['Hs_fit_m'] = a * ordered['gumbel_y'] + b
    ss_res = float(np.sum((ordered['Hs_m'] - ordered['Hs_fit_m']) ** 2))
    ss_tot = float(np.sum((ordered['Hs_m'] - ordered['Hs_m'].mean()) ** 2))
    R2 = 1 - ss_res / ss_tot
    rows = []
    for Tr in RETURN_PERIODS_YEARS:
        P = 1 - 1 / Tr
        y = gumbel_reduced_variate(P)
        H0 = a * y + b
        rows.append({'Tr_year': Tr, 'P_non_exceedance': P, 'Gumbel_y': y, 'H0_m': H0})
    return annual, pd.DataFrame(rows), {'a': float(a), 'b': float(b), 'R2': float(R2), 'N': N}


def required_navigation_depth() -> Dict[str, float]:
    ukc = UNDER_KEEL_CLEARANCE_FACTOR * MAX_DRAFT_M
    raw = MAX_DRAFT_M + ukc + WAVE_ALLOWANCE_M + DREDGING_TOLERANCE_M + SEDIMENTATION_ALLOWANCE_M
    return {
        'design_draft_m': MAX_DRAFT_M,
        'ukc_m': ukc,
        'wave_allowance_m': WAVE_ALLOWANCE_M,
        'dredging_tolerance_m': DREDGING_TOLERANCE_M,
        'sedimentation_allowance_m': SEDIMENTATION_ALLOWANCE_M,
        'required_depth_raw_m': raw,
        'required_depth_m': round_up_to(raw, DEPTH_ROUNDING_M),
    }


def bathymetry_model() -> Dict[str, float]:
    x1, d1 = BATHY_POINT_1
    x2, d2 = BATHY_POINT_2
    slope = (d2 - d1) / (x2 - x1)
    intercept = d1 - slope * x1
    return {'x1_m': x1, 'd1_m': d1, 'x2_m': x2, 'd2_m': d2, 'slope': slope, 'slope_inv': 1/slope, 'intercept': intercept}


def depth_at_distance(x: float, bathy: Dict[str, float]) -> float:
    return bathy['intercept'] + bathy['slope'] * x


def distance_for_depth(depth: float, bathy: Dict[str, float]) -> float:
    return (depth - bathy['intercept']) / bathy['slope']


def dispersion_wavelength(T: float, h: float, g: float = 9.81) -> float:
    L0 = g * T**2 / (2 * math.pi)
    L = L0
    for _ in range(100):
        kh = 2 * math.pi * h / L
        f = L - L0 * math.tanh(kh)
        sech2 = 1 / math.cosh(kh) ** 2
        df = 1 - L0 * sech2 * (-2 * math.pi * h / L**2)
        Lnew = L - f / df
        if abs(Lnew - L) < 1e-10:
            return Lnew
        L = Lnew
    return L


def shoaling_coefficient(T: float, h: float, g: float = 9.81) -> Tuple[float, float, float]:
    L0 = g * T**2 / (2 * math.pi)
    L = dispersion_wavelength(T, h, g)
    k = 2 * math.pi / L
    n = 0.5 * (1 + (2 * k * h) / math.sinh(2 * k * h))
    C = L / T
    Cg = n * C
    C0 = L0 / T
    Cg0 = 0.5 * C0
    Ks = math.sqrt(Cg0 / Cg)
    return Ks, L, L0


def goda_design_wave(H0: float, h: float, bed_slope: float, steepness: float = DEEP_WATER_STEEPNESS) -> Dict[str, float | str]:
    g = 9.81
    L0 = H0 / steepness
    T = math.sqrt(2 * math.pi * L0 / g)
    Ks, L, _ = shoaling_coefficient(T, h, g)
    beta0 = 0.028 * (steepness ** -0.38) * math.exp(20 * (bed_slope ** 1.5))
    beta1 = 0.52 * math.exp(4.2 * bed_slope)
    beta_max = max(0.92, 0.32 * (steepness ** -0.29) * math.exp(2.4 * bed_slope))
    term_shoaling = Ks * H0
    term_depth = beta0 * H0 + beta1 * h
    term_max = beta_max * H0
    Hd = min(term_shoaling, term_depth, term_max)
    if Hd == term_shoaling:
        governing = 'Shoaling limit'
    elif Hd == term_depth:
        governing = 'Depth-limited breaking'
    else:
        governing = 'Maximum breaker limit'
    return {
        'H0_m': H0, 's0': steepness, 'L0_m': L0, 'T_s': T, 'h_m': h, 'L_m': L, 'Ks': Ks,
        'bed_slope': bed_slope, 'slope_inv': 1/bed_slope,
        'beta0': beta0, 'beta1': beta1, 'beta_max': beta_max,
        'KsH0_m': term_shoaling, 'depth_term_m': term_depth, 'max_breaker_term_m': term_max,
        'Hd_m': Hd, 'governing': governing
    }


def compute_all():
    annual, ret, fit = compute_gumbel_return_heights()
    nav = required_navigation_depth()
    bathy = bathymetry_model()
    d_req = nav['required_depth_m']
    dist_req = distance_for_depth(d_req, bathy)
    bw_perp = round_up_to(dist_req + PERP_EXTENSION_AFTER_REQ_DEPTH_M, ROUND_BW_TO_NEAREST_M)
    toe_depth = depth_at_distance(bw_perp, bathy)
    turning_d = TURNING_BASIN_FACTOR * MAX_SHIP_LENGTH_M
    channel_w = CHANNEL_WIDTH_FACTOR * MAX_BEAM_M
    bw_parallel = round_up_to(QUAY_LENGTH_M + turning_d + PARALLEL_PROTECTION_MARGIN_M, ROUND_BW_TO_NEAREST_M)
    bw_total = bw_perp + bw_parallel
    goda = []
    for _, r in ret.iterrows():
        row = goda_design_wave(float(r['H0_m']), toe_depth, bathy['slope'])
        row['Tr_year'] = int(r['Tr_year'])
        goda.append(row)
    goda_df = pd.DataFrame(goda)
    summary = {
        'dist_req_m': dist_req, 'bw_perp_m': bw_perp, 'toe_depth_m': toe_depth,
        'turning_d_m': turning_d, 'channel_w_m': channel_w, 'bw_parallel_m': bw_parallel,
        'bw_total_m': bw_total
    }
    return annual, ret, fit, nav, bathy, goda_df, summary

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.25,
    })


def fig_bathymetry(bathy, nav, summary) -> Path:
    x = np.linspace(0, 2300, 400)
    d = bathy['intercept'] + bathy['slope'] * x
    d = np.maximum(d, 0)
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    ax.fill_between(x, d, d.max()+4, color='#d8ecff', alpha=0.9, label='Water column')
    ax.fill_between(x, 0, d, color='#c6a77b', alpha=0.5, label='Seabed')
    ax.plot(x, d, color='#174a7c', lw=2.5, label='Linear bathymetry fit')
    ax.scatter([bathy['x1_m'], bathy['x2_m']], [bathy['d1_m'], bathy['d2_m']],
               color='#b23a48', s=70, zorder=5, label='Google Earth Pro points')
    ax.axhline(nav['required_depth_m'], color='#333333', ls='--', lw=1.5, label=f"Required depth = {nav['required_depth_m']:.1f} m")
    ax.axvline(summary['dist_req_m'], color='#6a4c93', ls='--', lw=1.5, label=f"Required depth at {summary['dist_req_m']:.0f} m")
    ax.axvline(summary['bw_perp_m'], color='#008000', ls=':', lw=2.5, label=f"Selected BW head = {summary['bw_perp_m']:.0f} m")
    ax.scatter([summary['bw_perp_m']], [summary['toe_depth_m']], marker='x', s=110, color='#008000', lw=2)
    ax.annotate(f"BW toe depth\n{summary['toe_depth_m']:.1f} m", xy=(summary['bw_perp_m'], summary['toe_depth_m']),
                xytext=(summary['bw_perp_m']+120, summary['toe_depth_m']+4),
                arrowprops=dict(arrowstyle='->', color='#008000'), color='#004d00')
    ax.invert_yaxis()
    ax.set_xlabel('Distance offshore along profile (m)')
    ax.set_ylabel('Water depth (m, positive downward)')
    ax.set_title('Bathymetry profile, required depth, and selected breakwater reach')
    ax.text(0.02, 0.05, f"Estimated seabed slope m = {bathy['slope']:.4f} (approx. 1:{bathy['slope_inv']:.0f})",
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='0.7'))
    ax.legend(loc='lower right', ncol=2)
    fig.tight_layout()
    p = FIG_DIR / 'fig_1_bathymetry_profile_enhanced.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_layout(summary) -> Path:
    perp = summary['bw_perp_m']
    parallel = summary['bw_parallel_m']
    turning_d = summary['turning_d_m']
    channel_w = summary['channel_w_m']

    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_aspect('equal')
    # backgrounds
    ax.add_patch(Rectangle((-180, -170), 1250, 170, facecolor='#e8d9c5', edgecolor='none', alpha=0.9))
    ax.add_patch(Rectangle((-180, 0), 1250, 520, facecolor='#d9efff', edgecolor='none', alpha=0.95))
    ax.plot([-180, 1070], [0, 0], color='k', lw=1.8)
    ax.text(-150, -55, 'Land', fontsize=11)
    ax.text(-150, 420, 'Sea', fontsize=11, color='#174a7c')

    # breakwater
    ax.plot([0, perp], [0, 0], lw=10, color='#585858', solid_capstyle='round')
    ax.plot([perp, perp], [0, parallel], lw=10, color='#585858', solid_capstyle='round')
    ax.text(perp/2, -36, f'Perpendicular BW = {perp:.0f} m', ha='center', fontsize=9)
    ax.text(perp+25, parallel/2, f'Parallel BW = {parallel:.0f} m', rotation=90, va='center', fontsize=9)

    # channel
    ch_x = perp + 95
    ax.plot([ch_x, ch_x], [-120, 475], ls='--', color='#0066aa', lw=2)
    ax.add_patch(Rectangle((ch_x-channel_w/2, -120), channel_w, 595, facecolor='#4da6ff', alpha=0.18, edgecolor='#0066aa', lw=1.2, ls='--'))
    ax.text(ch_x+channel_w/2+10, 310, f'Channel width\n{channel_w:.0f} m', color='#005080', fontsize=9)

    # quay wall
    quay_x = max(100, perp - 230)
    quay_y0 = 95
    ax.plot([quay_x, quay_x], [quay_y0, quay_y0+QUAY_LENGTH_M], lw=7, color='#8b4513', solid_capstyle='butt')
    ax.text(quay_x-70, quay_y0+QUAY_LENGTH_M/2, f'Quay wall\n{QUAY_LENGTH_M:.0f} m', ha='center', va='center', fontsize=9)

    # turning basin
    center = (quay_x + 120, quay_y0 + QUAY_LENGTH_M/2)
    circ = Circle(center, turning_d/2, fill=False, lw=2.2, ls='--', color='#7b2cbf')
    ax.add_patch(circ)
    ax.text(center[0], center[1], f'Turning basin\nD = {turning_d:.0f} m', ha='center', va='center', fontsize=9, color='#4b0082')

    # entrance arrow
    ax.add_patch(FancyArrowPatch((ch_x, 430), (perp+15, parallel-10), arrowstyle='->', mutation_scale=15, lw=1.5, color='#d00000'))
    ax.text(ch_x+20, 410, 'Harbor approach', color='#d00000', fontsize=9)

    # dimension summary box
    text = (
        f"Design vessel: Lmax={MAX_SHIP_LENGTH_M:.0f} m, Bmax={MAX_BEAM_M:.0f} m, draft={MAX_DRAFT_M:.1f} m\n"
        f"Required depth={required_navigation_depth()['required_depth_m']:.1f} m; toe depth={summary['toe_depth_m']:.1f} m\n"
        f"Total preliminary BW length={summary['bw_total_m']:.0f} m"
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va='top', bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.92))
    ax.set_xlim(-180, 1070)
    ax.set_ylim(-160, 520)
    ax.set_xlabel('Schematic x-distance (m)')
    ax.set_ylabel('Schematic y-distance (m)')
    ax.set_title('Preliminary small commercial harbor layout schematic')
    fig.tight_layout()
    p = FIG_DIR / 'fig_2_harbor_layout_enhanced.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_navigation_depth(nav) -> Path:
    labels = ['Draft', 'UKC', 'Wave\nallow.', 'Dredging\ntol.', 'Sedim.\nallow.']
    vals = [nav['design_draft_m'], nav['ukc_m'], nav['wave_allowance_m'], nav['dredging_tolerance_m'], nav['sedimentation_allowance_m']]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bottom = 0
    colors = ['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b']
    for val, lab, c in zip(vals, labels, colors):
        ax.bar([0], [val], bottom=bottom, width=0.45, label=f'{lab}: {val:.2f} m', color=c)
        ax.text(0, bottom + val/2, f'{val:.2f}', va='center', ha='center', color='white', fontweight='bold')
        bottom += val
    ax.axhline(nav['required_depth_m'], color='black', ls='--', lw=1.5, label=f"Rounded design depth = {nav['required_depth_m']:.1f} m")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([0])
    ax.set_xticklabels(['Navigation\ndepth'])
    ax.set_ylabel('Depth component (m)')
    ax.set_title('Required navigation depth components')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    p = FIG_DIR / 'fig_3_navigation_depth_components.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_gumbel(annual, ret, fit) -> Path:
    ordered = annual.sort_values('Hs_m', ascending=False).reset_index(drop=True)
    N = len(ordered)
    ordered['rank_m'] = np.arange(1, N+1)
    ordered['P_non_exceedance'] = 1 - (ordered['rank_m'] - C1_GRINGORTEN) / (N + C2_GRINGORTEN)
    ordered['gumbel_y'] = [gumbel_reduced_variate(P) for P in ordered['P_non_exceedance']]
    xline = np.linspace(ordered['gumbel_y'].min()-0.3, ret['Gumbel_y'].max()+0.3, 300)
    yline = fit['a'] * xline + fit['b']
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(ordered['gumbel_y'], ordered['Hs_m'], s=60, color='#1f77b4', edgecolor='k', linewidth=0.4, label='Annual maximum Hs')
    ax.plot(xline, yline, color='#d62728', lw=2.3, label=fr"Gumbel fit: $H_s={fit['a']:.3f}y+{fit['b']:.3f}$, $R^2={fit['R2']:.2f}$")
    ax.scatter(ret['Gumbel_y'], ret['H0_m'], marker='X', s=120, color='#2ca02c', edgecolor='k', label='Return-period H0')
    for _, r in ret.iterrows():
        ax.annotate(f"{int(r['Tr_year'])}-yr\n{r['H0_m']:.2f} m", xy=(r['Gumbel_y'], r['H0_m']), xytext=(9, -10), textcoords='offset points', fontsize=8)
    ax.set_xlabel(r'Gumbel reduced variate, $y=-\ln[-\ln(P)]$')
    ax.set_ylabel(r'Significant wave height, $H_s$ (m)')
    ax.set_title('Extreme wave return heights from HW4 Gumbel analysis')
    ax.legend(loc='upper left')
    fig.tight_layout()
    p = FIG_DIR / 'fig_4_gumbel_return_heights_enhanced.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_annual_hs(annual, ret) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(annual['year'], annual['Hs_m'], marker='o', lw=1.8, color='#1f77b4', label='Annual Hs maxima')
    for _, r in ret[ret['Tr_year'].isin([10,50])].iterrows():
        ax.axhline(r['H0_m'], ls='--', lw=1.7, label=f"{int(r['Tr_year'])}-yr H0 = {r['H0_m']:.2f} m")
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual maximum significant wave height (m)')
    ax.set_title('Annual maximum wave heights used in the return-period analysis')
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / 'fig_5_annual_hs_timeseries.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_goda_candidates(goda) -> Path:
    labels = [f"{int(t)}-yr" for t in goda['Tr_year']]
    x = np.arange(len(labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x-w, goda['KsH0_m'], width=w, label=r'$K_sH_0$ shoaling term', color='#4c78a8')
    ax.bar(x, goda['depth_term_m'], width=w, label=r'$\beta_0H_0+\beta_1h$ depth term', color='#f58518')
    ax.bar(x+w, goda['max_breaker_term_m'], width=w, label=r'$\beta_{max}H_0$ breaker term', color='#54a24b')
    ax.plot(x, goda['Hd_m'], marker='D', color='#d62728', lw=2, label='Selected design wave (minimum)')
    for xi, hd in zip(x, goda['Hd_m']):
        ax.text(xi, hd+0.12, f'{hd:.2f} m', ha='center', color='#b00020', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Wave height candidate (m)')
    ax.set_title('Goda-type candidate terms and selected design wave height')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    p = FIG_DIR / 'fig_6_goda_candidate_terms_enhanced.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_wave_periods(goda) -> Path:
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(goda))
    labels = [f"{int(t)}-yr" for t in goda['Tr_year']]
    ax1.plot(x, goda['H0_m'], marker='o', color='#1f77b4', lw=2, label='Deep-water H0')
    ax1.plot(x, goda['Hd_m'], marker='s', color='#d62728', lw=2, label='Goda Hd')
    ax1.set_ylabel('Wave height (m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(x, goda['T_s'], marker='^', color='#2ca02c', lw=2, label='Wave period T')
    ax2.set_ylabel('Wave period (s)')
    ax1.set_title('Deep-water waves, local design waves, and periods')
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='upper left')
    fig.tight_layout()
    p = FIG_DIR / 'fig_7_wave_height_period_summary.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p

# -----------------------------------------------------------------------------
# TABLES AND LATEX REPORT
# -----------------------------------------------------------------------------
def save_tables(annual, ret, nav, bathy, goda, summary):
    # CSVs for transparency
    annual.to_csv(TABLE_DIR / 'annual_hs.csv', index=False)
    ret.to_csv(TABLE_DIR / 'return_wave_heights.csv', index=False)
    pd.DataFrame([
        ['Mean ship length', MEAN_SHIP_LENGTH_M, 'm'],
        ['Maximum/design ship length', MAX_SHIP_LENGTH_M, 'm'],
        ['Mean draft', MEAN_DRAFT_M, 'm'],
        ['Maximum/design draft', MAX_DRAFT_M, 'm'],
        ['Mean beam', MEAN_BEAM_M, 'm'],
        ['Maximum/design beam', MAX_BEAM_M, 'm'],
        ['Quay wall length', QUAY_LENGTH_M, 'm'],
        ['Turning basin diameter', summary['turning_d_m'], 'm'],
        ['One-way channel width', summary['channel_w_m'], 'm'],
    ], columns=['Item','Value','Unit']).to_csv(TABLE_DIR / 'vessel_layout_assumptions.csv', index=False)
    pd.DataFrame([
        ['Design draft', nav['design_draft_m']],
        ['Under-keel clearance', nav['ukc_m']],
        ['Wave allowance', nav['wave_allowance_m']],
        ['Dredging tolerance', nav['dredging_tolerance_m']],
        ['Sedimentation allowance', nav['sedimentation_allowance_m']],
        ['Required depth before rounding', nav['required_depth_raw_m']],
        ['Required design depth', nav['required_depth_m']],
    ], columns=['Component','Value_m']).to_csv(TABLE_DIR / 'navigation_depth.csv', index=False)
    pd.DataFrame([
        ['Bathymetry point 1 distance', bathy['x1_m'], 'm'],
        ['Bathymetry point 1 depth', bathy['d1_m'], 'm'],
        ['Bathymetry point 2 distance', bathy['x2_m'], 'm'],
        ['Bathymetry point 2 depth', bathy['d2_m'], 'm'],
        ['Bathymetry slope', bathy['slope'], 'm/m'],
        ['Bathymetry slope approx.', bathy['slope_inv'], '1:x'],
        ['Distance to required depth', summary['dist_req_m'], 'm'],
        ['Perpendicular breakwater length', summary['bw_perp_m'], 'm'],
        ['Depth at breakwater head/toe', summary['toe_depth_m'], 'm'],
        ['Parallel breakwater length', summary['bw_parallel_m'], 'm'],
        ['Total preliminary breakwater length', summary['bw_total_m'], 'm'],
    ], columns=['Item','Value','Unit']).to_csv(TABLE_DIR / 'bathymetry_breakwater_summary.csv', index=False)
    goda.to_csv(TABLE_DIR / 'goda_results.csv', index=False)


def latex_table_rows(rows, fmt_specs=None):
    lines = []
    for row in rows:
        out = []
        for v in row:
            if isinstance(v, float):
                out.append(f'{v:.2f}')
            else:
                out.append(str(v))
        lines.append(' & '.join(out) + r' \\')
    return '\n'.join(lines)


def write_latex(annual, ret, fit, nav, bathy, goda, summary, figs: Dict[str, Path]) -> Path:
    h10 = float(ret.loc[ret['Tr_year'] == 10, 'H0_m'].iloc[0])
    h50 = float(ret.loc[ret['Tr_year'] == 50, 'H0_m'].iloc[0])
    hd10 = float(goda.loc[goda['Tr_year'] == 10, 'Hd_m'].iloc[0])
    hd50 = float(goda.loc[goda['Tr_year'] == 50, 'Hd_m'].iloc[0])
    t10 = float(goda.loc[goda['Tr_year'] == 10, 'T_s'].iloc[0])
    t50 = float(goda.loc[goda['Tr_year'] == 50, 'T_s'].iloc[0])
    slope_str = f"1:{bathy['slope_inv']:.0f}"

    def rel(p: Path) -> str:
        return str(p.relative_to(OUT_DIR)).replace('\\', '/')

    tex = rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1.9cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\hypersetup{{colorlinks=true, linkcolor=blue!50!black, urlcolor=blue!50!black, citecolor=blue!50!black}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}
\captionsetup{{font=small, labelfont=bf}}

\begin{{document}}

\begin{{titlepage}}
\centering
\vspace*{{1.8cm}}
{{\Large \textbf{{CE480 - HW5}}\\[0.3cm]}}
{{\LARGE \textbf{{Preliminary Small Commercial Harbor Layout and Breakwater Design}}\\[0.8cm]}}
\rule{{0.85\textwidth}}{{0.6pt}}\\[0.6cm]
\begin{{tabular}}{{ll}}
\textbf{{Site}} & {SITE_NAME} \\
\textbf{{Coordinates}} & {SITE_LAT:.6f}$^\circ$, {SITE_LON:.6f}$^\circ$ \\
\textbf{{Harbor type}} & {HARBOR_TYPE} \\
\textbf{{Critical wave direction}} & {CRITICAL_DIRECTION_DEG:.1f}$^\circ$ \\
\textbf{{Effective fetch}} & {EFFECTIVE_FETCH_KM:.2f} km \\
\end{{tabular}}
\vfill
\textit{{Prepared from CE480 HW2, HW3, HW4 results and the confirmed HW5 assumptions.}}\\[0.5cm]
\end{{titlepage}}

\tableofcontents
\newpage

\section{{Executive Summary}}
This report presents a preliminary harbor-layout and breakwater-design calculation for a small commercial harbor at the same site used in the previous CE480 homeworks. The design uses a maximum vessel length of {MAX_SHIP_LENGTH_M:.0f} m, maximum draft of {MAX_DRAFT_M:.1f} m, and maximum beam of {MAX_BEAM_M:.0f} m. A quay wall length of {QUAY_LENGTH_M:.0f} m is adopted based on similar small commercial harbor scales.

The calculated navigation depth is \textbf{{{nav['required_depth_m']:.1f} m}}. The estimated bathymetry slope from the Google Earth Pro profile is \textbf{{{bathy['slope']:.4f}}} m/m, approximately \textbf{{{slope_str}}}. The required depth is reached at approximately \textbf{{{summary['dist_req_m']:.0f} m}} from the shore. Therefore, the selected perpendicular breakwater extension is \textbf{{{summary['bw_perp_m']:.0f} m}}, giving a local water depth of \textbf{{{summary['toe_depth_m']:.1f} m}} at the breakwater head/toe. The parallel arm is estimated as \textbf{{{summary['bw_parallel_m']:.0f} m}}, so the total preliminary breakwater length is \textbf{{{summary['bw_total_m']:.0f} m}}.

From HW4, the 10-year and 50-year deep-water design wave heights are \textbf{{{h10:.2f} m}} and \textbf{{{h50:.2f} m}}, respectively. Using the course-style Goda nearshore design-wave check at the selected breakwater depth, the corresponding design wave heights in front of the breakwater are \textbf{{{hd10:.2f} m}} and \textbf{{{hd50:.2f} m}}. The governing condition is depth-limited breaking.

\section{{Problem Statement and Methodology}}
The homework requires the harbor dimensions and wave-design checks listed below:
\begin{{enumerate}}[leftmargin=1.0cm]
    \item Select representative maximum and mean ship sizes for the selected harbor type.
    \item Estimate quay wall or pier length from similar harbor scale.
    \item Calculate the turning basin.
    \item Determine the breakwater extension perpendicular and parallel to shore.
    \item Determine the necessary water depth using a navigation-channel depth formula.
    \item Check whether the breakwater reaches the required depth; if not, propose a navigation channel.
    \item Use HW4 10-year and 50-year deep-water wave heights and calculate the design wave height in front of the breakwater using Goda's formula with wave steepness $s_0=0.04$.
\end{{enumerate}}

The workflow is summarized as follows:
\[
\text{{Ship size}} \rightarrow \text{{quay and basin dimensions}} \rightarrow \text{{navigation depth}} \rightarrow \text{{breakwater reach}} \rightarrow \text{{Goda design wave}}
\]

\section{{Input Data and Assumptions}}
\subsection{{Site and wave inputs}}
The site and wave inputs are taken from the previous CE480 homework workflow. The critical wave direction is the direction with the largest effective fetch. The fetch and wave-return values are treated as fixed inputs for this HW5 preliminary design.

\begin{{table}}[H]
\centering
\caption{{Main input data used in HW5.}}
\begin{{tabular}}{{lll}}
\toprule
\textbf{{Input}} & \textbf{{Value}} & \textbf{{Unit}} \\
\midrule
Site latitude & {SITE_LAT:.6f} & deg \\
Site longitude & {SITE_LON:.6f} & deg \\
Harbor type & {HARBOR_TYPE} & - \\
Critical wave direction & {CRITICAL_DIRECTION_DEG:.1f} & deg \\
Effective fetch & {EFFECTIVE_FETCH_KM:.2f} & km \\
Deep-water wave steepness, $s_0$ & {DEEP_WATER_STEEPNESS:.2f} & - \\
Bathymetry point 1 & {BATHY_POINT_1[0]:.1f} m, depth {BATHY_POINT_1[1]:.2f} m & - \\
Bathymetry point 2 & {BATHY_POINT_2[0]:.1f} m, depth {BATHY_POINT_2[1]:.2f} m & - \\
Estimated seabed slope & {bathy['slope']:.4f} & m/m \\
Approximate seabed slope & {slope_str} & - \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Design vessel and quay assumptions}}
For a small commercial harbor, the design vessel was selected as a small general-cargo/coaster-type vessel. Since no specific vessel fleet was assigned in the homework statement, the values below are preliminary assumptions and should be replaced if the instructor requires a specific vessel type.

\begin{{table}}[H]
\centering
\caption{{Design vessel, quay, and harbor-basin assumptions.}}
\begin{{tabular}}{{lll}}
\toprule
\textbf{{Item}} & \textbf{{Value}} & \textbf{{Unit}} \\
\midrule
Mean ship length & {MEAN_SHIP_LENGTH_M:.0f} & m \\
Maximum/design ship length, $L_{{max}}$ & {MAX_SHIP_LENGTH_M:.0f} & m \\
Mean draft & {MEAN_DRAFT_M:.1f} & m \\
Maximum/design draft, $T_d$ & {MAX_DRAFT_M:.1f} & m \\
Mean beam & {MEAN_BEAM_M:.0f} & m \\
Maximum/design beam, $B_{{max}}$ & {MAX_BEAM_M:.0f} & m \\
Quay wall length & {QUAY_LENGTH_M:.0f} & m \\
Entrance width & {ENTRANCE_WIDTH_M:.0f} & m \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Turning Basin, Channel Width, and Quay Length}}
The turning basin diameter is taken as twice the maximum design vessel length:
\begin{{equation}}
D_T = 2L_{{max}}
\end{{equation}}
\begin{{equation}}
D_T = 2({MAX_SHIP_LENGTH_M:.0f}) = {summary['turning_d_m']:.0f}\ \mathrm{{m}}
\end{{equation}}

The one-way preliminary navigation channel width is estimated as:
\begin{{equation}}
B_c = 5B_{{max}}
\end{{equation}}
\begin{{equation}}
B_c = 5({MAX_BEAM_M:.0f}) = {summary['channel_w_m']:.0f}\ \mathrm{{m}}
\end{{equation}}

The adopted quay wall length is {QUAY_LENGTH_M:.0f} m. This provides berthing space for approximately one small commercial vessel with operational allowance, or a shorter combination of small service vessels.

\section{{Navigation Depth Calculation}}
The required navigation depth is computed by adding the design draft, under-keel clearance, wave allowance, dredging tolerance, and sedimentation allowance:
\begin{{equation}}
D_{{req}} = T_d + UKC + A_w + A_d + A_s
\end{{equation}}
where $T_d$ is the design vessel draft, $UKC$ is under-keel clearance, $A_w$ is wave allowance, $A_d$ is dredging tolerance, and $A_s$ is sedimentation allowance.

The under-keel clearance is assumed as 10\% of the design draft:
\begin{{equation}}
UKC = 0.10T_d = 0.10({MAX_DRAFT_M:.1f}) = {nav['ukc_m']:.2f}\ \mathrm{{m}}
\end{{equation}}
Therefore,
\begin{{equation}}
D_{{req}} = {MAX_DRAFT_M:.1f} + {nav['ukc_m']:.2f} + {WAVE_ALLOWANCE_M:.2f} + {DREDGING_TOLERANCE_M:.2f} + {SEDIMENTATION_ALLOWANCE_M:.2f} = {nav['required_depth_raw_m']:.2f}\ \mathrm{{m}}
\end{{equation}}
After rounding upward, the adopted required navigation depth is:
\begin{{equation}}
\boxed{{D_{{req}} = {nav['required_depth_m']:.1f}\ \mathrm{{m}}}}
\end{{equation}}

\begin{{table}}[H]
\centering
\caption{{Navigation depth calculation.}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Component}} & \textbf{{Value (m)}} \\
\midrule
Design draft & {nav['design_draft_m']:.2f} \\
Under-keel clearance (10\% draft) & {nav['ukc_m']:.2f} \\
Wave allowance & {nav['wave_allowance_m']:.2f} \\
Dredging tolerance & {nav['dredging_tolerance_m']:.2f} \\
Sedimentation allowance & {nav['sedimentation_allowance_m']:.2f} \\
Required depth before rounding & {nav['required_depth_raw_m']:.2f} \\
\textbf{{Required design depth}} & \textbf{{{nav['required_depth_m']:.2f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{{rel(figs['nav'])}}}
\caption{{Components of the required navigation depth.}}
\end{{figure}}

\section{{Bathymetry, Required Depth, and Breakwater Reach}}
The Google Earth Pro profile points were converted from English units to SI units. A linear bathymetry model was fitted between the two points:
\begin{{equation}}
h(x)=h_0+mx
\end{{equation}}
where $x$ is offshore distance and $h$ is water depth. The estimated slope is:
\begin{{equation}}
m = \frac{{36.58-5.79}}{{2027.8-457.2}} = {bathy['slope']:.4f}\ \approx {slope_str}
\end{{equation}}

The distance where the required depth is reached is:
\begin{{equation}}
x_{{req}} = \frac{{D_{{req}}-h_0}}{{m}} = {summary['dist_req_m']:.0f}\ \mathrm{{m}}
\end{{equation}}

To ensure the breakwater reaches slightly deeper water than the minimum navigation requirement, an additional {PERP_EXTENSION_AFTER_REQ_DEPTH_M:.0f} m is added and the result is rounded up:
\begin{{equation}}
L_{{BW,\perp}} = \mathrm{{round\ up}}(x_{{req}}+{PERP_EXTENSION_AFTER_REQ_DEPTH_M:.0f}) = {summary['bw_perp_m']:.0f}\ \mathrm{{m}}
\end{{equation}}
The resulting depth at the breakwater head/toe is approximately:
\begin{{equation}}
h_{{toe}} = {summary['toe_depth_m']:.1f}\ \mathrm{{m}}
\end{{equation}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\textwidth]{{{rel(figs['bathy'])}}}
\caption{{Bathymetry profile, required navigation depth, and selected breakwater head distance.}}
\end{{figure}}

\section{{Breakwater Layout}}
The preliminary breakwater layout is composed of two parts:
\begin{{enumerate}}[leftmargin=1.0cm]
    \item A perpendicular arm extending from shore until the required water depth is exceeded.
    \item A parallel arm protecting the quay and turning basin from the critical wave direction.
\end{{enumerate}}

The parallel breakwater length is estimated from the quay length, turning basin diameter, and an additional protection margin:
\begin{{equation}}
L_{{BW,\parallel}} = L_q + D_T + M_p
\end{{equation}}
\begin{{equation}}
L_{{BW,\parallel}} = {QUAY_LENGTH_M:.0f} + {summary['turning_d_m']:.0f} + {PARALLEL_PROTECTION_MARGIN_M:.0f} = {QUAY_LENGTH_M + summary['turning_d_m'] + PARALLEL_PROTECTION_MARGIN_M:.0f}\ \mathrm{{m}} \approx {summary['bw_parallel_m']:.0f}\ \mathrm{{m}}
\end{{equation}}
Thus, the total preliminary breakwater length is:
\begin{{equation}}
L_{{BW,total}} = {summary['bw_perp_m']:.0f}+{summary['bw_parallel_m']:.0f}={summary['bw_total_m']:.0f}\ \mathrm{{m}}
\end{{equation}}

\begin{{table}}[H]
\centering
\caption{{Summary of harbor layout dimensions.}}
\begin{{tabular}}{{lrl}}
\toprule
\textbf{{Item}} & \textbf{{Value}} & \textbf{{Unit}} \\
\midrule
Turning basin diameter & {summary['turning_d_m']:.0f} & m \\
Channel width & {summary['channel_w_m']:.0f} & m \\
Required navigation depth & {nav['required_depth_m']:.1f} & m \\
Distance to required depth & {summary['dist_req_m']:.0f} & m from shoreline \\
Perpendicular breakwater length & {summary['bw_perp_m']:.0f} & m \\
Depth at breakwater head/toe & {summary['toe_depth_m']:.1f} & m \\
Parallel breakwater length & {summary['bw_parallel_m']:.0f} & m \\
\textbf{{Total breakwater length}} & \textbf{{{summary['bw_total_m']:.0f}}} & \textbf{{m}} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\textwidth]{{{rel(figs['layout'])}}}
\caption{{Preliminary harbor layout schematic. The figure is a conceptual layout, not a georeferenced engineering drawing.}}
\end{{figure}}

\section{{Deep-Water Design Waves from HW4}}
The deep-water return wave heights were obtained using the Gumbel extreme-value method. The plotting position used in the previous HW4 code is the Gringorten form:
\begin{{equation}}
P = 1 - \frac{{m-C_1}}{{N+C_2}}, \quad C_1=0.44, \quad C_2=0.12
\end{{equation}}
where $m$ is rank and $N$ is the number of annual maxima. The Gumbel reduced variate is:
\begin{{equation}}
y=-\ln[-\ln(P)]
\end{{equation}}
The fitted relation is:
\begin{{equation}}
H_s = ay+b = {fit['a']:.4f}y + {fit['b']:.4f}, \quad R^2={fit['R2']:.3f}
\end{{equation}}

For a return period $T_R$, the exceedance probability is $Q=1/T_R$ and the non-exceedance probability is $P=1-Q$.

\begin{{table}}[H]
\centering
\caption{{Deep-water return-period wave heights used in HW5.}}
\begin{{tabular}}{{rrrr}}
\toprule
\textbf{{$T_R$ (yr)}} & \textbf{{$P$}} & \textbf{{$y$}} & \textbf{{$H_0$ (m)}} \\
\midrule
"""
    for _, r in ret.iterrows():
        tex += f"{int(r['Tr_year'])} & {r['P_non_exceedance']:.3f} & {r['Gumbel_y']:.3f} & {r['H0_m']:.2f} \\\\\n"
    tex += rf"""\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\textwidth]{{{rel(figs['annual'])}}}
\caption{{Annual maximum significant wave heights and 10-year/50-year return wave levels.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\textwidth]{{{rel(figs['gumbel'])}}}
\caption{{Gumbel fit and return-period wave heights.}}
\end{{figure}}

\section{{Goda Design Wave Height at the Breakwater}}
The design wave height in front of the breakwater is evaluated at the local depth $h={summary['toe_depth_m']:.1f}$ m. The wave steepness is given by the homework statement as:
\begin{{equation}}
s_0=\frac{{H_0}}{{L_0}}=0.04
\end{{equation}}
Therefore, for each return-period wave:
\begin{{equation}}
L_0=\frac{{H_0}}{{s_0}}
\end{{equation}}
and the wave period is obtained from the deep-water dispersion relation:
\begin{{equation}}
L_0=\frac{{gT^2}}{{2\pi}} \quad \Rightarrow \quad T=\sqrt{{\frac{{2\pi L_0}}{{g}}}}
\end{{equation}}

The course-style Goda nearshore wave-height check is implemented as:
\begin{{equation}}
H_d = \min\left(K_sH_0,\ \beta_0H_0+\beta_1h,\ \beta_{{max}}H_0\right)
\end{{equation}}
where $K_s$ is the shoaling coefficient, $h$ is the local water depth, and the $\beta$ coefficients depend on wave steepness and seabed slope. The coefficient set used in the calculation is:
\begin{{align}}
\beta_0 &= 0.028s_0^{{-0.38}}\exp(20m^{{1.5}}) \\
\beta_1 &= 0.52\exp(4.2m) \\
\beta_{{max}} &= \max\left[0.92,\ 0.32s_0^{{-0.29}}\exp(2.4m)\right]
\end{{align}}
where $m={bathy['slope']:.4f}$ is the seabed slope. The selected design wave is the smallest of the three candidate terms because it represents the limiting transformation/breaking condition at the breakwater location.

\begin{{table}}[H]
\centering
\caption{{Goda-type design wave calculation results.}}
\begin{{tabular}}{{rrrrrl}}
\toprule
\textbf{{$T_R$ (yr)}} & \textbf{{$H_0$ (m)}} & \textbf{{$T$ (s)}} & \textbf{{$h$ (m)}} & \textbf{{$H_d$ (m)}} & \textbf{{Governing condition}} \\
\midrule
"""
    for _, r in goda.iterrows():
        tex += f"{int(r['Tr_year'])} & {r['H0_m']:.2f} & {r['T_s']:.2f} & {r['h_m']:.2f} & {r['Hd_m']:.2f} & {r['governing']} \\\\\n"
    tex += rf"""\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Candidate terms in Goda design wave calculation.}}
\begin{{tabular}}{{rrrrr}}
\toprule
\textbf{{$T_R$ (yr)}} & \textbf{{$K_sH_0$ (m)}} & \textbf{{$\beta_0H_0+\beta_1h$ (m)}} & \textbf{{$\beta_{{max}}H_0$ (m)}} & \textbf{{Selected $H_d$ (m)}} \\
\midrule
"""
    for _, r in goda.iterrows():
        tex += f"{int(r['Tr_year'])} & {r['KsH0_m']:.2f} & {r['depth_term_m']:.2f} & {r['max_breaker_term_m']:.2f} & {r['Hd_m']:.2f} \\\\\n"
    tex += rf"""\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.94\textwidth]{{{rel(figs['goda'])}}}
\caption{{Goda-type candidate terms and selected design wave height.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.90\textwidth]{{{rel(figs['wave_summary'])}}}
\caption{{Comparison of deep-water wave height, local design wave height, and wave period.}}
\end{{figure}}

\section{{Final Design Summary}}
\begin{{table}}[H]
\centering
\caption{{Final preliminary design values.}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Design item}} & \textbf{{Adopted/calculated value}} \\
\midrule
Harbor type & {HARBOR_TYPE} \\
Maximum/design vessel length & {MAX_SHIP_LENGTH_M:.0f} m \\
Maximum/design draft & {MAX_DRAFT_M:.1f} m \\
Maximum/design beam & {MAX_BEAM_M:.0f} m \\
Quay wall length & {QUAY_LENGTH_M:.0f} m \\
Turning basin diameter & {summary['turning_d_m']:.0f} m \\
Navigation channel width & {summary['channel_w_m']:.0f} m \\
Required navigation depth & {nav['required_depth_m']:.1f} m \\
Perpendicular breakwater length & {summary['bw_perp_m']:.0f} m \\
Parallel breakwater length & {summary['bw_parallel_m']:.0f} m \\
Total preliminary breakwater length & {summary['bw_total_m']:.0f} m \\
10-year deep-water wave height, $H_{{0,10}}$ & {h10:.2f} m \\
50-year deep-water wave height, $H_{{0,50}}$ & {h50:.2f} m \\
10-year Goda design wave, $H_{{d,10}}$ & {hd10:.2f} m \\
50-year Goda design wave, $H_{{d,50}}$ & {hd50:.2f} m \\
Governing wave transformation condition & Depth-limited breaking \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Conclusions}}
\begin{{enumerate}}[leftmargin=1.0cm]
    \item A small commercial harbor layout was developed using a design vessel length of {MAX_SHIP_LENGTH_M:.0f} m and design draft of {MAX_DRAFT_M:.1f} m.
    \item The required navigation depth is {nav['required_depth_m']:.1f} m. Based on the linear bathymetry approximation, this depth occurs approximately {summary['dist_req_m']:.0f} m offshore.
    \item The selected perpendicular breakwater length of {summary['bw_perp_m']:.0f} m reaches a local depth of about {summary['toe_depth_m']:.1f} m, so the breakwater reaches the required navigation depth. A navigation channel may still be marked or dredged for safe access, but the preliminary layout satisfies the depth requirement.
    \item The preliminary parallel breakwater length is {summary['bw_parallel_m']:.0f} m, giving a total preliminary breakwater length of {summary['bw_total_m']:.0f} m.
    \item The Goda-type design wave heights at the breakwater are {hd10:.2f} m for the 10-year return period and {hd50:.2f} m for the 50-year return period. For both cases, the governing condition is depth-limited breaking.
\end{{enumerate}}

\section{{Limitations}}
This is a preliminary academic design. The bathymetry is estimated from only two Google Earth Pro profile points, and the harbor layout is schematic rather than georeferenced. A final engineering design should use a detailed hydrographic survey, geotechnical data, wave transformation modelling, sedimentation assessment, navigational safety checks, and the exact coefficient set required by the course instructor.

\section{{References and Data Sources}}
\begin{{enumerate}}[leftmargin=1.0cm]
    \item CE480 Lecture Notes: Wave generation and prediction; JONSWAP wind-wave growth method.
    \item CE480 Lecture Notes: Wave climate and extreme-wave analysis.
    \item CE480 Lecture Notes: Wave transformation and Goda design wave height method for breakwater design.
    \item CE480 HW2 Group 3: Site selection and effective fetch calculation. Used for site coordinate, critical fetch directions, and effective fetch methodology.
    \item CE480 HW3 Group 3: JONSWAP wind-wave prediction from ERA5 wind data. Used for annual wave-height inputs and critical direction/fetch workflow.
    \item CE480 HW4 Group 3: Gumbel extreme wave analysis. Used for 10-year and 50-year deep-water design wave heights.
    \item Google Earth Pro profile points provided by the user: 1500 ft, -19 ft and 1.26 mi, -120 ft; converted to 457.2 m, 5.79 m depth and 2027.8 m, 36.58 m depth.
\end{{enumerate}}

\end{{document}}
"""
    tex_path = OUT_DIR / 'CE480_HW5_improved_report.tex'
    tex_path.write_text(tex, encoding='utf-8')
    return tex_path


def main():
    set_style()
    annual, ret, fit, nav, bathy, goda, summary = compute_all()
    save_tables(annual, ret, nav, bathy, goda, summary)
    figs = {
        'bathy': fig_bathymetry(bathy, nav, summary),
        'layout': fig_layout(summary),
        'nav': fig_navigation_depth(nav),
        'gumbel': fig_gumbel(annual, ret, fit),
        'annual': fig_annual_hs(annual, ret),
        'goda': fig_goda_candidates(goda),
        'wave_summary': fig_wave_periods(goda),
    }
    tex_path = write_latex(annual, ret, fit, nav, bathy, goda, summary, figs)
    # Summary text for terminal
    print(f'LaTeX report written: {tex_path}')
    print(f'Output folder: {OUT_DIR}')
    print('\nKey results:')
    print(f"Required navigation depth: {nav['required_depth_m']:.2f} m")
    print(f"Distance to required depth: {summary['dist_req_m']:.1f} m")
    print(f"Perpendicular breakwater: {summary['bw_perp_m']:.0f} m")
    print(f"Parallel breakwater: {summary['bw_parallel_m']:.0f} m")
    print(f"Total breakwater: {summary['bw_total_m']:.0f} m")
    print(goda[['Tr_year','H0_m','T_s','h_m','Hd_m','governing']].to_string(index=False))

if __name__ == '__main__':
    main()
