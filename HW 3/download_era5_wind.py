"""
CE480 HW3 - Download ERA5 hourly wind data from CDS API
Group 1 | Site: Finike-Offshore (36.25000 N, 30.62500 E)

Downloads 'Ocean surface stress equivalent 10m neutral wind speed & direction'
from ERA5 single-levels reanalysis, one year at a time (20 years: 2005-2024).

Prerequisites
-------------
  pip install cdsapi netcdf4 xarray
  ~/.cdsapirc must contain your CDS API url + key

References
----------
  Dataset : reanalysis-era5-single-levels
  Category: Ocean Waves (NOT Wind -- per professor's instructions)
  Variables:
    - ocean_surface_stress_equivalent_10m_neutral_wind_speed
    - ocean_surface_stress_equivalent_10m_neutral_wind_direction
"""

import sys
import subprocess

_DEPS = {"cdsapi": "cdsapi", "netCDF4": "netcdf4", "xarray": "xarray"}
_missing = []
for _mod, _pkg in _DEPS.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(f"[setup] Missing: {', '.join(_missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *_missing]
        )
        print("[setup] Installed.\n")
    except subprocess.CalledProcessError:
        print("[setup] ERROR: pip install failed.")
        print("[setup] Install manually: pip install " + " ".join(_missing))
        sys.exit(1)
else:
    print("[setup] All dependencies satisfied.\n")


import os
import cdsapi

SITE_LAT = 41.934003
SITE_LON = 28.069026
SITE_NAME = "Finike-Offshore"


AREA = [27.97, 41.72, 28.40, 42.12]

YEARS = list(range(2006, 2026))

VARIABLES = [
    "ocean_surface_stress_equivalent_10m_neutral_wind_speed",
    "ocean_surface_stress_equivalent_10m_neutral_wind_direction",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data", "era5_finike")
os.makedirs(OUT_DIR, exist_ok=True)


client = cdsapi.Client()

print(f"Site       : {SITE_NAME} ({SITE_LAT:.5f} N, {SITE_LON:.5f} E)")
print(f"Area (NWSE): {AREA}")
print(f"Years      : {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)")
print(f"Variables  : {len(VARIABLES)}")
print(f"Output     : {os.path.abspath(OUT_DIR)}\n")

for year in YEARS:
    out_file = os.path.join(OUT_DIR, f"era5_wind_{year}.nc")

    if os.path.exists(out_file):
        sz = os.path.getsize(out_file) / 1024 / 1024
        print(f"[skip] {year}  ({sz:.1f} MB already on disk)")
        continue

    print(f"[download] {year} ... ", end="", flush=True)

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        out_file,
    )

    sz = os.path.getsize(out_file) / 1024 / 1024
    print(f"done  ({sz:.1f} MB)")

print("\n=== All downloads complete ===")
