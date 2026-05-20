import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================
# 1. ORIGIN (offshore point, 1-2 km from land)
ORIGIN = (41.934003, 28.069026)  #  (latitude, longitude)
# ============================================================

# ============================================================
# 2. RAY DATA: list of (bearing_deg, latitude, longitude)
# For each main direction (0,22.5,45,...,337.5) measure three rays:
#   bearing = main - 7.5°, main, main + 7.5°
# Total = 16 sectors × 3 = 48 entries.
# ------------------------------------------------------------
RAY_COORDS = [
    # Main 0° sector
    (352.5,  42.845649,27.895897),
    (0.0,    43.339963,28.078544),
    (7.5,    43.409006,28.340631),
    # Main 22.5° sector
    (15.0,   44.754456,29.135816),
    (22.5,   46.540415,30.908103),
    (30.0,   46.213721,31.684169),
    # Main 45° sector
    (37.5,   46.018012, 32.683854),
    (45.0,   45.292607, 33.001442),
    (52.5,   44.761736, 33.541665),
    # Main 67.5° sector
    (60.0,   45.080096, 45.080096),
    (67.5,   44.465643, 38.129395),
    (75.0,   43.607073, 39.739132),
    # Main 90° sector
    (82.5,   42.474583, 41.532527),
    (90.0,   41.819372, 32.646569),
    (97.5,   41.486146, 31.860025),
    # Main 112.5° sector
    (105.0,  41.198339, 31.382485),
    (112.5,  41.183836, 30.386227),
    (120.0,  41.160665, 29.756132),
    # Main 135° sector
    (127.5,  41.234781, 29.234569),
    (135.0,  41.254633, 28.959466),
    (142.5,  41.347997, 28.645114),
    # Main 157.5° sector
    (150.0,  41.418274, 28.460106),
    (157.5,  41.476751, 28.316214),
    (165.0,  41.550431, 28.191785),
    # Main 180° sector
    (172.5,  41.615920, 28.122759),
    (180.0,  41.700600, 28.066728),
    (187.5,  41.753531, 28.040335),
    # Main 337.5° sector
    (330.0,  42.552445,27.585892),
    (337.5,  42.633179,27.677957),
    (345.0,  42.695080,27.794461),
]
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two (lat,lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def main():
    # Step 1: Compute fetch distances for all rays
    rays = []  # list of (bearing, fetch_km)
    for bearing, lat, lon in RAY_COORDS:
        dist = haversine(ORIGIN[0], ORIGIN[1], lat, lon)
        rays.append((bearing, dist))
    
    # Step 2: Define main directions (every 22.5°)
    main_dirs = [i * 22.5 for i in range(16)]
    results = []
    
    for main in main_dirs:
        total_weight_cos = 0.0
        weighted_sum_cos2 = 0.0
        for br, f in rays:
            delta = abs(br - main)
            if delta > 180:
                delta = 360 - delta
            if delta <= 7.5:
                cos_alpha = math.cos(math.radians(delta))
                total_weight_cos += cos_alpha
                weighted_sum_cos2 += f * (cos_alpha ** 2)
        if total_weight_cos == 0:
            eff = 0.0
        else:
            eff = weighted_sum_cos2 / total_weight_cos
        status = "VALID" if eff >= 5.0 else "EXCLUDED (<5km)"
        results.append((main, eff, status))
    
    # Print table
    print("\n" + "="*65)
    print("EFFECTIVE FETCH LENGTHS (using ±7.5° sectors, 3 rays per sector)")
    print("Formula: F_eff = Σ(F_i · cos²α_i) / Σ(cos α_i)")
    print("="*65)
    print(f"{'Main Dir (°)':>12} {'Eff. Fetch (km)':>18} {'Status':>20}")
    print("-"*65)
    for main, eff, stat in results:
        print(f"{main:12.1f} {eff:18.2f} {stat:>20}")

    valid = [(m, e) for (m, e, s) in results if s == "VALID"]
    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)
    top3 = valid_sorted[:3]
    
    print("\n" + "="*65)
    print("MOST CRITICAL 3 DIRECTIONS (largest effective fetch):")
    for i, (m, e) in enumerate(top3, 1):
        print(f"  {i}. Bearing {m:.1f}°  →  {e:.2f} km")
    print("="*65)

    angles = np.deg2rad([m for m,_,_ in results])
    values = [e for _,e,_ in results]
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.bar(angles, values, width=np.deg2rad(15), color='teal', alpha=0.6, edgecolor='black')
    theta_circ = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta_circ, [5]*100, color='red', linestyle='--', label='5 km limit')
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{int(m)}°" for m,_,_ in results])
    ax.set_ylim(0, max(values)*1.1)
    plt.title("Wave Fetch Analysis - Effective Fetch Lengths\n(cos² weighting, ±7.5° sectors)", pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig('fetch_rose.png', dpi=150)
    plt.show()
    print("\nFetch rose saved as 'fetch_rose.png'")

if __name__ == "__main__":
    main()