"""
scripts/make_synthetic_samples.py
Generates SYNTHETIC sample GeoTIFFs for demonstration.
All outputs are labelled SYNTHETIC in filenames and carry a note in metadata.

Usage:
    python scripts/make_synthetic_samples.py

Output:
    samples/synthetic/SYNTHETIC_optical_t1.tif   (Sentinel-2-like, 6 bands)
    samples/synthetic/SYNTHETIC_optical_t2.tif   (with edits: clearing, built-up, reservoir)
    samples/synthetic/SYNTHETIC_sar_vv.tif       (single-band SAR VV)
    samples/synthetic/truth.json                 (known edits for validation)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

OUTDIR = Path(__file__).resolve().parents[1] / "samples" / "synthetic"
OUTDIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ── Image parameters ──────────────────────────────────────────────────────────
H, W = 256, 256
CRS_WGS84 = CRS.from_epsg(4326)
# Small area near lat=20, lon=75
BOUNDS = (75.0, 20.0, 75.025, 20.025)   # lon_min, lat_min, lon_max, lat_max
TRANSFORM = from_bounds(*BOUNDS, width=W, height=H)

# Sentinel-2 band order: Blue, Green, Red, NIR, SWIR1, SWIR2
BAND_DESCRIPTIONS = ("B02_blue", "B03_green", "B04_red", "B08_nir", "B11_swir1", "B12_swir2")


def make_base_scene() -> np.ndarray:
    """
    Create a 6-band [C,H,W] float32 base scene.
    Land cover layout:
      - Top-left 1/4: water body (low red/NIR, high green)
      - Middle block: dense vegetation (high NIR, moderate red)
      - Bottom-right 1/4: built-up (high red, moderate NIR, moderate SWIR)
      - Rest: bare soil / sparse vegetation
    """
    bands = np.zeros((6, H, W), dtype=np.float32)

    # Background: sparse vegetation / bare soil
    bands[0] = RNG.uniform(600, 900, (H, W))      # blue
    bands[1] = RNG.uniform(700, 1000, (H, W))     # green
    bands[2] = RNG.uniform(800, 1100, (H, W))     # red
    bands[3] = RNG.uniform(900, 1300, (H, W))     # nir
    bands[4] = RNG.uniform(500, 800, (H, W))      # swir1
    bands[5] = RNG.uniform(300, 600, (H, W))      # swir2

    # Water body top-left (rows 0:64, cols 0:64)
    bands[0, 0:64, 0:64] = RNG.uniform(1200, 1600, (64, 64))   # blue high
    bands[1, 0:64, 0:64] = RNG.uniform(1100, 1400, (64, 64))   # green moderate
    bands[2, 0:64, 0:64] = RNG.uniform(300, 500, (64, 64))     # red low
    bands[3, 0:64, 0:64] = RNG.uniform(200, 400, (64, 64))     # nir low
    bands[4, 0:64, 0:64] = RNG.uniform(100, 300, (64, 64))     # swir low
    bands[5, 0:64, 0:64] = RNG.uniform(80, 200, (64, 64))

    # Dense vegetation middle block (rows 80:176, cols 64:192)
    bands[2, 80:176, 64:192] = RNG.uniform(400, 700, (96, 128))    # red moderate
    bands[3, 80:176, 64:192] = RNG.uniform(2500, 3500, (96, 128))  # nir very high
    bands[4, 80:176, 64:192] = RNG.uniform(300, 600, (96, 128))    # swir low

    # Built-up bottom-right (rows 192:256, cols 192:256)
    bands[2, 192:, 192:] = RNG.uniform(1500, 2000, (64, 64))   # red high
    bands[3, 192:, 192:] = RNG.uniform(1200, 1800, (64, 64))   # nir moderate
    bands[4, 192:, 192:] = RNG.uniform(1800, 2500, (64, 64))   # swir high

    # Add sensor noise
    bands += RNG.normal(0, 30, bands.shape).astype(np.float32)
    bands = np.clip(bands, 0, 10000).astype(np.float32)
    return bands


def make_t2_edits(t1: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply known edits to t1 to create t2. Returns (t2, truth_dict)."""
    t2 = t1.copy()
    edits = []

    # Edit 1: Clearing — convert vegetation patch to bare soil (rows 80:120, cols 64:104)
    t2[2, 80:120, 64:104] = RNG.uniform(1000, 1400, (40, 40))   # red increases
    t2[3, 80:120, 64:104] = RNG.uniform(800, 1200, (40, 40))    # nir drops
    edits.append({
        "type": "vegetation_to_bare_soil",
        "rows": [80, 120], "cols": [64, 104],
        "description": "Clearing: vegetation → bare soil",
    })

    # Edit 2: New built-up block (rows 140:176, cols 150:192)
    t2[2, 140:176, 150:192] = RNG.uniform(1600, 2100, (36, 42))
    t2[3, 140:176, 150:192] = RNG.uniform(1100, 1700, (36, 42))
    t2[4, 140:176, 150:192] = RNG.uniform(1900, 2600, (36, 42))
    edits.append({
        "type": "vegetation_to_builtup",
        "rows": [140, 176], "cols": [150, 192],
        "description": "New built-up block",
    })

    # Edit 3: Reservoir — extend water body (rows 0:80, cols 0:80)
    t2[0, 64:80, 0:80] = RNG.uniform(1200, 1600, (16, 80))
    t2[1, 64:80, 0:80] = RNG.uniform(1100, 1400, (16, 80))
    t2[2, 64:80, 0:80] = RNG.uniform(300, 500, (16, 80))
    t2[3, 64:80, 0:80] = RNG.uniform(200, 400, (16, 80))
    edits.append({
        "type": "soil_to_water",
        "rows": [64, 80], "cols": [0, 80],
        "description": "Reservoir extension",
    })

    # Add noise
    t2 += RNG.normal(0, 30, t2.shape).astype(np.float32)
    t2 = np.clip(t2, 0, 10000).astype(np.float32)
    return t2, {"edits": edits, "note": "SYNTHETIC ground truth for change-detection validation"}


def make_sar(optical: np.ndarray) -> np.ndarray:
    """
    Create a synthetic SAR (single-band VV, linear scale) from the optical.
    Water = very low backscatter, vegetation = moderate, built-up = high (double-bounce).
    """
    # Use NIR to estimate vegetation, use SWIR for built-up
    nir = optical[3]
    swir = optical[4]
    red = optical[2]

    # Normalise to [0,1]
    def norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1)

    veg_proxy = np.clip(norm(nir) - norm(red), 0, 1)
    water_proxy = (nir < 500).astype(np.float32)
    builtup_proxy = np.clip(norm(swir), 0, 1)

    # SAR backscatter in linear scale: water ≈ 0.002, veg ≈ 0.05, builtup ≈ 0.2
    sar = 0.05 * np.ones((1, H, W), dtype=np.float32)
    sar[0] = (
        0.002 * water_proxy
        + 0.04 * veg_proxy
        + 0.25 * builtup_proxy
        + 0.02
    )
    # Speckle noise (multiplicative)
    sar *= RNG.exponential(1.0, (1, H, W)).astype(np.float32)
    sar = np.clip(sar, 1e-6, 1.0).astype(np.float32)
    return sar


def write_geotiff(path: Path, data: np.ndarray, descriptions: tuple | None = None,
                  nodata: float | None = None):
    c, h, w = data.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w, count=c,
        dtype="float32",
        crs=CRS_WGS84,
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions[:c], start=1):
                dst.update_tags(i, description=desc)


def main():
    print("Generating SYNTHETIC samples...")
    t1 = make_base_scene()
    t2, truth = make_t2_edits(t1)
    sar = make_sar(t1)

    out_t1  = OUTDIR / "SYNTHETIC_optical_t1.tif"
    out_t2  = OUTDIR / "SYNTHETIC_optical_t2.tif"
    out_sar = OUTDIR / "SYNTHETIC_sar_vv.tif"
    out_truth = OUTDIR / "truth.json"

    write_geotiff(out_t1,  t1,  BAND_DESCRIPTIONS)
    write_geotiff(out_t2,  t2,  BAND_DESCRIPTIONS)
    write_geotiff(out_sar, sar, ("VV",))

    out_truth.write_text(json.dumps(truth, indent=2))

    print(f"  Written: {out_t1.name}")
    print(f"  Written: {out_t2.name}")
    print(f"  Written: {out_sar.name}")
    print(f"  Written: {out_truth.name}")
    print("Done. All files labelled SYNTHETIC.")


if __name__ == "__main__":
    main()
