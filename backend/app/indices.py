"""
indices.py — Spectral index computation from multispectral/SAR band arrays.
All functions take NumPy arrays and return float32 arrays.
"""
from __future__ import annotations
import numpy as np


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-9, 0.0, a / b)
    return result.astype(np.float32)


# ── Optical indices ───────────────────────────────────────────────────────────

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index. Range [-1, 1]."""
    return _safe_div(nir - red, nir + red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Water Index (McFeeters). Range [-1, 1]."""
    return _safe_div(green - nir, green + nir)


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu). Range [-1, 1]. Better for turbid water."""
    return _safe_div(green - swir, green + swir)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index. Range [-1, 1]."""
    return _safe_div(swir - nir, swir + nir)


def bsi(swir: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Bare Soil Index. Range [-1, 1]."""
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue)
    return _safe_div(num, den)


# ── RGB proxies (when only 3 uint8-style bands available) ────────────────────

def rgb_water_proxy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simple RGB water heuristic: blue channel dominates, low overall brightness.
    Returns a score [0, 1]; threshold at ~0.5 for water mask.
    """
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    maxv = np.maximum(r, np.maximum(g, b)) + 1e-6
    score = (b / maxv) * (1.0 - (r + g + b) / (3 * 255 + 1))
    return np.clip(score, 0, 1)


def rgb_vegetation_proxy(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Green-channel dominance proxy for vegetation."""
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    maxv = np.maximum(r, np.maximum(g, b)) + 1e-6
    return np.clip(g / maxv, 0, 1)


# ── SAR ───────────────────────────────────────────────────────────────────────

def to_db(linear: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear SAR backscatter to dB."""
    return (10.0 * np.log10(np.maximum(linear, eps))).astype(np.float32)


# ── Band extractor from metadata ─────────────────────────────────────────────

def extract_bands(data: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """
    Given a [C, H, W] array and metadata, return named bands.
    Attempts Sentinel-2 name matching first, then position-based fallback.
    """
    band_count = data.shape[0]
    descriptions = [str(d).upper() for d in meta.get("descriptions", [])]
    modality = meta.get("modality", "unknown")
    bands: dict[str, np.ndarray] = {}

    # ── Sentinel-2 name matching ──────────────────────────────────────────────
    s2_map = {
        "B02": "blue", "B2": "blue",
        "B03": "green", "B3": "green",
        "B04": "red", "B4": "red",
        "B08": "nir", "B8": "nir",
        "B8A": "nir",
        "B11": "swir",
    }
    for i, d in enumerate(descriptions):
        for k, v in s2_map.items():
            if k in d and v not in bands:
                bands[v] = data[i]

    # ── Position-based fallback ───────────────────────────────────────────────
    if modality == "optical" and not bands:
        if band_count >= 4:
            bands.setdefault("blue", data[0])
            bands.setdefault("green", data[1])
            bands.setdefault("red", data[2])
            bands.setdefault("nir", data[3])
        if band_count >= 6:
            bands.setdefault("swir", data[5])

    if modality == "rgb":
        bands.setdefault("red", data[0])
        bands.setdefault("green", data[1])
        bands.setdefault("blue", data[2])

    if modality == "sar":
        bands.setdefault("vv", data[0])
        if band_count >= 2:
            bands.setdefault("vh", data[1])

    return bands


def compute_all_indices(data: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    """Compute all applicable indices. Returns name→array map."""
    b = extract_bands(data, meta)
    result: dict[str, np.ndarray] = {}

    if "nir" in b and "red" in b:
        result["ndvi"] = ndvi(b["nir"], b["red"])
    if "green" in b and "nir" in b:
        result["ndwi"] = ndwi(b["green"], b["nir"])
    if "green" in b and "swir" in b:
        result["mndwi"] = mndwi(b["green"], b["swir"])
    if "swir" in b and "nir" in b:
        result["ndbi"] = ndbi(b["swir"], b["nir"])
    if all(k in b for k in ("swir", "red", "nir", "blue")):
        result["bsi"] = bsi(b["swir"], b["red"], b["nir"], b["blue"])
    if all(k in b for k in ("red", "green", "blue")):
        result["rgb_water"] = rgb_water_proxy(b["red"], b["green"], b["blue"])
        result["rgb_vegetation"] = rgb_vegetation_proxy(b["red"], b["green"], b["blue"])
    if "vv" in b:
        result["sar_db"] = to_db(b["vv"])

    return result
