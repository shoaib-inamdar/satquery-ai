"""
tools/grounding.py — Text-guided region grounding.
Builds masks using spectral indices / SAR thresholding.
Returns GeoJSON polygons, overlay PNG, area in km².
"""
from __future__ import annotations
import io
import base64
from typing import Any

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk
import rasterio.features
from shapely.geometry import shape, mapping

from ..indices import compute_all_indices, extract_bands, to_db
from ..evidence import EvidenceBundle
from ..schemas import ToolResult
from ..confidence import score_otsu_separation


# ── Target class → index selection ───────────────────────────────────────────
CLASS_INDEX_PRIORITY: dict[str, list[str]] = {
    "water":       ["mndwi", "ndwi", "rgb_water", "sar_db"],
    "vegetation":  ["ndvi", "rgb_vegetation"],
    "built-up":    ["ndbi"],
    "bare-soil":   ["bsi", "ndbi"],
    "agriculture": ["ndvi"],
}

# For SAR: water = low backscatter (negative dB)
SAR_INVERT_CLASSES = {"water"}


def _make_mask_from_index(
    index_arr: np.ndarray,
    index_name: str,
    target_class: str,
    smoothing: int = 3,
) -> tuple[np.ndarray, float, float]:
    """
    Threshold index_arr to get a binary mask.
    Returns (mask [H,W bool], threshold_used, otsu_separation_score).
    """
    valid = index_arr[np.isfinite(index_arr)]
    if valid.size < 10:
        return np.zeros(index_arr.shape, dtype=bool), 0.0, 0.0

    # Otsu threshold
    try:
        thr = float(threshold_otsu(valid))
    except Exception:
        thr = float(np.median(valid))

    sep = score_otsu_separation(index_arr, thr)

    if index_name == "sar_db" and target_class in SAR_INVERT_CLASSES:
        # Water = low backscatter in dB
        mask = index_arr <= thr
    elif index_name in ("ndwi", "mndwi", "rgb_water"):
        mask = index_arr > 0.0
    elif index_name in ("ndvi", "rgb_vegetation"):
        mask = index_arr > 0.3
    elif index_name == "ndbi":
        mask = index_arr > 0.0
    elif index_name == "bsi":
        mask = index_arr > 0.0
    else:
        mask = index_arr > thr

    # Morphological cleaning
    k = max(1, smoothing)
    selem = disk(k)
    mask = opening(closing(mask, selem), selem)

    return mask.astype(bool), thr, sep


def _mask_to_geojson(
    mask: np.ndarray, meta: dict
) -> dict:
    """Vectorise a boolean mask to GeoJSON FeatureCollection."""
    if meta.get("transform") and meta.get("crs"):
        transform = rasterio.transform.Affine(*meta["transform"][:6])
        crs = meta["crs"]
    else:
        transform = rasterio.transform.from_bounds(0, 0, mask.shape[1], mask.shape[0],
                                                    mask.shape[1], mask.shape[0])
        crs = None

    mask_uint8 = mask.astype(np.uint8)
    shapes = list(rasterio.features.shapes(mask_uint8, mask=(mask_uint8 == 1),
                                            transform=transform))
    features = []
    for geom, val in shapes:
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {"class": "target", "value": int(val)},
        })
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}} if crs else None,
        "features": features,
    }


def _compute_area_km2(mask: np.ndarray, meta: dict, gsd_m: float | None) -> float:
    """Compute area of True pixels in km²."""
    n_pixels = int(mask.sum())
    res = meta.get("resolution_m") or gsd_m
    if res is None:
        return float("nan")  # cannot compute without GSD
    area_m2 = n_pixels * (res ** 2)
    return round(area_m2 / 1e6, 6)


def _mask_to_overlay_b64(data: np.ndarray, meta: dict, mask: np.ndarray) -> str:
    """Return base64-encoded PNG of the mask overlay on a greyscale preview."""
    from ..preprocessing import make_rgb_preview, percentile_stretch
    rgb = make_rgb_preview(data, meta)
    overlay = rgb.copy()
    # Highlight mask in cyan
    overlay[mask, 0] = 0
    overlay[mask, 1] = 200
    overlay[mask, 2] = 255
    # Encode
    success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def run_grounding(
    data: np.ndarray,
    meta: dict,
    query: str,
    bundle: EvidenceBundle,
    params: dict,
) -> ToolResult:
    target_class = str(params.get("target_class", "water")).lower()
    smoothing = int(params.get("smoothing", 3))
    gsd_m = params.get("gsd_m")

    indices = compute_all_indices(data, meta)
    priority = CLASS_INDEX_PRIORITY.get(target_class, ["ndvi"])

    best_mask: np.ndarray | None = None
    best_sep = -1.0
    best_index_name = "none"
    best_threshold = 0.0

    for idx_name in priority:
        if idx_name in indices:
            mask, thr, sep = _make_mask_from_index(
                indices[idx_name], idx_name, target_class, smoothing
            )
            if sep > best_sep:
                best_sep = sep
                best_mask = mask
                best_index_name = idx_name
                best_threshold = thr

    if best_mask is None or not best_mask.any():
        return ToolResult(
            tool="grounding",
            provenance="classical-cv-baseline",
            answer=f"No '{target_class}' regions could be detected in this image with the available bands.",
            measurements={"available_indices": list(indices.keys())},
        )

    area_km2 = _compute_area_km2(best_mask, meta, gsd_m)
    pixel_fraction = float(best_mask.sum() / best_mask.size)
    geojson = _mask_to_geojson(best_mask, meta)
    overlay_b64 = _mask_to_overlay_b64(data, meta, best_mask)

    num_polygons = len(geojson["features"])

    measurements: dict[str, Any] = {
        "target_class": target_class,
        "index_used": best_index_name,
        "threshold": round(best_threshold, 4),
        "separation_score": round(best_sep, 4),
        "pixel_fraction": round(pixel_fraction, 4),
        "area_km2": area_km2,
        "num_polygons": num_polygons,
    }

    for k, v in measurements.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
            bundle.add_measurement(name=k, value=v, method=f"grounding:{best_index_name}",
                                   provenance="classical-cv-baseline", confidence=round(best_sep, 3))

    area_str = f"{area_km2:.4f} km²" if area_km2 == area_km2 else "unknown (no GSD available)"
    answer = (
        f"Detected {num_polygons} '{target_class}' region(s) covering "
        f"{pixel_fraction*100:.2f}% of the image ({area_str}). "
        f"Index used: {best_index_name} (Otsu threshold: {best_threshold:.4f}, "
        f"separation score: {best_sep:.3f})."
    )

    return ToolResult(
        tool="grounding",
        provenance="classical-cv-baseline",
        answer=answer,
        measurements=measurements,
        geojson=geojson,
        overlay_url=f"data:image/png;base64,{overlay_b64}",
    )
