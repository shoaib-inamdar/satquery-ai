"""
ingestion.py — GeoTIFF/TIFF loading, modality detection, compatibility checking.
Uses rasterio + NumPy only. No OpenCV for band interpretation.
"""
from __future__ import annotations
import io
import math
import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box

from .schemas import (
    CheckItem, CompatibilityReport, ImageMeta,
)
from .config import MIN_OVERLAP_FRACTION


# ── Band-name→index mappings for Sentinel-2 ──────────────────────────────────
S2_BAND_MAP = {
    "B02": "blue", "B2": "blue",
    "B03": "green", "B3": "green",
    "B04": "red", "B4": "red",
    "B08": "nir", "B8": "nir",
    "B8A": "nir",
    "B11": "swir", "B11A": "swir",
}

SAR_POLARISATION_KEYWORDS = {"VV", "VH", "HH", "HV"}


def detect_modality(
    band_count: int,
    dtype: str,
    descriptions: list[str | None],
    band_min: float,
    band_max: float,
) -> str:
    """
    Heuristic modality detection.
    Returns: 'sar' | 'optical' | 'rgb' | 'unknown'
    """
    desc_upper = [str(d).upper() for d in descriptions if d]
    # SAR: 1–2 bands, float dtype or polarisation keywords in descriptions
    if band_count in (1, 2):
        if any(kw in d for d in desc_upper for kw in SAR_POLARISATION_KEYWORDS):
            return "sar"
        if "float" in dtype or band_max > 1000:
            return "sar"
    # Optical (multispectral) — 4+ bands, or S2 keywords
    if band_count >= 4:
        return "optical"
    if any(k in d for d in desc_upper for k in S2_BAND_MAP):
        return "optical"
    # RGB fallback
    if band_count == 3 and "uint8" in dtype:
        return "rgb"
    return "unknown"


def load_image(filepath: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a GeoTIFF/TIFF/PNG/JPEG. Returns (bands_array [C,H,W], meta_dict).
    meta_dict keys: crs, transform, band_count, dtype, nodata, resolution_m,
                    bounds, descriptions, modality, filename.
    """
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        descriptions = list(src.descriptions)
        meta = {
            "filename": filepath.name,
            "band_count": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform)[:6],
            "nodata": src.nodata,
            "bounds": list(src.bounds),
            "width": src.width,
            "height": src.height,
            "descriptions": descriptions,
        }
        # Resolution in metres (approximate for geographic CRS)
        if src.crs and src.crs.is_geographic:
            meta["resolution_m"] = abs(src.transform.a) * 111320.0
        elif src.crs and src.crs.is_projected:
            meta["resolution_m"] = abs(src.transform.a)
        else:
            meta["resolution_m"] = None

    band_min = float(data.min())
    band_max = float(data.max())
    nodata_fraction = float(np.isnan(data).sum() / data.size)
    if meta["nodata"] is not None:
        nodata_fraction = float((data == meta["nodata"]).sum() / data.size)

    meta["nodata_fraction"] = nodata_fraction
    meta["modality"] = detect_modality(
        meta["band_count"], meta["dtype"], descriptions, band_min, band_max
    )
    return data, meta


def build_image_meta(meta: dict) -> ImageMeta:
    return ImageMeta(
        filename=meta["filename"],
        modality=meta["modality"],
        band_count=meta["band_count"],
        dtype=meta["dtype"],
        crs=meta.get("crs"),
        transform=meta.get("transform"),
        resolution_m=meta.get("resolution_m"),
        bounds=meta.get("bounds"),
        nodata_fraction=meta.get("nodata_fraction", 0.0),
        date_hint=meta.get("date_hint"),
    )


def _bounds_overlap_fraction(bounds_a, bounds_b, crs_a, crs_b) -> float:
    """Return the fraction of bounds_a that overlaps with bounds_b (both in EPSG:4326)."""
    try:
        if crs_a and crs_b:
            if crs_a != crs_b:
                b = transform_bounds(CRS.from_string(crs_b), CRS.from_string(crs_a), *bounds_b)
                bounds_b = b
        box_a = box(*bounds_a)
        box_b = box(*bounds_b)
        inter = box_a.intersection(box_b)
        if box_a.area == 0:
            return 0.0
        return float(inter.area / box_a.area)
    except Exception:
        return 0.0  # cannot compute → warn rather than crash


def check_compatibility(
    metas: list[dict],
    declared_config: Optional[str] = None,
) -> CompatibilityReport:
    """
    Run compatibility checks on 1 or 2 loaded images.
    declared_config: 'single' | 'bitemporal' | 'optical_sar' | None (auto-detect)
    """
    checks: list[CheckItem] = []
    images = [build_image_meta(m) for m in metas]

    # ── Single-image checks ───────────────────────────────────────────────────
    for i, m in enumerate(metas):
        tag = f"image{i+1}"
        if m["modality"] == "unknown":
            checks.append(CheckItem(
                key=f"{tag}_modality",
                status="warn",
                message=f"{m['filename']}: modality could not be detected — treat as generic.",
            ))
        else:
            checks.append(CheckItem(
                key=f"{tag}_modality",
                status="pass",
                message=f"{m['filename']}: detected as {m['modality']} ({m['band_count']} bands).",
            ))

        nf = m.get("nodata_fraction", 0.0)
        if nf > 0.3:
            checks.append(CheckItem(
                key=f"{tag}_nodata",
                status="warn",
                message=f"{m['filename']}: {nf:.1%} nodata/NaN pixels — results may be unreliable.",
            ))
        else:
            checks.append(CheckItem(
                key=f"{tag}_nodata",
                status="pass",
                message=f"{m['filename']}: nodata fraction {nf:.1%}.",
            ))

        fmt = m["filename"].rsplit(".", 1)[-1].lower()
        if fmt not in ("tif", "tiff", "geotiff", "png", "jpg", "jpeg"):
            checks.append(CheckItem(key=f"{tag}_format", status="fail",
                message=f"{m['filename']}: unsupported format '{fmt}'."))
        elif fmt in ("png", "jpg", "jpeg"):
            checks.append(CheckItem(key=f"{tag}_format", status="warn",
                message=f"{m['filename']}: PNG/JPEG accepted only for benchmark samples — no georeferencing."))
        else:
            checks.append(CheckItem(key=f"{tag}_format", status="pass",
                message=f"{m['filename']}: GeoTIFF/TIFF format accepted."))

    # ── Determine input config ────────────────────────────────────────────────
    if len(metas) == 1:
        input_config = "single"
    else:
        m0, m1 = metas[0], metas[1]
        modalities = {m0["modality"], m1["modality"]}
        if "sar" in modalities and len(modalities) > 1:
            input_config = "optical_sar"
        else:
            input_config = "bitemporal"

    if declared_config and declared_config != input_config:
        checks.append(CheckItem(
            key="input_config_override",
            status="warn",
            message=f"Declared config '{declared_config}' differs from auto-detected '{input_config}'. Using declared.",
        ))
        input_config = declared_config

    # ── Pair-specific checks ──────────────────────────────────────────────────
    if len(metas) == 2:
        m0, m1 = metas[0], metas[1]

        # CRS compatibility
        if m0.get("crs") and m1.get("crs"):
            if m0["crs"] == m1["crs"]:
                checks.append(CheckItem(key="crs_match", status="pass",
                    message="Both images share the same CRS."))
            else:
                checks.append(CheckItem(key="crs_match", status="warn",
                    message=f"CRS mismatch: {m0['crs']} vs {m1['crs']} — will reproject image2 onto image1."))
        else:
            checks.append(CheckItem(key="crs_match", status="warn",
                message="One or both images lack a CRS — georeferenced co-registration unavailable."))

        # Spatial overlap
        if m0.get("bounds") and m1.get("bounds") and m0.get("crs") and m1.get("crs"):
            frac = _bounds_overlap_fraction(
                m0["bounds"], m1["bounds"], m0["crs"], m1["crs"]
            )
            if frac < MIN_OVERLAP_FRACTION:
                checks.append(CheckItem(key="spatial_overlap", status="fail",
                    message=f"Spatial overlap {frac:.1%} < {MIN_OVERLAP_FRACTION:.0%} — cannot analyse this pair. Upload images of the same area."))
            else:
                checks.append(CheckItem(key="spatial_overlap", status="pass",
                    message=f"Spatial overlap {frac:.1%} — sufficient for analysis."))

        # Resolution similarity
        r0 = m0.get("resolution_m")
        r1 = m1.get("resolution_m")
        if r0 and r1:
            ratio = max(r0, r1) / max(min(r0, r1), 1e-6)
            if ratio > 10:
                checks.append(CheckItem(key="resolution_ratio", status="warn",
                    message=f"Resolution ratio {ratio:.1f}× ({r0:.1f}m vs {r1:.1f}m) — image2 will be resampled."))
            else:
                checks.append(CheckItem(key="resolution_ratio", status="pass",
                    message=f"Resolution similar: {r0:.1f}m vs {r1:.1f}m."))

    # ── Overall ───────────────────────────────────────────────────────────────
    statuses = [c.status for c in checks]
    if "fail" in statuses:
        overall = "fail"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "pass"

    return CompatibilityReport(
        input_config=input_config,
        checks=checks,
        overall=overall,
        images=images,
    )
