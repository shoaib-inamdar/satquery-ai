"""
tools/presets.py — Preset workflows: flood extent and land-disturbance candidates.
These are built on the same index/threshold tools, just with pre-set task logic.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from ..indices import compute_all_indices, extract_bands, to_db
from ..evidence import EvidenceBundle
from ..schemas import ToolResult
from .grounding import _make_mask_from_index, _compute_area_km2, _mask_to_overlay_b64


# ── Flood extent ──────────────────────────────────────────────────────────────

def run_flood_extent(
    data_before: np.ndarray,
    meta_before: dict,
    data_after: np.ndarray,
    meta_after: dict,
    bundle: EvidenceBundle,
    params: dict,
) -> ToolResult:
    """
    Flood extent: water mask before and after, flooded area = after_water - before_water.
    Works with bi-temporal (optical or SAR) or optical+SAR pairs.
    """
    water_index = str(params.get("water_index", "auto"))
    gsd_m = params.get("gsd_m")

    def get_water_mask(data, meta, pref):
        indices = compute_all_indices(data, meta)
        priority = []
        if pref == "auto":
            priority = ["mndwi", "ndwi", "rgb_water", "sar_db"]
        elif pref == "sar_threshold":
            priority = ["sar_db"]
        else:
            priority = [pref]
        for idx_name in priority:
            if idx_name in indices:
                mask, thr, sep = _make_mask_from_index(indices[idx_name], idx_name, "water")
                return mask, idx_name, thr, sep
        return np.zeros(data.shape[1:], dtype=bool), "none", 0.0, 0.0

    mask_before, idx_before, thr_before, sep_before = get_water_mask(data_before, meta_before, water_index)
    mask_after,  idx_after,  thr_after,  sep_after  = get_water_mask(data_after,  meta_after,  water_index)

    # Align shapes
    h = min(mask_before.shape[0], mask_after.shape[0])
    w = min(mask_before.shape[1], mask_after.shape[1])
    mask_before = mask_before[:h, :w]
    mask_after  = mask_after[:h, :w]

    flooded = mask_after & ~mask_before
    receded = mask_before & ~mask_after

    before_area = _compute_area_km2(mask_before, meta_before, gsd_m)
    after_area  = _compute_area_km2(mask_after,  meta_after,  gsd_m)
    flood_area  = _compute_area_km2(flooded, meta_before, gsd_m)
    recede_area = _compute_area_km2(receded, meta_before, gsd_m)

    for k, v in [("water_before_km2", before_area), ("water_after_km2", after_area),
                  ("flooded_area_km2", flood_area), ("receded_area_km2", recede_area)]:
        if v == v:
            bundle.add_measurement(name=k, value=round(v, 6), method=f"flood-extent:{idx_after}",
                                   provenance="classical-cv-baseline", confidence=round(sep_after, 3))

    overlay_b64 = _mask_to_overlay_b64(data_after, meta_after, flooded[:data_after.shape[1], :data_after.shape[2]])

    area_str = f"{flood_area:.4f} km²" if flood_area == flood_area else "unknown (no GSD)"
    answer = (
        f"Flood extent analysis: water before = {before_area:.4f} km², "
        f"water after = {after_area:.4f} km². "
        f"Estimated flooded area (new water): {area_str}. "
        f"Index used: {idx_after}. NOTE: Results are CLASSICAL-CV-BASELINE estimates for human review."
    )

    return ToolResult(
        tool="flood_extent",
        provenance="classical-cv-baseline",
        answer=answer,
        measurements={
            "water_before_km2": before_area,
            "water_after_km2":  after_area,
            "flooded_area_km2": flood_area,
            "receded_area_km2": recede_area,
            "index_before": idx_before,
            "index_after":  idx_after,
        },
        overlay_url=f"data:image/png;base64,{overlay_b64}" if overlay_b64 else None,
    )


# ── Land-disturbance candidates ───────────────────────────────────────────────

def run_land_disturbance(
    data_t1: np.ndarray,
    meta_t1: dict,
    data_t2: np.ndarray,
    meta_t2: dict,
    bundle: EvidenceBundle,
    params: dict,
) -> ToolResult:
    """
    NDVI drop + bare-soil gain between dates → ranked candidate polygons.
    Output labelled as CANDIDATES FOR HUMAN REVIEW. Never called 'detected landslides'.
    """
    ndvi_drop_thr = float(params.get("ndvi_drop_threshold", 0.15))
    gsd_m = params.get("gsd_m")

    idx_t1 = compute_all_indices(data_t1, meta_t1)
    idx_t2 = compute_all_indices(data_t2, meta_t2)

    ndvi_t1 = idx_t1.get("ndvi")
    ndvi_t2 = idx_t2.get("ndvi")
    bsi_t1  = idx_t1.get("bsi")
    bsi_t2  = idx_t2.get("bsi")

    if ndvi_t1 is None or ndvi_t2 is None:
        return ToolResult(
            tool="land_disturbance",
            provenance="classical-cv-baseline",
            answer="Cannot compute land-disturbance candidates: NDVI requires NIR + Red bands.",
            measurements={},
        )

    h = min(ndvi_t1.shape[0], ndvi_t2.shape[0])
    w = min(ndvi_t1.shape[1], ndvi_t2.shape[1])
    ndvi_drop = ndvi_t1[:h, :w] - ndvi_t2[:h, :w]  # positive = vegetation lost

    candidate_mask = ndvi_drop > ndvi_drop_thr
    if bsi_t1 is not None and bsi_t2 is not None:
        bsi_gain = bsi_t2[:h, :w] - bsi_t1[:h, :w]
        candidate_mask = candidate_mask & (bsi_gain > 0)

    candidate_area_km2 = _compute_area_km2(candidate_mask, meta_t1, gsd_m)
    candidate_fraction = float(candidate_mask.sum() / candidate_mask.size)

    bundle.add_measurement(
        name="land_disturbance_candidate_fraction", value=round(candidate_fraction, 4),
        method=f"ndvi-drop>{ndvi_drop_thr}+bsi-gain",
        provenance="classical-cv-baseline", confidence=0.5,
    )

    overlay_b64 = _mask_to_overlay_b64(data_t1, meta_t1, candidate_mask)

    answer = (
        f"Land-disturbance CANDIDATES (NDVI drop > {ndvi_drop_thr}): "
        f"{candidate_fraction*100:.2f}% of image area, "
        f"estimated {candidate_area_km2:.4f} km². "
        "⚠️ These are CANDIDATES FOR HUMAN REVIEW — not confirmed detections. "
        "Do not use for operational decisions without expert validation."
    )

    return ToolResult(
        tool="land_disturbance",
        provenance="classical-cv-baseline",
        answer=answer,
        measurements={
            "candidate_fraction": round(candidate_fraction, 4),
            "candidate_area_km2": candidate_area_km2,
            "ndvi_drop_threshold": ndvi_drop_thr,
        },
        overlay_url=f"data:image/png;base64,{overlay_b64}" if overlay_b64 else None,
    )
