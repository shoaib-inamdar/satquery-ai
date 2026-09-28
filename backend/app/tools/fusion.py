"""
tools/fusion.py — Optical-SAR cross-modal fusion.
Agreement map: confirmed by both / optical-only / SAR-only.
SAR-only ambiguous regions are reported as candidates, never certain.
"""
from __future__ import annotations
import base64
from typing import Any

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk

from ..indices import compute_all_indices, extract_bands, to_db
from ..evidence import EvidenceBundle
from ..schemas import ToolResult
from ..confidence import score_otsu_separation, score_cross_modal_agreement
from ..preprocessing import make_rgb_preview


CLASSES_FUSION = ["confirmed_water", "optical_only_water", "sar_only_water",
                  "confirmed_builtup", "optical_only_builtup", "sar_only_builtup"]


def _sar_water_mask(sar_data: np.ndarray, meta: dict, smoothing: int = 3) -> np.ndarray:
    """Low-backscatter SAR water mask using Otsu on dB."""
    from ..indices import extract_bands
    b = extract_bands(sar_data, meta)
    band = b.get("vv") or sar_data[0]
    db = to_db(band)
    valid = db[np.isfinite(db)]
    if valid.size < 10:
        return np.zeros(band.shape, dtype=bool)
    thr = float(threshold_otsu(valid))
    mask = db <= thr  # water = low backscatter
    selem = disk(smoothing)
    return closing(mask, selem)


def _sar_builtup_mask(sar_data: np.ndarray, meta: dict) -> np.ndarray:
    """High-backscatter SAR rough proxy for built-up (double-bounce)."""
    from ..indices import extract_bands
    b = extract_bands(sar_data, meta)
    band = b.get("vv") or sar_data[0]
    db = to_db(band)
    valid = db[np.isfinite(db)]
    if valid.size < 10:
        return np.zeros(band.shape, dtype=bool)
    thr = float(threshold_otsu(valid))
    return db >= thr


def run_fusion(
    optical_data: np.ndarray,
    optical_meta: dict,
    sar_data: np.ndarray,
    sar_meta: dict,
    query: str,
    bundle: EvidenceBundle,
    params: dict,
) -> ToolResult:
    agreement_threshold = float(params.get("agreement_threshold", 0.5))

    # ── Optical masks ──────────────────────────────────────────────────────────
    opt_indices = compute_all_indices(optical_data, optical_meta)

    if "mndwi" in opt_indices:
        opt_water = opt_indices["mndwi"] > 0.0
        water_index = "mndwi"
    elif "ndwi" in opt_indices:
        opt_water = opt_indices["ndwi"] > 0.0
        water_index = "ndwi"
    else:
        opt_water = np.zeros(optical_data.shape[1:], dtype=bool)
        water_index = "none"

    opt_builtup = opt_indices.get("ndbi", np.zeros(optical_data.shape[1:])) > 0.0

    # ── SAR masks ──────────────────────────────────────────────────────────────
    sar_water = _sar_water_mask(sar_data, sar_meta)
    sar_builtup = _sar_builtup_mask(sar_data, sar_meta)

    # ── Align shapes ───────────────────────────────────────────────────────────
    h = min(opt_water.shape[0], sar_water.shape[0])
    w = min(opt_water.shape[1], sar_water.shape[1])
    opt_water  = opt_water[:h, :w]
    opt_builtup = opt_builtup[:h, :w]
    sar_water  = sar_water[:h, :w]
    sar_builtup = sar_builtup[:h, :w]

    total_px = h * w

    # ── Agreement classes ──────────────────────────────────────────────────────
    conf_water      = opt_water & sar_water
    opt_only_water  = opt_water & ~sar_water
    sar_only_water  = sar_water & ~opt_water   # ambiguous candidates
    conf_builtup    = opt_builtup & sar_builtup
    opt_only_builtup = opt_builtup & ~sar_builtup
    sar_only_builtup = sar_builtup & ~opt_builtup  # ambiguous

    areas: dict[str, float] = {
        "confirmed_water":       float(conf_water.sum() / total_px),
        "optical_only_water":    float(opt_only_water.sum() / total_px),
        "sar_only_water_candidates": float(sar_only_water.sum() / total_px),
        "confirmed_builtup":     float(conf_builtup.sum() / total_px),
        "optical_only_builtup":  float(opt_only_builtup.sum() / total_px),
        "sar_only_builtup_candidates": float(sar_only_builtup.sum() / total_px),
    }

    agree_water = conf_water.sum() / max((opt_water | sar_water).sum(), 1)
    agree_builtup = conf_builtup.sum() / max((opt_builtup | sar_builtup).sum(), 1)
    cross_modal_agree = float((agree_water + agree_builtup) / 2)

    # ── Overlay ────────────────────────────────────────────────────────────────
    rgb = make_rgb_preview(optical_data, optical_meta)
    h_rgb, w_rgb = rgb.shape[:2]
    def resize_mask(m):
        return cv2.resize(m[:h, :w].astype(np.uint8), (w_rgb, h_rgb)).astype(bool)

    overlay = rgb.copy()
    overlay[resize_mask(conf_water), :]      = [0, 120, 255]    # blue = confirmed water
    overlay[resize_mask(opt_only_water), :]  = [100, 180, 255]  # light blue = optical-only
    overlay[resize_mask(sar_only_water), :]  = [80, 200, 200]   # teal = SAR-only (ambiguous)
    overlay[resize_mask(conf_builtup), :]    = [200, 80, 50]    # red = confirmed built-up
    overlay[resize_mask(opt_only_builtup), :] = [255, 160, 100] # orange = optical-only
    overlay[resize_mask(sar_only_builtup), :] = [180, 50, 100]  # purple = SAR-only (ambiguous)

    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buf.tobytes()).decode() if _ else ""

    # ── Evidence ────────────────────────────────────────────────────────────────
    for k, v in areas.items():
        bundle.add_measurement(
            name=k, value=round(v, 4),
            method=f"fusion:{water_index}+sar_otsu",
            provenance="rule-based", confidence=round(cross_modal_agree, 3),
        )

    measurements: dict[str, Any] = {
        **{k: round(v, 4) for k, v in areas.items()},
        "optical_water_index": water_index,
        "cross_modal_agreement": round(cross_modal_agree, 4),
        "agreement_water": round(float(agree_water), 4),
        "agreement_builtup": round(float(agree_builtup), 4),
    }

    answer = (
        f"Fusion result: confirmed water {areas['confirmed_water']*100:.2f}%, "
        f"confirmed built-up {areas['confirmed_builtup']*100:.2f}%. "
        f"SAR-only water candidates: {areas['sar_only_water_candidates']*100:.2f}% "
        f"(possibly flooded vegetation — for human review). "
        f"Cross-modal agreement: {cross_modal_agree*100:.1f}%."
    )

    return ToolResult(
        tool="fusion",
        provenance="rule-based",
        answer=answer,
        measurements=measurements,
        overlay_url=f"data:image/png;base64,{overlay_b64}" if overlay_b64 else None,
        extra={"cross_modal_agreement": round(cross_modal_agree, 4)},
    )
