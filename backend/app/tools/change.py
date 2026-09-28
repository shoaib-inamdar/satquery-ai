"""
tools/change.py — Bi-temporal change analysis.
Change vector analysis, Otsu threshold, class transition matrix,
change-VQA (increased/decreased/unchanged per class).
"""
from __future__ import annotations
import base64
from typing import Any

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk
import rasterio.features
from scipy.ndimage import median_filter

from ..indices import compute_all_indices, extract_bands
from ..evidence import EvidenceBundle
from ..schemas import ToolResult
from ..config import CHANGE_UNCHANGED_TOLERANCE
from ..confidence import score_otsu_separation, score_coreg_residual
from ..preprocessing import reproject_to_grid, orb_ransac_register, histogram_match


# ── Land-cover map from indices ───────────────────────────────────────────────
CLASSES = ["water", "vegetation", "built-up", "bare-soil", "other"]


def classify_pixels(indices: dict[str, np.ndarray], meta: dict) -> np.ndarray:
    """
    Hard-classify every pixel into one of CLASSES.
    Returns int array [H, W] with values 0–4.
    """
    h, w = next(iter(indices.values())).shape if indices else (1, 1)
    lc = np.full((h, w), CLASSES.index("other"), dtype=np.int8)

    if "ndwi" in indices:
        lc[indices["ndwi"] > 0.0] = CLASSES.index("water")
    if "mndwi" in indices:
        lc[indices["mndwi"] > 0.0] = CLASSES.index("water")
    if "ndvi" in indices:
        lc[(lc != CLASSES.index("water")) & (indices["ndvi"] > 0.3)] = CLASSES.index("vegetation")
    if "ndbi" in indices:
        lc[(lc == CLASSES.index("other")) & (indices["ndbi"] > 0.0)] = CLASSES.index("built-up")
    if "bsi" in indices:
        lc[(lc == CLASSES.index("other")) & (indices["bsi"] > 0.0)] = CLASSES.index("bare-soil")

    return lc


def _class_fractions(lc: np.ndarray) -> dict[str, float]:
    total = lc.size
    return {c: float((lc == i).sum() / total) for i, c in enumerate(CLASSES)}


def run_change(
    data_t1: np.ndarray,
    meta_t1: dict,
    data_t2: np.ndarray,
    meta_t2: dict,
    query: str,
    bundle: EvidenceBundle,
    params: dict,
) -> ToolResult:
    threshold_method = str(params.get("threshold_method", "otsu"))
    smoothing = int(params.get("smoothing", 3))

    # ── Co-registration ────────────────────────────────────────────────────────
    coreg_report = {}
    if meta_t1.get("crs") and meta_t2.get("crs"):
        data_t2_reg, method = reproject_to_grid(data_t2, meta_t2, meta_t1)
        coreg_report["method"] = method
        shift_px = 0.0
    else:
        data_t2_reg, coreg_report = orb_ransac_register(data_t2, data_t1)
        shift_px = coreg_report.get("shift_px") or 0.0
        coreg_report["note"] = "ORB+RANSAC fallback"

    # Histogram matching
    data_t2_matched = histogram_match(data_t2_reg, data_t1)

    # ── Compute indices ────────────────────────────────────────────────────────
    idx_t1 = compute_all_indices(data_t1, meta_t1)
    idx_t2 = compute_all_indices(data_t2_matched, meta_t1)  # use t1 meta after reprojection

    # ── Change vector analysis ─────────────────────────────────────────────────
    # Stack common indices
    common = sorted(set(idx_t1.keys()) & set(idx_t2.keys()))
    if not common:
        return ToolResult(
            tool="change",
            provenance="classical-cv-baseline",
            answer="Insufficient spectral bands for change analysis.",
            measurements={},
        )

    t1_stack = np.stack([idx_t1[k] for k in common], axis=0)  # [C, H, W]
    t2_stack = np.stack([idx_t2[k] for k in common], axis=0)

    # Align shapes (crop to min)
    h = min(t1_stack.shape[1], t2_stack.shape[1])
    w = min(t1_stack.shape[2], t2_stack.shape[2])
    t1_stack = t1_stack[:, :h, :w]
    t2_stack = t2_stack[:, :h, :w]

    diff = t2_stack - t1_stack
    magnitude = np.sqrt((diff ** 2).sum(axis=0))  # [H, W]

    # Threshold
    valid_mag = magnitude[np.isfinite(magnitude)]
    if threshold_method == "otsu":
        thr = float(threshold_otsu(valid_mag)) if valid_mag.size > 1 else float(np.median(valid_mag))
    elif threshold_method == "percentile95":
        thr = float(np.percentile(valid_mag, 95))
    else:
        thr = float(np.mean(valid_mag))

    sep = score_otsu_separation(magnitude, thr)
    change_mask = (magnitude > thr)
    selem = disk(smoothing)
    change_mask = opening(change_mask, selem)

    changed_fraction = float(change_mask.sum() / change_mask.size)

    # ── Land-cover classification before/after ─────────────────────────────────
    lc_t1 = classify_pixels(idx_t1, meta_t1)[:h, :w]
    lc_t2 = classify_pixels(idx_t2, meta_t1)[:h, :w]

    fracs_t1 = _class_fractions(lc_t1)
    fracs_t2 = _class_fractions(lc_t2)

    # ── Transition matrix ──────────────────────────────────────────────────────
    nc = len(CLASSES)
    trans_matrix = np.zeros((nc, nc), dtype=np.int32)
    lc_t1_flat = lc_t1.flatten()
    lc_t2_flat = lc_t2.flatten()
    for i in range(nc):
        for j in range(nc):
            trans_matrix[i, j] = int(((lc_t1_flat == i) & (lc_t2_flat == j)).sum())

    # Dominant transitions (exclude same-class)
    dominant_transitions = []
    for i in range(nc):
        for j in range(nc):
            if i != j and trans_matrix[i, j] > 0:
                dom_frac = trans_matrix[i, j] / max(trans_matrix.sum(), 1)
                if dom_frac > 0.005:
                    dominant_transitions.append({
                        "from": CLASSES[i],
                        "to": CLASSES[j],
                        "fraction": round(float(dom_frac), 4),
                    })
    dominant_transitions = sorted(dominant_transitions, key=lambda x: -x["fraction"])[:5]

    # ── Change-VQA: answer increase/decrease/unchanged per class ──────────────
    class_deltas: dict[str, dict] = {}
    for cls in CLASSES:
        delta = fracs_t2.get(cls, 0) - fracs_t1.get(cls, 0)
        tol = CHANGE_UNCHANGED_TOLERANCE
        verdict = "unchanged" if abs(delta) < tol else ("increased" if delta > 0 else "decreased")
        class_deltas[cls] = {
            "t1_fraction": round(fracs_t1.get(cls, 0), 4),
            "t2_fraction": round(fracs_t2.get(cls, 0), 4),
            "delta": round(delta, 4),
            "verdict": verdict,
        }

    # ── Overlay PNG ───────────────────────────────────────────────────────────
    from ..preprocessing import make_rgb_preview
    rgb_t1 = make_rgb_preview(data_t1, meta_t1)
    h_overlay, w_overlay = rgb_t1.shape[:2]
    change_resized = cv2.resize(
        change_mask[:h_overlay, :w_overlay].astype(np.uint8),
        (w_overlay, h_overlay),
    ).astype(bool)
    overlay = rgb_t1.copy()
    overlay[change_resized, 0] = 255
    overlay[change_resized, 1] = 80
    overlay[change_resized, 2] = 0
    _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buf.tobytes()).decode() if _ else ""

    # ── Measurements & evidence ────────────────────────────────────────────────
    measurements: dict[str, Any] = {
        "changed_fraction": round(changed_fraction, 4),
        "threshold": round(thr, 4),
        "threshold_method": threshold_method,
        "separation_score": round(sep, 4),
        "coreg_report": coreg_report,
        "class_deltas": class_deltas,
        "dominant_transitions": dominant_transitions,
        "indices_used": common,
    }

    for cls, d in class_deltas.items():
        bundle.add_measurement(
            name=f"change_{cls}_delta", value=d["delta"],
            method="change-vector-analysis",
            provenance="classical-cv-baseline",
            confidence=round(sep, 3),
        )
        bundle.add_measurement(
            name=f"{cls}_t1_fraction", value=d["t1_fraction"],
            method="spectral-classification",
            provenance="classical-cv-baseline", confidence=0.7,
        )
        bundle.add_measurement(
            name=f"{cls}_t2_fraction", value=d["t2_fraction"],
            method="spectral-classification",
            provenance="classical-cv-baseline", confidence=0.7,
        )
    bundle.add_measurement(
        name="changed_fraction", value=changed_fraction,
        method="change-vector-otsu", provenance="classical-cv-baseline",
        confidence=round(sep, 3),
    )

    # ── Formulate natural-language answer ─────────────────────────────────────
    # Try to answer the specific query if it asks about a class
    q_lower = query.lower()
    targeted_cls = None
    for cls in CLASSES:
        if cls in q_lower or cls.replace("-", "") in q_lower:
            targeted_cls = cls
            break

    if targeted_cls and targeted_cls in class_deltas:
        d = class_deltas[targeted_cls]
        answer = (
            f"The {targeted_cls} area has **{d['verdict']}** between the two dates. "
            f"T1 coverage: {d['t1_fraction']*100:.2f}%, T2 coverage: {d['t2_fraction']*100:.2f}% "
            f"(Δ = {d['delta']*100:+.2f}%). "
        )
    else:
        top_trans = dominant_transitions[:2]
        trans_str = "; ".join(f"{t['from']} → {t['to']} ({t['fraction']*100:.1f}%)" for t in top_trans)
        answer = (
            f"Overall change: {changed_fraction*100:.2f}% of the image changed. "
            f"Dominant transitions: {trans_str if trans_str else 'none significant'}. "
        )

    answer += (
        f"Separation quality: {sep:.3f}. "
        f"Co-registration: {coreg_report.get('method', 'unknown')}."
    )

    coreg_conf = score_coreg_residual(coreg_report.get("shift_px") or 0.0)

    return ToolResult(
        tool="change",
        provenance="classical-cv-baseline",
        answer=answer,
        measurements=measurements,
        overlay_url=f"data:image/png;base64,{overlay_b64}" if overlay_b64 else None,
        extra={
            "coreg_confidence": round(coreg_conf, 3),
            "separation_confidence": round(sep, 3),
        },
    )
