"""
preprocessing.py — Display stretch, co-registration, SAR preprocessing, histogram matching.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import median_filter
import rasterio
from rasterio.warp import reproject, Resampling
from .indices import to_db


# ── Display stretch ───────────────────────────────────────────────────────────

def percentile_stretch(band: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Stretch a 2-D band array to uint8 [0, 255] using percentile clipping."""
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_lo = float(np.percentile(valid, lo))
    p_hi = float(np.percentile(valid, hi))
    if p_hi == p_lo:
        return np.zeros(band.shape, dtype=np.uint8)
    out = np.clip((band - p_lo) / (p_hi - p_lo), 0, 1) * 255
    return out.astype(np.uint8)


def make_rgb_preview(data: np.ndarray, meta: dict) -> np.ndarray:
    """
    Return an [H, W, 3] uint8 RGB array suitable for PNG saving.
    Tries to use red/green/blue or nir/red/green as a false colour.
    """
    from .indices import extract_bands
    b = extract_bands(data, meta)
    if "red" in b and "green" in b and "blue" in b:
        r = percentile_stretch(b["red"])
        g = percentile_stretch(b["green"])
        bl = percentile_stretch(b["blue"])
    elif "nir" in b and "red" in b and "green" in b:
        r = percentile_stretch(b["nir"])
        g = percentile_stretch(b["red"])
        bl = percentile_stretch(b["green"])
    elif "vv" in b:
        grey = percentile_stretch(to_db(b["vv"]))
        r = g = bl = grey
    else:
        grey = percentile_stretch(data[0])
        r = g = bl = grey
    return np.stack([r, g, bl], axis=-1)


# ── SAR preprocessing ─────────────────────────────────────────────────────────

def preprocess_sar(band: np.ndarray, filter_size: int = 5) -> tuple[np.ndarray, dict]:
    """
    Convert to dB and apply median speckle filter.
    Returns (processed_band, params_used).
    """
    db = to_db(band)
    filtered = median_filter(db, size=filter_size).astype(np.float32)
    return filtered, {"filter": "median", "size": filter_size, "unit": "dB"}


# ── Georeferenced co-registration ─────────────────────────────────────────────

def reproject_to_grid(
    src_data: np.ndarray,
    src_meta: dict,
    ref_meta: dict,
) -> tuple[np.ndarray, str]:
    """
    Reproject src onto ref's grid using rasterio.warp.reproject.
    Returns (reprojected_array, method_used).
    """
    src_crs = rasterio.crs.CRS.from_string(src_meta["crs"]) if src_meta.get("crs") else None
    ref_crs = rasterio.crs.CRS.from_string(ref_meta["crs"]) if ref_meta.get("crs") else None
    if not src_crs or not ref_crs:
        return src_data, "skipped-no-crs"

    ref_transform = rasterio.transform.Affine(*ref_meta["transform"][:6])
    dst = np.zeros(
        (src_data.shape[0], ref_meta["height"], ref_meta["width"]),
        dtype=np.float32
    )
    for band_idx in range(src_data.shape[0]):
        src_transform = rasterio.transform.Affine(*src_meta["transform"][:6])
        reproject(
            source=src_data[band_idx],
            destination=dst[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    return dst, "rasterio-reproject"


def orb_ransac_register(
    src: np.ndarray,
    ref: np.ndarray,
    max_features: int = 500,
) -> tuple[np.ndarray, dict]:
    """
    ORB + RANSAC homography fallback for non-georeferenced images.
    src and ref are [C, H, W] float arrays.
    Returns (warped_src, report_dict).
    """
    # Convert first bands to uint8 for ORB
    def to_uint8(arr):
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

    ref_grey = to_uint8(ref[0])
    src_grey = to_uint8(src[0])

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(ref_grey, None)
    kp2, des2 = orb.detectAndCompute(src_grey, None)

    report = {"method": "ORB+RANSAC", "matched": 0, "inliers": 0, "shift_px": None}

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return src, {**report, "note": "not-enough-keypoints"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:100]
    report["matched"] = len(matches)

    if len(matches) < 4:
        return src, {**report, "note": "not-enough-matches"}

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_src = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)
    if H is None:
        return src, {**report, "note": "homography-failed"}

    inliers = int(mask.sum())
    report["inliers"] = inliers

    h, w = ref[0].shape
    warped = np.zeros_like(src)
    for i in range(src.shape[0]):
        warped[i] = cv2.warpPerspective(src[i], H, (w, h))
    report["shift_px"] = float(np.linalg.norm(H[:2, 2]))
    return warped.astype(np.float32), report


# ── Histogram matching for bi-temporal optical pairs ─────────────────────────

def histogram_match_band(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match the histogram of src to ref (1-D band arrays).
    Uses a rank-based cumulative distribution function mapping.
    """
    src_flat = src.flatten()
    ref_flat = ref.flatten()

    src_counts, src_bins = np.histogram(src_flat[np.isfinite(src_flat)], bins=256)
    ref_counts, ref_bins = np.histogram(ref_flat[np.isfinite(ref_flat)], bins=256)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Build mapping
    interp = np.interp(src_cdf, ref_cdf, ref_bins[:-1])
    lut = np.interp(src_flat, src_bins[:-1], interp)
    return lut.reshape(src.shape).astype(np.float32)


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Band-wise histogram matching. src and ref are [C, H, W]."""
    out = np.zeros_like(src)
    for i in range(min(src.shape[0], ref.shape[0])):
        out[i] = histogram_match_band(src[i], ref[i])
    return out
