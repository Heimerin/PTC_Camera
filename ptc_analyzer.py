#!/usr/bin/env python3
"""
Photon Transfer Curve analyzer for astronomical cameras.

Implements the classical Janesick-style PTC workflow:
  - build a master bias / offset frame,
  - subtract it from flats,
  - remove fixed-pattern noise by differencing pairs of flats at the same exposure,
  - estimate read+shot noise from the pair difference divided by sqrt(2),
  - estimate shot variance, ADC sensitivity K_ADC [e-/ADU], conversion gain [ADU/e-],
    read noise, full-well proxy, dynamic range and nonlinearity.

The script can be used with a Tkinter GUI or in command-line mode:

    python ptc_analyzer.py
    python ptc_analyzer.py --no-gui --camera "Cam A" --bias ./bias --flats ./flats --output ./ptc_results

Supported input formats:
  - FITS/FIT/FTS if astropy is installed,
  - TIFF/PNG/JPEG if imageio or Pillow is installed,
  - NPY arrays, useful for testing.

Author: Perplexity Computer
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required: pip install matplotlib") from exc


SUPPORTED_EXTENSIONS = {
    ".fit",
    ".fits",
    ".fts",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".npy",
}

EXPOSURE_HEADER_KEYS = (
    "EXPTIME",
    "EXPOSURE",
    "EXP_TIME",
    "EXP",
    "ITIME",
    "INTTIME",
    "EXPOS",
)

DEFAULT_EXPOSURE_REGEXES = (
    r"(?:exp|exptime|exposure|czas|t|ms|s)[_\-\s=]*([0-9]+(?:[.,][0-9]+)?)\s*(ms|s|sec|secs|second|seconds)?",
    r"([0-9]+(?:[.,][0-9]+)?)\s*(ms|s|sec|secs|second|seconds)",
)


@dataclass
class AnalysisConfig:
    camera_name: str
    bias_dir: str
    flat_dir: str
    output_dir: str
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    color_mode: str = "mono"
    exposure_regex: Optional[str] = None
    fit_low_fraction: float = 0.10
    fit_high_fraction: float = 0.70
    linearity_threshold_percent: float = 1.0
    saturation_adu: Optional[float] = None
    max_plotted_points: int = 10000


@dataclass
class PTCPoint:
    exposure: float
    pair_index: int
    signal_dn: float
    total_noise_dn: float
    read_shot_noise_dn: float
    read_noise_dn: float
    shot_noise_dn: float
    fpn_noise_dn: float
    shot_variance_dn2: float
    k_adc_e_per_dn_point: float
    conversion_gain_dn_per_e_point: float
    nonlinearity_k_percent: float
    signal_linearity_percent: float
    max_raw_adu: float
    n_pixels: int
    file_1: str
    file_2: str


@dataclass
class PTCResults:
    camera_name: str
    color_mode: str
    n_bias: int
    n_flat_files: int
    n_pairs: int
    n_pixels: int
    master_bias_mean_dn: float
    read_noise_dn: float
    read_noise_e: float
    k_adc_e_per_dn: float
    conversion_gain_dn_per_e: float
    ptc_fit_slope_dn_per_e: float
    ptc_fit_intercept_dn2: float
    ptc_fit_r2: float
    shot_noise_loglog_slope: float
    fpn_loglog_slope: float
    k_low_e_per_dn: float
    max_abs_k_nonlinearity_percent: float
    max_abs_signal_linearity_percent: float
    full_well_dn_observed: float
    full_well_e_observed: float
    full_well_dn_ptc_turnover: float
    full_well_e_ptc_turnover: float
    full_well_dn_at_linearity_limit: float
    full_well_e_at_linearity_limit: float
    dynamic_range: float
    dynamic_range_db: float
    fpn_percent_median: float
    points: List[PTCPoint]


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+(?:\.\d+)?)", path.name.lower())
    key: List[object] = []
    for part in parts:
        try:
            key.append(float(part))
        except ValueError:
            key.append(part)
    return key


def list_image_files(folder: str) -> List[Path]:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=natural_sort_key)


def parse_roi(roi_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_text:
        return None
    clean = roi_text.replace(" ", "")
    if not clean:
        return None
    vals = re.split(r"[,;xX]", clean)
    if len(vals) != 4:
        raise ValueError("ROI must be x,y,width,height, for example: 100,120,800,600")
    x, y, w, h = [int(v) for v in vals]
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    return x, y, w, h


def apply_roi(data: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return data
    x, y, w, h = roi
    if data.ndim > 2:
        data = data[..., 0]
    if y + h > data.shape[0] or x + w > data.shape[1]:
        raise ValueError(
            f"ROI {roi} exceeds image shape {data.shape}; remember ROI is x,y,width,height"
        )
    return data[y : y + h, x : x + w]


def extract_color_plane(data: np.ndarray, color_mode: str) -> np.ndarray:
    mode = (color_mode or "mono").lower().strip()
    if mode in {"mono", "none"}:
        if data.ndim > 2:
            return np.asarray(data[..., 0], dtype=np.float64)
        return np.asarray(data, dtype=np.float64)

    if data.ndim >= 3:
        if mode in {"first", "channel_0"}:
            return np.asarray(data[..., 0], dtype=np.float64)
        if mode in {"red", "rgb_red", "r"}:
            return np.asarray(data[..., 0], dtype=np.float64)
        if mode in {"green", "rgb_green", "g"}:
            return np.asarray(data[..., 1], dtype=np.float64)
        if mode in {"blue", "rgb_blue", "b"}:
            return np.asarray(data[..., 2], dtype=np.float64)
        if mode in {"luminance", "rgb_luminance"}:
            return 0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]

    if data.ndim != 2:
        raise ValueError(f"Unsupported image dimensions {data.shape} for color mode {color_mode}")

    bayer_offsets = {
        "bayer_rggb_red": (0, 0),
        "bayer_rggb_green1": (0, 1),
        "bayer_rggb_green2": (1, 0),
        "bayer_rggb_blue": (1, 1),
        "bayer_bggr_red": (1, 1),
        "bayer_bggr_green1": (0, 1),
        "bayer_bggr_green2": (1, 0),
        "bayer_bggr_blue": (0, 0),
        "bayer_grbg_red": (0, 1),
        "bayer_grbg_green1": (0, 0),
        "bayer_grbg_green2": (1, 1),
        "bayer_grbg_blue": (1, 0),
        "bayer_gbrg_red": (1, 0),
        "bayer_gbrg_green1": (0, 0),
        "bayer_gbrg_green2": (1, 1),
        "bayer_gbrg_blue": (0, 1),
    }
    if mode in bayer_offsets:
        row, col = bayer_offsets[mode]
        return np.asarray(data[row::2, col::2], dtype=np.float64)

    raise ValueError(
        f"Unknown color mode '{color_mode}'. Use mono, red, green, blue, luminance, "
        "or bayer_rggb_red / bayer_bggr_red / bayer_grbg_red / bayer_gbrg_red."
    )


def read_image(
    path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    color_mode: str = "mono",
) -> Tuple[np.ndarray, Dict[str, object]]:
    suffix = path.suffix.lower()
    header: Dict[str, object] = {}
    if suffix in {".fit", ".fits", ".fts"}:
        try:
            from astropy.io import fits
        except Exception as exc:
            raise RuntimeError(
                "Reading FITS requires astropy. Install with: pip install astropy"
            ) from exc
        with fits.open(path, memmap=False) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                raise ValueError(f"No image HDU found in {path}")
            data = np.asarray(hdu.data, dtype=np.float64)
            header = dict(hdu.header)
    elif suffix == ".npy":
        data = np.asarray(np.load(path), dtype=np.float64)
    else:
        try:
            import imageio.v3 as iio

            data = np.asarray(iio.imread(path), dtype=np.float64)
        except Exception:
            try:
                from PIL import Image

                with Image.open(path) as im:
                    data = np.asarray(im, dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read {path}. Install imageio or Pillow for non-FITS images."
                ) from exc
    data = extract_color_plane(data, color_mode)
    data = apply_roi(data, roi)
    return np.asarray(data, dtype=np.float64), header


def extract_exposure_seconds(
    path: Path, header: Dict[str, object], exposure_regex: Optional[str] = None
) -> float:
    for key in EXPOSURE_HEADER_KEYS:
        if key in header:
            try:
                return float(header[key])
            except Exception:
                pass

    regexes = [exposure_regex] if exposure_regex else []
    regexes.extend(DEFAULT_EXPOSURE_REGEXES)
    filename = path.stem.lower().replace(",", ".")
    for pattern in regexes:
        if not pattern:
            continue
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1).replace(",", "."))
            unit = match.group(2).lower() if match.lastindex and match.lastindex >= 2 and match.group(2) else "s"
            if unit == "ms":
                return value / 1000.0
            return value
    raise ValueError(
        f"Could not determine exposure time for {path.name}. "
        "Use FITS EXPTIME/EXPOSURE headers or include exposure in filename, e.g. flat_250ms_001.fit."
    )


def robust_std(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(vals, ddof=1))


def mean_stack(paths: Sequence[Path], roi: Optional[Tuple[int, int, int, int]], color_mode: str) -> np.ndarray:
    if not paths:
        raise ValueError("No image files found")
    acc = None
    for idx, path in enumerate(paths):
        data, _ = read_image(path, roi, color_mode)
        if acc is None:
            acc = np.zeros_like(data, dtype=np.float64)
        if data.shape != acc.shape:
            raise ValueError(f"Shape mismatch for {path}: {data.shape} vs {acc.shape}")
        acc += data
    assert acc is not None
    return acc / float(len(paths))


def estimate_read_noise_from_biases(
    bias_files: Sequence[Path],
    master_bias: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    color_mode: str,
) -> float:
    if len(bias_files) >= 2:
        noises = []
        for i in range(0, len(bias_files) - 1, 2):
            b1, _ = read_image(bias_files[i], roi, color_mode)
            b2, _ = read_image(bias_files[i + 1], roi, color_mode)
            noises.append(np.std(b1 - b2, ddof=1) / math.sqrt(2.0))
        if noises:
            return float(np.median(noises))
    return float(np.std(master_bias, ddof=1))


def group_flats_by_exposure(
    flat_files: Sequence[Path],
    roi: Optional[Tuple[int, int, int, int]],
    exposure_regex: Optional[str],
) -> Dict[float, List[Tuple[Path, Dict[str, object]]]]:
    groups: Dict[float, List[Tuple[Path, Dict[str, object]]]] = {}
    for path in flat_files:
        _, header = read_image(path, roi=None)
        exposure = extract_exposure_seconds(path, header, exposure_regex)
        exposure_key = round(float(exposure), 9)
        groups.setdefault(exposure_key, []).append((path, header))
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def pairwise(items: Sequence[Tuple[Path, Dict[str, object]]]) -> List[Tuple[Tuple[Path, Dict[str, object]], Tuple[Path, Dict[str, object]]]]:
    pairs = []
    for i in range(0, len(items) - 1, 2):
        pairs.append((items[i], items[i + 1]))
    return pairs


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def loglog_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    slope, _intercept, _r2 = linear_fit(np.log10(x[mask]), np.log10(y[mask]))
    return float(slope)


def through_origin_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan"), float("nan")
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum(x * y) / denom)
    yhat = slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def select_fit_mask(
    signal: np.ndarray,
    variance: np.ndarray,
    fit_low_fraction: float,
    fit_high_fraction: float,
    saturation_adu: Optional[float],
) -> np.ndarray:
    mask = np.isfinite(signal) & np.isfinite(variance) & (signal > 0) & (variance > 0)
    if not np.any(mask):
        return mask
    max_signal = float(np.nanmax(signal[mask]))
    low = max_signal * max(0.0, min(0.95, fit_low_fraction))
    high = max_signal * max(0.05, min(1.0, fit_high_fraction))
    if high <= low:
        low, high = max_signal * 0.10, max_signal * 0.70
    mask &= signal >= low
    mask &= signal <= high
    if saturation_adu is not None:
        mask &= signal <= 0.85 * saturation_adu
    if np.count_nonzero(mask) < 3:
        valid = np.where(np.isfinite(signal) & np.isfinite(variance) & (signal > 0) & (variance > 0))[0]
        if valid.size >= 3:
            lo_idx = max(0, int(valid.size * 0.15))
            hi_idx = max(lo_idx + 3, int(valid.size * 0.75))
            fallback = np.zeros_like(mask, dtype=bool)
            fallback[valid[lo_idx:hi_idx]] = True
            return fallback
    return mask


def choose_k_low(signal: np.ndarray, k_values: np.ndarray) -> float:
    mask = np.isfinite(signal) & np.isfinite(k_values) & (signal > 0) & (k_values > 0)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    sig = signal[mask]
    kval = k_values[mask]
    order = np.argsort(sig)
    n = max(1, min(5, int(math.ceil(len(order) * 0.15))))
    return float(np.median(kval[order[:n]]))


def estimate_ptc_turnover(signal: np.ndarray, read_shot_noise: np.ndarray) -> float:
    mask = np.isfinite(signal) & np.isfinite(read_shot_noise) & (signal > 0) & (read_shot_noise > 0)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    sig = signal[mask]
    noise = read_shot_noise[mask]
    order = np.argsort(sig)
    sig = sig[order]
    noise = noise[order]
    peak_idx = int(np.argmax(noise))
    if peak_idx < len(sig) - 1:
        return float(sig[peak_idx])
    return float(np.nanmax(sig))


def save_csv(path: Path, points: Sequence[PTCPoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()) if points else [])
        if points:
            writer.writeheader()
            for p in points:
                writer.writerow(asdict(p))


def save_summary_json(path: Path, results: PTCResults) -> None:
    payload = asdict(results)
    payload["points"] = [asdict(p) for p in results.points]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def plot_ptc(results: PTCResults, output_dir: Path) -> None:
    setup_plot_style()
    pts = results.points
    signal_adu = np.array([p.signal_dn for p in pts])
    signal_e = signal_adu * results.k_adc_e_per_dn
    total_e = np.array([p.total_noise_dn for p in pts]) * results.k_adc_e_per_dn
    below_fwc = signal_adu <= results.full_well_dn_ptc_turnover
    if not np.any(below_fwc):
        below_fwc = np.ones_like(signal_adu, dtype=bool)

    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    ax.loglog(signal_e[below_fwc], total_e[below_fwc], "o", ms=4.8, label="total noise", color="#20808D", alpha=0.9)
    if np.any(~below_fwc):
        ax.loglog(signal_e[~below_fwc], total_e[~below_fwc], "o", ms=4.8, label="total noise > FWC", color="#7A7974", alpha=0.65)

    xs = np.logspace(np.log10(np.nanmin(signal_e[signal_e > 0])), np.log10(np.nanmax(signal_e)), 250)
    if np.isfinite(results.read_noise_e):
        ax.axhline(results.read_noise_e, color="#7A7974", lw=1.2, ls="--", label=f"read noise {results.read_noise_e:.3g} e-")
    ax.loglog(xs, np.sqrt(xs), "-", color="#1B474D", lw=1.7, label="shot noise, slope 1/2")
    pn = results.fpn_percent_median / 100.0
    if np.isfinite(pn) and pn > 0:
        ax.loglog(xs, pn * xs, "-", color="#A84B2F", lw=1.7, label=f"FPN, slope 1")
    if np.isfinite(results.full_well_e_ptc_turnover):
        ax.axvline(results.full_well_e_ptc_turnover, color="#964219", lw=1, ls=":", label=f"FWC~{results.full_well_e_ptc_turnover:.3g} e-")
    ax.set_title(f"Photon Transfer Curve: {results.camera_name}")
    ax.set_xlabel("Mean signal after bias subtraction [e-]")
    ax.set_ylabel("Noise RMS [e-]")
    ax.legend(loc="best")
    fig.savefig(output_dir / "ptc_loglog.png", bbox_inches="tight")
    plt.close(fig)


def plot_variance_fit(results: PTCResults, output_dir: Path) -> None:
    setup_plot_style()
    pts = results.points
    signal = np.array([p.signal_dn for p in pts])
    variance = np.array([p.read_shot_noise_dn**2 for p in pts])
    shot_variance = np.array([p.shot_variance_dn2 for p in pts])

    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    ax.plot(signal, variance, "o", color="#20808D", label="N² read+shot [ADU²]")
    ax.plot(signal, shot_variance, "o", color="#1B474D", alpha=0.55, label="N²-R² shot variance [ADU²]")
    xs = np.linspace(np.nanmin(signal), np.nanmax(signal), 200)
    if np.isfinite(results.ptc_fit_slope_dn_per_e):
        ys = results.ptc_fit_slope_dn_per_e * xs + results.ptc_fit_intercept_dn2
        ax.plot(xs, ys, "-", color="#A84B2F", label=f"fit: slope={results.ptc_fit_slope_dn_per_e:.4g}, R²={results.ptc_fit_r2:.4f}")
    ax.set_title(f"Variance PTC fit: {results.camera_name}")
    ax.set_xlabel("Mean signal after bias subtraction [ADU]")
    ax.set_ylabel("Variance [ADU²]")
    ax.legend(loc="best")
    fig.savefig(output_dir / "variance_fit.png", bbox_inches="tight")
    plt.close(fig)


def plot_nonlinearity(results: PTCResults, output_dir: Path) -> None:
    setup_plot_style()
    pts = results.points
    signal = np.array([p.signal_dn for p in pts])
    k = np.array([p.k_adc_e_per_dn_point for p in pts])
    nlk = np.array([p.nonlinearity_k_percent for p in pts])
    #siglin = np.array([p.signal_linearity_percent for p in pts])   #do wyjebania 1

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), layout="constrained", sharex=True)
    axes[0].plot(signal, k, "o-", color="#20808D", ms=4)
    axes[0].axhline(results.k_low_e_per_dn, color="#7A7974", lw=1, ls="--", label=f"K low={results.k_low_e_per_dn:.4g} e-/ADU")
    axes[0].set_ylabel("K_ADC [e-/ADU]")
    axes[0].set_title(f"K_ADC nonlinearity: {results.camera_name}")
    axes[0].legend(loc="best")

    axes[1].plot(signal, nlk, "o-", color="#A84B2F", ms=4, label="PTC K nonlinearity")
    axes[1].axhline(results.max_abs_k_nonlinearity_percent, color="#A84B2F", lw=0.7, ls=":")
    axes[1].axhline(-results.max_abs_k_nonlinearity_percent, color="#A84B2F", lw=0.7, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mean signal after bias subtraction [ADU]")
    axes[1].set_ylabel("Nonlinearity [%]")
    axes[1].legend(loc="best")
    fig.savefig(output_dir / "nonlinearity.png", bbox_inches="tight")
    plt.close(fig)


def plot_signal_linearity(results: PTCResults, output_dir: Path) -> None:
    setup_plot_style()
    pts = results.points
    exposure = np.array([p.exposure for p in pts])
    signal = np.array([p.signal_dn for p in pts])
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), layout="constrained", sharex=True)
    axes[0].plot(exposure, signal, "o-", color="#20808D")
    axes[0].set_ylabel("Signal [ADU]")
    axes[0].set_title(f"Signal linearity vs exposure: {results.camera_name}")
    axes[1].plot(exposure, [p.signal_linearity_percent for p in pts], "o-", color="#A84B2F")
    axes[1].axhline(results.max_abs_signal_linearity_percent, color="#A84B2F", lw=0.7, ls=":")
    axes[1].axhline(-results.max_abs_signal_linearity_percent, color="#A84B2F", lw=0.7, ls=":")
    axes[1].set_xlabel("Exposure [s]")
    axes[1].set_ylabel("Residual [%]")
    fig.savefig(output_dir / "signal_linearity.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(results: PTCResults, output_dir: Path) -> None:
    report_path = output_dir / "report.md"

    def fmt_units(adu_val, e_val):
        ke_val = e_val / 1000.0 if np.isfinite(e_val) else float('nan')
        return f"{adu_val:.6g} ADU = {e_val:.6g} e- ({ke_val:.4g} ke-)"
    
    lines = [
        f"# Photon Transfer Curve report: {results.camera_name}",
        "",
        "## Summary",
        "",
        f"- Bias frames: {results.n_bias}",
        f"- Color mode: {results.color_mode}",
        f"- Flat files: {results.n_flat_files}",
        f"- Flat pairs used: {results.n_pairs}",
        f"- Pixels per measurement: {results.n_pixels}",
        f"- Master bias mean: {results.master_bias_mean_dn:.6g} ADU",
        f"- ADC sensitivity K_ADC: {results.k_adc_e_per_dn:.6g} e-/ADU",
        f"- Conversion gain: {results.conversion_gain_dn_per_e:.6g} ADU/e-",
        f"- Read noise: {results.read_noise_dn:.6g} ADU = {results.read_noise_e:.6g} e-",
        f"- Observed full-well proxy: {results.full_well_dn_observed:.6g} ADU = {results.full_well_e_observed:.6g} e-",
        f"- PTC-turnover full-well proxy: {results.full_well_dn_ptc_turnover:.6g} ADU = {results.full_well_e_ptc_turnover:.6g} e-",
        f"- Full well within linearity threshold: {results.full_well_dn_at_linearity_limit:.6g} ADU = {results.full_well_e_at_linearity_limit:.6g} e-",
        f"- Dynamic range: {results.dynamic_range:.6g} = {results.dynamic_range_db:.3f} dB",
        f"- Max |K nonlinearity|: {results.max_abs_k_nonlinearity_percent:.4g} %",
        f"- Max |signal-vs-exposure residual|: {results.max_abs_signal_linearity_percent:.4g} %",
        f"- Median FPN quality factor: {results.fpn_percent_median:.4g} %",
        f"- Shot noise log-log slope below FWC: {results.shot_noise_loglog_slope:.4g} (ideal: 0.5)",
        f"- FPN log-log slope below FWC: {results.fpn_loglog_slope:.4g} (ideal: 1.0)",
        "",
        "## Method notes",
        "",
        "- Signal is the mean flat level after pixel-by-pixel master bias subtraction.",
        "- Fixed-pattern noise is suppressed by subtracting two flats at the same exposure. The RMS of that difference is divided by sqrt(2) to obtain read+shot noise.",
        "- Shot variance is computed as `(read+shot noise)^2 - (read noise)^2`.",
        "- K_ADC is computed point-wise as `signal / shot_variance`; the reported K_ADC is also obtained from a linear variance fit where slope is approximately `1 / K_ADC`.",
        "- K nonlinearity follows the Janesick relation `100 * (K_ADC - K_LOW) / K_LOW`.",
        "- The log-log PTC chart is plotted in electrons. Points above the PTC-turnover FWC proxy remain visible, but are not used for the reported shot/FPN slope estimates.",
        "- Full well from ordinary flats is estimated as a proxy: the observed maximum signal and the point where the PTC noise curve turns over. Confirm with exposure series reaching true saturation.",
        "",
        "## Output files",
        "",
        "- `points.csv`: all per-exposure measurements.",
        "- `summary.json`: machine-readable summary and all points.",
        "- `ptc_loglog.png`: classical PTC in electron units.",
        "- `variance_fit.png`: variance PTC and fit.",
        "- `nonlinearity.png`: K_ADC and nonlinearity curves.",
        "- `signal_linearity.png`: signal vs exposure and residuals.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


#nieliniowowsc vv

def calculate_vv_correction(exposure: np.ndarray, signal_dn: np.ndarray, degree: int = 3) -> np.ndarray:
    valid = np.isfinite(exposure) & np.isfinite(signal_dn) & (exposure > 0)
    exp_valid = exposure[valid]
    sig_valid = signal_dn[valid]
    
    linear_region = sig_valid < np.percentile(sig_valid, 20)
    slope, _ = through_origin_fit(exp_valid[linear_region], sig_valid[linear_region])
    
    ideal_signal = slope * exp_valid
    coeffs = np.polyfit(sig_valid, ideal_signal, degree)
    return coeffs

def vv_correction(image_data: np.ndarray, correction_coeffs: np.ndarray) -> np.ndarray:
    corrected_image = np.polyval(correction_coeffs, image_data)
    return corrected_image

#kurwa
from matplotlib.widgets import SpanSelector

def auto_detect_linear_range(signal: np.ndarray, k_points: np.ndarray) -> Tuple[float, float]:
    window_size = max(5, len(signal) // 10)
    min_std = float('inf')
    best_idx = 0
    
    for i in range(len(signal) - window_size):
        window_k = k_points[i:i+window_size]
        current_std = np.std(window_k)
        if current_std < min_std:
            min_std = current_std
            best_idx = i
            
    low_bound = signal[best_idx]
    high_bound = signal[best_idx + window_size - 1]
    return low_bound, high_bound

def interactive_range_selection(signal: np.ndarray, k_points: np.ndarray, camera_name: str) -> Tuple[float, float]:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(signal, k_points, "o-", color="#20808D")
    ax.set_title(f"Obszar liniowy dla: {camera_name}")
    ax.set_xlabel("Sygnał [ADU]")
    ax.set_ylabel("K_ADC [e-/ADU]")
    
    selected_range = [np.nanmin(signal), np.nanmax(signal)]
    
    def onselect(xmin, xmax):
        selected_range[0] = xmin
        selected_range[1] = xmax
        
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        props=dict(alpha=0.3, facecolor='red'))
    plt.show(block=True)
    return selected_range[0], selected_range[1]


def analyze_camera(config: AnalysisConfig) -> PTCResults:
    bias_files = list_image_files(config.bias_dir)
    flat_files = list_image_files(config.flat_dir)
    if not bias_files:
        raise ValueError(f"No bias files found in {config.bias_dir}")
    if not flat_files:
        raise ValueError(f"No flat files found in {config.flat_dir}")

    output_dir = Path(config.output_dir).expanduser() / safe_name(config.camera_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_bias = mean_stack(bias_files, config.roi, config.color_mode)
    read_noise_dn = estimate_read_noise_from_biases(bias_files, master_bias, config.roi, config.color_mode)
    n_pixels = int(master_bias.size)

    groups = group_flats_by_exposure(flat_files, config.roi, config.exposure_regex)
    raw_points: List[Dict[str, object]] = []

    for exposure, items in groups.items():
        if len(items) < 2:
            print(f"Warning: exposure {exposure:g}s has only one flat; skipping", file=sys.stderr)
            continue
        for pair_index, ((p1, _h1), (p2, _h2)) in enumerate(pairwise(items), start=1):
            d1_raw, _ = read_image(p1, config.roi, config.color_mode)
            d2_raw, _ = read_image(p2, config.roi, config.color_mode)
            if d1_raw.shape != master_bias.shape or d2_raw.shape != master_bias.shape:
                raise ValueError(f"Shape mismatch in flat pair {p1.name}, {p2.name}")

            d1 = d1_raw - master_bias
            d2 = d2_raw - master_bias
            avg = 0.5 * (d1 + d2)
            signal_dn = float(np.mean(avg))
            total_noise_dn = float(0.5 * (np.std(d1, ddof=1) + np.std(d2, ddof=1)))
            read_shot_noise_dn = float(np.std(d1 - d2, ddof=1) / math.sqrt(2.0))
            shot_variance = float(read_shot_noise_dn**2 - read_noise_dn**2)
            shot_variance = max(shot_variance, float("nan") if not np.isfinite(shot_variance) else shot_variance)
            shot_noise_dn = math.sqrt(shot_variance) if shot_variance > 0 else float("nan")
            fpn_var = total_noise_dn**2 - read_shot_noise_dn**2
            fpn_noise_dn = math.sqrt(fpn_var) if fpn_var > 0 else float("nan")
            k_point = signal_dn / shot_variance if shot_variance > 0 and signal_dn > 0 else float("nan")
            cg_point = 1.0 / k_point if k_point > 0 and np.isfinite(k_point) else float("nan")
            raw_points.append(
                {
                    "exposure": exposure,
                    "pair_index": pair_index,
                    "signal_dn": signal_dn,
                    "total_noise_dn": total_noise_dn,
                    "read_shot_noise_dn": read_shot_noise_dn,
                    "read_noise_dn": read_noise_dn,
                    "shot_noise_dn": shot_noise_dn,
                    "fpn_noise_dn": fpn_noise_dn,
                    "shot_variance_dn2": shot_variance,
                    "k_adc_e_per_dn_point": k_point,
                    "conversion_gain_dn_per_e_point": cg_point,
                    "max_raw_adu": float(max(np.max(d1_raw), np.max(d2_raw))),
                    "n_pixels": n_pixels,
                    "file_1": str(p1),
                    "file_2": str(p2),
                }
            )

    if not raw_points:
        raise ValueError("No flat pairs were available. Need at least two flats per exposure.")

    raw_points.sort(key=lambda p: (float(p["exposure"]), float(p["signal_dn"])))
    signal = np.array([float(p["signal_dn"]) for p in raw_points])
    read_shot_noise = np.array([float(p["read_shot_noise_dn"]) for p in raw_points])
    read_shot_variance = read_shot_noise**2
    shot_variance = np.array([float(p["shot_variance_dn2"]) for p in raw_points])
    k_points = np.array([float(p["k_adc_e_per_dn_point"]) for p in raw_points])

    fit_mask = select_fit_mask(
        signal=signal,
        variance=read_shot_variance,
        fit_low_fraction=config.fit_low_fraction,
        fit_high_fraction=config.fit_high_fraction,
        saturation_adu=config.saturation_adu,
    )
    slope, intercept, r2 = linear_fit(signal[fit_mask], read_shot_variance[fit_mask])
    if np.isfinite(slope) and slope > 0:
        k_adc = 1.0 / slope
    else:
        valid_k = k_points[np.isfinite(k_points) & (k_points > 0)]
        k_adc = float(np.median(valid_k)) if valid_k.size else float("nan")
    conversion_gain = 1.0 / k_adc if np.isfinite(k_adc) and k_adc > 0 else float("nan")

    k_low = choose_k_low(signal, k_points)
    nlk = np.full_like(signal, np.nan, dtype=np.float64)
    if np.isfinite(k_low) and k_low > 0:
        nlk = 100.0 * (k_points - k_low) / k_low

    exposure = np.array([float(p["exposure"]) for p in raw_points])
    sig_slope, _sig_r2 = through_origin_fit(exposure[fit_mask], signal[fit_mask])
    signal_linearity = np.full_like(signal, np.nan, dtype=np.float64)
    if np.isfinite(sig_slope):
        expected = sig_slope * exposure
        mask_expected = np.isfinite(expected) & (np.abs(expected) > 0)
        signal_linearity[mask_expected] = 100.0 * (signal[mask_expected] - expected[mask_expected]) / expected[mask_expected]

    max_abs_k_nl = float(np.nanmax(np.abs(nlk[np.isfinite(nlk)]))) if np.any(np.isfinite(nlk)) else float("nan")
    max_abs_sig_nl = (
        float(np.nanmax(np.abs(signal_linearity[np.isfinite(signal_linearity)])))
        if np.any(np.isfinite(signal_linearity))
        else float("nan")
    )

    observed_fw_dn = float(np.nanmax(signal))
    observed_fw_e = observed_fw_dn * k_adc if np.isfinite(k_adc) else float("nan")
    turnover_fw_dn = estimate_ptc_turnover(signal, read_shot_noise)
    turnover_fw_e = turnover_fw_dn * k_adc if np.isfinite(k_adc) else float("nan")

    linear_mask = np.isfinite(signal) & np.isfinite(nlk) & (np.abs(nlk) <= config.linearity_threshold_percent)
    if np.any(linear_mask):
        fw_lin_dn = float(np.nanmax(signal[linear_mask]))
    else:
        fw_lin_dn = float("nan")
    fw_lin_e = fw_lin_dn * k_adc if np.isfinite(k_adc) and np.isfinite(fw_lin_dn) else float("nan")

    read_noise_e = read_noise_dn * k_adc if np.isfinite(k_adc) else float("nan")
    dynamic_range = observed_fw_e / read_noise_e if read_noise_e and read_noise_e > 0 else float("nan")
    dynamic_range_db = 20.0 * math.log10(dynamic_range) if dynamic_range and dynamic_range > 0 else float("nan")

    fpn = np.array([float(p["fpn_noise_dn"]) for p in raw_points])
    fpn_percent = 100.0 * fpn / signal
    fpn_percent_median = float(np.nanmedian(fpn_percent[np.isfinite(fpn_percent)])) if np.any(np.isfinite(fpn_percent)) else float("nan")
    below_fwc_mask = np.isfinite(signal) & (signal > 0) & (signal <= turnover_fw_dn)
    if np.count_nonzero(below_fwc_mask) < 3:
        below_fwc_mask = fit_mask
    shot_slope = loglog_slope(signal * k_adc, np.array([float(p["shot_noise_dn"]) for p in raw_points]) * k_adc, below_fwc_mask)
    fpn_slope = loglog_slope(signal * k_adc, fpn * k_adc, below_fwc_mask)

    points: List[PTCPoint] = []
    for i, raw in enumerate(raw_points):
        points.append(
            PTCPoint(
                exposure=float(raw["exposure"]),
                pair_index=int(raw["pair_index"]),
                signal_dn=float(raw["signal_dn"]),
                total_noise_dn=float(raw["total_noise_dn"]),
                read_shot_noise_dn=float(raw["read_shot_noise_dn"]),
                read_noise_dn=float(raw["read_noise_dn"]),
                shot_noise_dn=float(raw["shot_noise_dn"]),
                fpn_noise_dn=float(raw["fpn_noise_dn"]),
                shot_variance_dn2=float(raw["shot_variance_dn2"]),
                k_adc_e_per_dn_point=float(raw["k_adc_e_per_dn_point"]),
                conversion_gain_dn_per_e_point=float(raw["conversion_gain_dn_per_e_point"]),
                nonlinearity_k_percent=float(nlk[i]),
                signal_linearity_percent=float(signal_linearity[i]),
                max_raw_adu=float(raw["max_raw_adu"]),
                n_pixels=int(raw["n_pixels"]),
                file_1=str(raw["file_1"]),
                file_2=str(raw["file_2"]),
            )
        )

    results = PTCResults(
        camera_name=config.camera_name,
        color_mode=config.color_mode,
        n_bias=len(bias_files),
        n_flat_files=len(flat_files),
        n_pairs=len(points),
        n_pixels=n_pixels,
        master_bias_mean_dn=float(np.mean(master_bias)),
        read_noise_dn=float(read_noise_dn),
        read_noise_e=float(read_noise_e),
        k_adc_e_per_dn=float(k_adc),
        conversion_gain_dn_per_e=float(conversion_gain),
        ptc_fit_slope_dn_per_e=float(slope),
        ptc_fit_intercept_dn2=float(intercept),
        ptc_fit_r2=float(r2),
        shot_noise_loglog_slope=float(shot_slope),
        fpn_loglog_slope=float(fpn_slope),
        k_low_e_per_dn=float(k_low),
        max_abs_k_nonlinearity_percent=max_abs_k_nl,
        max_abs_signal_linearity_percent=max_abs_sig_nl,
        full_well_dn_observed=observed_fw_dn,
        full_well_e_observed=observed_fw_e,
        full_well_dn_ptc_turnover=turnover_fw_dn,
        full_well_e_ptc_turnover=turnover_fw_e,
        full_well_dn_at_linearity_limit=fw_lin_dn,
        full_well_e_at_linearity_limit=fw_lin_e,
        dynamic_range=float(dynamic_range),
        dynamic_range_db=float(dynamic_range_db),
        fpn_percent_median=fpn_percent_median,
        points=points,
    )

    save_csv(output_dir / "points.csv", points)
    save_summary_json(output_dir / "summary.json", results)
    write_markdown_report(results, output_dir)
    plot_ptc(results, output_dir)
    plot_variance_fit(results, output_dir)
    plot_nonlinearity(results, output_dir)
    plot_signal_linearity(results, output_dir)
    return results


def safe_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name.strip())
    return clean or "camera"


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Photon Transfer Curve Analyzer")
    root.geometry("980x620")

    cameras: List[Dict[str, tk.StringVar]] = []

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)

    intro = ttk.Label(
        main,
        text=(
            "Add one row per camera. Select a bias folder and a flat folder. "
            "Flat exposures are read from FITS headers or from filenames such as flat_100ms_01.fit."
        ),
        wraplength=920,
    )
    intro.pack(anchor="w", pady=(0, 10))

    options = ttk.LabelFrame(main, text="Global options", padding=8)
    options.pack(fill="x", pady=(0, 10))

    output_var = tk.StringVar(value=str(Path.cwd() / "ptc_results"))
    roi_var = tk.StringVar()
    regex_var = tk.StringVar()
    fit_low_var = tk.StringVar(value="0.10")
    fit_high_var = tk.StringVar(value="0.70")
    linearity_var = tk.StringVar(value="1.0")
    saturation_var = tk.StringVar()
    color_mode_var = tk.StringVar(value="mono")

    def browse_output() -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            output_var.set(path)

    row = ttk.Frame(options)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Output folder", width=18).pack(side="left")
    ttk.Entry(row, textvariable=output_var).pack(side="left", fill="x", expand=True, padx=4)
    ttk.Button(row, text="Browse", command=browse_output).pack(side="left")

    row = ttk.Frame(options)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="ROI x,y,w,h", width=18).pack(side="left")
    ttk.Entry(row, textvariable=roi_var, width=18).pack(side="left", padx=4)
    ttk.Label(row, text="Exposure regex").pack(side="left", padx=(12, 4))
    ttk.Entry(row, textvariable=regex_var).pack(side="left", fill="x", expand=True, padx=4)

    row = ttk.Frame(options)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Fit signal range", width=18).pack(side="left")
    ttk.Entry(row, textvariable=fit_low_var, width=8).pack(side="left", padx=4)
    ttk.Label(row, text="to").pack(side="left")
    ttk.Entry(row, textvariable=fit_high_var, width=8).pack(side="left", padx=4)
    ttk.Label(row, text="Linearity threshold %").pack(side="left", padx=(12, 4))
    ttk.Entry(row, textvariable=linearity_var, width=8).pack(side="left", padx=4)
    ttk.Label(row, text="Saturation ADU").pack(side="left", padx=(12, 4))
    ttk.Entry(row, textvariable=saturation_var, width=10).pack(side="left", padx=4)

    row = ttk.Frame(options)
    row.pack(fill="x", pady=2)
    ttk.Label(row, text="Color mode", width=18).pack(side="left")
    color_modes = [
        "mono",
        "red",
        "green",
        "blue",
        "luminance",
        "bayer_rggb_red",
        "bayer_bggr_red",
        "bayer_grbg_red",
        "bayer_gbrg_red",
        "bayer_rggb_green1",
        "bayer_rggb_green2",
        "bayer_rggb_blue",
        "bayer_bggr_green1",
        "bayer_bggr_green2",
        "bayer_bggr_blue",
        "bayer_grbg_green1",
        "bayer_grbg_green2",
        "bayer_grbg_blue",
        "bayer_gbrg_green1",
        "bayer_gbrg_green2",
        "bayer_gbrg_blue",
    ]
    ttk.Combobox(row, textvariable=color_mode_var, values=color_modes, state="readonly", width=24).pack(side="left", padx=4)
    ttk.Label(
        row,
        text="Use red for RGB files; use matching bayer_*_red for raw Bayer color cameras.",
    ).pack(side="left", padx=8)

    camera_frame = ttk.LabelFrame(main, text="Cameras", padding=8)
    camera_frame.pack(fill="both", expand=True)

    header = ttk.Frame(camera_frame)
    header.pack(fill="x")
    for text, width in [("Camera", 18), ("Bias folder", 38), ("Flat folder", 38), ("", 8)]:
        ttk.Label(header, text=text, width=width).pack(side="left", padx=3)

    rows_frame = ttk.Frame(camera_frame)
    rows_frame.pack(fill="both", expand=True)

    log_text = tk.Text(main, height=8)
    log_text.pack(fill="both", expand=False, pady=(10, 0))

    def log(msg: str) -> None:
        log_text.insert("end", msg + "\n")
        log_text.see("end")
        root.update_idletasks()

    def add_camera(default_name: Optional[str] = None) -> None:
        vars_ = {
            "name": tk.StringVar(value=default_name or f"Camera_{len(cameras)+1}"),
            "bias": tk.StringVar(),
            "flats": tk.StringVar(),
        }
        cameras.append(vars_)
        fr = ttk.Frame(rows_frame)
        fr.pack(fill="x", pady=3)
        ttk.Entry(fr, textvariable=vars_["name"], width=18).pack(side="left", padx=3)
        ttk.Entry(fr, textvariable=vars_["bias"], width=38).pack(side="left", padx=3)

        def browse_bias(v=vars_["bias"]) -> None:
            path = filedialog.askdirectory(title="Select bias folder")
            if path:
                v.set(path)

        ttk.Button(fr, text="Bias", command=browse_bias).pack(side="left", padx=3)
        ttk.Entry(fr, textvariable=vars_["flats"], width=38).pack(side="left", padx=3)

        def browse_flats(v=vars_["flats"]) -> None:
            path = filedialog.askdirectory(title="Select flat folder")
            if path:
                v.set(path)

        ttk.Button(fr, text="Flats", command=browse_flats).pack(side="left", padx=3)

    def run_analysis() -> None:
        try:
            roi = parse_roi(roi_var.get())
            output = output_var.get().strip()
            if not output:
                raise ValueError("Output folder is required")
            configs = []
            for vars_ in cameras:
                name = vars_["name"].get().strip()
                bias = vars_["bias"].get().strip()
                flats = vars_["flats"].get().strip()
                if not name and not bias and not flats:
                    continue
                if not name or not bias or not flats:
                    raise ValueError("Each camera row needs camera name, bias folder and flat folder")
                configs.append(
                    AnalysisConfig(
                        camera_name=name,
                        bias_dir=bias,
                        flat_dir=flats,
                        output_dir=output,
                        roi=roi,
                        color_mode=color_mode_var.get(),
                        exposure_regex=regex_var.get().strip() or None,
                        fit_low_fraction=float(fit_low_var.get()),
                        fit_high_fraction=float(fit_high_var.get()),
                        linearity_threshold_percent=float(linearity_var.get()),
                        saturation_adu=float(saturation_var.get()) if saturation_var.get().strip() else None,
                    )
                )
            if not configs:
                raise ValueError("Add at least one camera")
            for cfg in configs:
                log(f"Analyzing {cfg.camera_name}...")
                res = analyze_camera(cfg)
                log(
                    f"Done {cfg.camera_name}: K={res.k_adc_e_per_dn:.4g} e-/ADU, "
                    f"RN={res.read_noise_e:.4g} e-, FW~{res.full_well_e_observed:.4g} e-"
                )
            messagebox.showinfo("PTC analysis complete", f"Results saved in:\n{output}")
        except Exception as exc:
            log("ERROR: " + str(exc))
            log(traceback.format_exc())
            messagebox.showerror("PTC analysis failed", str(exc))

    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(10, 0))
    ttk.Button(buttons, text="Add camera", command=add_camera).pack(side="left")
    ttk.Button(buttons, text="Run analysis", command=run_analysis).pack(side="right")

    add_camera("Camera_1")
    root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Photon Transfer Curve Analyzer")
    parser.add_argument("--no-gui", action="store_true", help="Run in command-line mode")
    parser.add_argument("--camera", default="Camera_1", help="Camera name")
    parser.add_argument("--bias", help="Bias folder")
    parser.add_argument("--flats", help="Flat folder")
    parser.add_argument("--output", default="ptc_results", help="Output folder")
    parser.add_argument("--roi", default=None, help="ROI as x,y,width,height")
    parser.add_argument(
        "--color-mode",
        default="mono",
        help=(
            "Color extraction mode: mono, red, green, blue, luminance, "
            "bayer_rggb_red, bayer_bggr_red, bayer_grbg_red, bayer_gbrg_red, etc."
        ),
    )
    parser.add_argument("--exposure-regex", default=None, help="Custom regex; first group must be exposure, optional second group unit")
    parser.add_argument("--fit-low", type=float, default=0.10, help="Low fraction of max signal for variance fit")
    parser.add_argument("--fit-high", type=float, default=0.70, help="High fraction of max signal for variance fit")
    parser.add_argument("--linearity-threshold", type=float, default=1.0, help="Linearity threshold in percent")
    parser.add_argument("--saturation-adu", type=float, default=None, help="Known saturation ADU, optional")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.no_gui:
        run_gui()
        return 0
    if not args.bias or not args.flats:
        parser.error("--bias and --flats are required with --no-gui")
    config = AnalysisConfig(
        camera_name=args.camera,
        bias_dir=args.bias,
        flat_dir=args.flats,
        output_dir=args.output,
        roi=parse_roi(args.roi),
        color_mode=args.color_mode,
        exposure_regex=args.exposure_regex,
        fit_low_fraction=args.fit_low,
        fit_high_fraction=args.fit_high,
        linearity_threshold_percent=args.linearity_threshold,
        saturation_adu=args.saturation_adu,
    )
    results = analyze_camera(config)
    print(f"Results saved to: {Path(args.output) / safe_name(args.camera)}")
    print(f"K_ADC: {results.k_adc_e_per_dn:.6g} e-/ADU")
    print(f"Conversion gain: {results.conversion_gain_dn_per_e:.6g} ADU/e-")
    print(f"Read noise: {results.read_noise_dn:.6g} ADU = {results.read_noise_e:.6g} e-")
    print(f"Full well observed proxy: {results.full_well_dn_observed:.6g} ADU = {results.full_well_e_observed:.6g} e-")
    print(f"Max |K nonlinearity|: {results.max_abs_k_nonlinearity_percent:.6g} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
