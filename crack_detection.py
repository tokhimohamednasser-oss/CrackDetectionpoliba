"""
Crack Detection in Metal Components — Computer Vision Pipeline
==============================================================
Original work: Fraunhofer IPK Berlin (Internship & MSc Thesis, 2025)
Author       : Mohamed Nasser
Reference    : IEEE MetroInd 2026 — Integration of Measurements for Digitation

This module ports the original MATLAB SLS crack detection pipeline to Python
using scikit-image, OpenCV, and matplotlib. It accepts a folder of grayscale
or RGB images of metal surfaces and classifies detected cracks as Small,
Medium, or Large based on their major axis length.

Methodology
-----------
1. Load images from a directory (supports jpg, png, bmp, tiff)
2. Convert to grayscale and normalise to float [0, 1]
3. Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalisation)
4. Adaptive binarisation with local thresholding (dark foreground)
5. Morphological cleaning: remove noise, close gaps, fill holes
6. Skeletonise — thin structures to 1-pixel width
7. Measure region properties (major axis length, bounding box, area)
8. Classify: Small (<10 px), Medium (10–20 px), Large (>20 px)
9. Display annotated results and print summary statistics

Usage
-----
    python crack_detection.py --input ./images
    python crack_detection.py --input ./images --sensitivity 0.65 --save

Public datasets to test with
-----------------------------
- Kaggle Surface Crack Detection: https://www.kaggle.com/arunrk7/surface-crack-detection
- SDNET2018 Concrete Crack: https://digitalcommons.usu.edu/all_datasets/48
- Mendeley Crack Dataset:     https://data.mendeley.com/datasets/5y9wdsg2zt/2
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, color, exposure, filters, morphology, measure
from scipy import ndimage


# ── Classification thresholds (mirror MATLAB code) ────────────────────────────
SMALL_THRESH  = 10   # major axis length in pixels
MEDIUM_THRESH = 20


def load_images(folder: str) -> list:
    """Return sorted list of image paths from a folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale float and apply CLAHE.
    Mirrors MATLAB: rgb2gray → im2double → adapthisteq(ClipLimit=0.01)
    """
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float)
        if gray.max() > 1.0:
            gray = gray / 255.0

    # CLAHE — clip limit 0.01 matches MATLAB default nbins=256, ClipLimit=0.01
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.01)
    return enhanced


def binarise(enhanced: np.ndarray, sensitivity: float = 0.65) -> np.ndarray:
    """
    Adaptive binarisation for dark foreground (cracks are dark on bright metal).
    Mirrors MATLAB: imbinarize(img, 'adaptive', 'ForegroundPolarity', 'dark',
                                'Sensitivity', 0.65)
    """
    # Local threshold using Sauvola method (handles varying illumination)
    threshold = filters.threshold_local(enhanced, block_size=51,
                                        method='mean', offset=0.0)
    # Dark foreground: crack pixels are BELOW the local threshold
    bw = enhanced < (threshold * sensitivity)
    return bw


def clean_morphology(bw: np.ndarray) -> np.ndarray:
    """
    Morphological cleaning pipeline.
    Mirrors MATLAB: bwareaopen(3) → imclose(disk(1)) → imfill('holes')
    """
    # Remove objects smaller than 3 pixels (noise)
    bw = morphology.remove_small_objects(bw, max_size=3)
    # Close small gaps with disk structuring element radius=1
    selem = morphology.disk(1)
    bw = morphology.closing(bw, selem)
    # Fill holes inside detected structures
    bw = ndimage.binary_fill_holes(bw)
    return bw


def skeletonise(bw: np.ndarray) -> np.ndarray:
    """
    Thin binary structures to 1-pixel width.
    Mirrors MATLAB: bwmorph(bw, 'thin', Inf)
    """
    return morphology.skeletonize(bw)


def classify_crack(major_axis_length: float) -> str:
    """Classify a detected crack by its major axis length."""
    if major_axis_length < SMALL_THRESH:
        return "Small"
    elif major_axis_length < MEDIUM_THRESH:
        return "Medium"
    else:
        return "Large"


def detect_cracks(image_path: str, sensitivity: float = 0.65,
                  show: bool = True, save: bool = False) -> dict:
    """
    Full crack detection pipeline for a single image.

    Parameters
    ----------
    image_path  : path to the input image
    sensitivity : binarisation sensitivity (default 0.65, matching MATLAB)
    show        : display annotated result
    save        : save annotated result as PNG alongside input

    Returns
    -------
    dict with keys: path, total, small, medium, large, regions
    """
    img = io.imread(image_path)
    enhanced = preprocess(img)
    bw       = binarise(enhanced, sensitivity)
    bw       = clean_morphology(bw)
    skeleton = skeletonise(bw)

    # Measure connected regions on skeleton
    labeled  = measure.label(skeleton)
    props    = measure.regionprops(labeled)

    results = {
        "path"   : image_path,
        "total"  : 0,
        "small"  : 0,
        "medium" : 0,
        "large"  : 0,
        "regions": []
    }

    for region in props:
        length = region.axis_major_length
        if length == 0:   # skip single-pixel artefacts
            continue
        label  = classify_crack(length)
        minr, minc, maxr, maxc = region.bbox   # (row_min, col_min, row_max, col_max)
        results["regions"].append({
            "label"  : label,
            "length" : round(length, 2),
            "area"   : region.area,
            "bbox"   : (minc, minr, maxc - minc, maxr - minr)  # (x, y, w, h)
        })
        results["total"] += 1
        results[label.lower()] += 1

    if show or save:
        _visualise(image_path, enhanced, results, save)

    return results


def _visualise(image_path, enhanced, results, save):
    """Draw bounding boxes and labels on the image."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.imshow(enhanced, cmap="gray")
    ax.set_title(
        f"{Path(image_path).name}  —  "
        f"{results['total']} cracks detected  "
        f"(S:{results['small']}  M:{results['medium']}  L:{results['large']})",
        fontsize=11
    )

    colour_map = {"Small": "cyan", "Medium": "yellow", "Large": "red"}

    for region in results["regions"]:
        x, y, w, h = region["bbox"]
        colour = colour_map[region["label"]]
        rect = mpatches.Rectangle((x, y), w, h,
                                   linewidth=1, edgecolor=colour, facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 3, region["label"],
                color=colour, fontsize=7, fontweight="bold")

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in colour_map.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    if save:
        out_path = Path(image_path).with_suffix("") + "_annotated.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.show()
    plt.close()


def run_batch(folder: str, sensitivity: float = 0.65,
              show: bool = True, save: bool = False):
    """Run detection on all images in a folder and print a summary report."""
    paths = load_images(folder)
    if not paths:
        print(f"No images found in: {folder}")
        return

    print(f"\n{'='*60}")
    print(f"  Crack Detection Pipeline — {len(paths)} images")
    print(f"  Sensitivity: {sensitivity}  |  Save: {save}")
    print(f"{'='*60}\n")

    totals = {"total": 0, "small": 0, "medium": 0, "large": 0}

    for path in paths:
        res = detect_cracks(path, sensitivity=sensitivity,
                            show=show, save=save)
        print(f"Image : {Path(path).name}")
        print(f"  Total: {res['total']}  |  "
              f"Small: {res['small']}  "
              f"Medium: {res['medium']}  "
              f"Large: {res['large']}")
        for key in totals:
            totals[key] += res[key]

    print(f"\n{'─'*60}")
    print(f"  SUMMARY — {len(paths)} images processed")
    print(f"  Total cracks : {totals['total']}")
    print(f"  Small        : {totals['small']}")
    print(f"  Medium       : {totals['medium']}")
    print(f"  Large        : {totals['large']}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metal component crack detection — Python port of MATLAB pipeline"
    )
    parser.add_argument("--input",       required=True, help="Folder containing metal surface images")
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Binarisation sensitivity (default 0.65)")
    parser.add_argument("--no-display",  action="store_true", help="Skip interactive display")
    parser.add_argument("--save",        action="store_true", help="Save annotated images")
    args = parser.parse_args()

    run_batch(
        folder      = args.input,
        sensitivity = args.sensitivity,
        show        = not args.no_display,
        save        = args.save
    )
