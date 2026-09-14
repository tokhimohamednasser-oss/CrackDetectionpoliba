# Metal Surface Crack Detection — Computer Vision Pipeline

> Automated defect detection in manufactured metal components using adaptive computer vision.

Developed during my internship and MSc thesis at **Fraunhofer IPK Berlin** (Mar–Oct 2025), as part of a broader Industry 4.0 monitoring framework for metal manufacturing. The research contributed to a peer-reviewed paper published at **IEEE MetroInd 2026**.

---

## What this does

Detects and classifies surface cracks in metal components from greyscale or RGB images.
No deep learning required for basic detection — the pipeline uses classical computer vision
techniques that run efficiently on edge hardware (Raspberry Pi tested at Fraunhofer).

```
Input image → Preprocessing → Binarisation → Morphological cleaning → Skeletonisation → Classification
```

| Class  | Major axis length |
|--------|-------------------|
| Small  | < 10 px           |
| Medium | 10 – 20 px        |
| Large  | > 20 px           |

---

## Pipeline steps

| Step | Method | Purpose |
|------|--------|---------|
| Grayscale conversion | `skimage.color.rgb2gray` | Normalise input |
| CLAHE | `skimage.exposure.equalize_adapthist` | Handle uneven lighting |
| Adaptive binarisation | `skimage.filters.threshold_local` | Isolate dark cracks on bright metal |
| Noise removal | `skimage.morphology.remove_small_objects` | Remove < 3 px artefacts |
| Gap closing | `skimage.morphology.closing (disk r=1)` | Connect broken crack segments |
| Hole filling | `scipy.ndimage.binary_fill_holes` | Clean enclosed regions |
| Skeletonisation | `skimage.morphology.skeletonize` | Thin to 1-px width |
| Region measurement | `skimage.measure.regionprops` | Extract length, area, bounding box |

*Original implementation in MATLAB (adapthisteq / imbinarize / bwmorph). This repo is the Python port.*

---

## Installation

```bash
git clone https://github.com//crack-detection.git
cd crack-detection
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-image>=0.26
scipy>=1.10
matplotlib>=3.7
numpy>=1.24
```

---

## Usage

```bash
# Run on a folder of images
python crack_detection.py --input ./images

# Adjust binarisation sensitivity (default 0.65)
python crack_detection.py --input ./images --sensitivity 0.60

# Save annotated output images
python crack_detection.py --input ./images --save

# Batch mode, no display (e.g. on a server)
python crack_detection.py --input ./images --no-display --save
```

### Python API

```python
from crack_detection import detect_cracks, run_batch

# Single image
result = detect_cracks("surface.jpg", sensitivity=0.65)
print(f"Total: {result['total']} — S:{result['small']} M:{result['medium']} L:{result['large']}")

# All images in a folder
run_batch("./images", save=True)
```

---

## Test with public datasets

No proprietary data is included in this repo (original data belongs to Fraunhofer IPK).
The following public datasets work directly with this pipeline:

| Dataset | Source | Notes |
|---------|--------|-------|
| Surface Crack Detection | [Kaggle](https://www.kaggle.com/arunrk7/surface-crack-detection) | 40k images, concrete + metal |
| SDNET2018 | [Utah State University](https://digitalcommons.usu.edu/all_datasets/48) | Structural surface cracks |
| Mendeley Crack Dataset | [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/2) | Steel surface defects |

---

## Output example

```
Image : sample_001.jpg
  Total: 4  |  Small: 1  Medium: 2  Large: 1

Image : sample_002.jpg
  Total: 2  |  Small: 0  Medium: 1  Large: 1

────────────────────────────────────────────
  SUMMARY — 2 images processed
  Total cracks : 6
  Small        : 1
  Medium       : 3
  Large        : 2
────────────────────────────────────────────
```

---

## Context and publication

This work was part of an integrated monitoring framework for metal additive manufacturing and SLS processes at Fraunhofer IPK Berlin. The wider system architecture combined:

- IIoT sensor integration (vibration, temperature, environmental)
- MQTT communication to ThingsBoard Cloud
- Edge computing on Raspberry Pi 4B
- MATLAB and Python image processing pipelines

**Publication:** M. Nasser et al., *"Integration of Measurements for Digitation: Challenges and Opportunities for the Industry of Tomorrow"*, IEEE International Workshop on Metrology for Industry 4.0 & IoT (MetroInd4.0&IoT), 2026.

---

## Author

**Mohamed Nasser** — Digital Manufacturing Engineer  
Fraunhofer IPK Berlin · Politecnico di Bari  
[LinkedIn](https://www.linkedin.com/in/mohamed-nasser-427489b6) · [Website](https://tokhimohamednasser.github.io)
