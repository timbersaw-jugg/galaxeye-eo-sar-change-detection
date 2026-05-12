# GalaxEye — Lightweight EO--SAR Change Detection Framework

**Position:** Satellite AI Research Intern  
**Task:** Pixel-level EO--SAR change detection for remote sensing imagery  

---

## Overview

This repository implements a lightweight multimodal EO--SAR change detection framework for binary pixel-level remote sensing segmentation. The objective is to identify changed regions between paired Electro-Optical (EO) imagery and Synthetic Aperture Radar (SAR) observations under noisy and heterogeneous remote sensing conditions.

The framework combines EO RGB imagery with SAR-derived structural representations and performs end-to-end semantic segmentation using an EfficientNet-B0 encoder together with a UNet-style decoder architecture.

Key challenges addressed include:
- Severe foreground-background imbalance
- SAR speckle noise
- Cross-scene EO--SAR variability
- Sparse change-region localization

**Approach:** Lightweight multimodal EO--SAR change detection using an EfficientNet-B0 UNet architecture with hybrid BCE + Dice + Focal optimization, threshold calibration, and GradCAM-based explainability analysis.

---

## Repository Structure

```text
galaxeye/
├── config.yaml
├── train_efficientnet_b0.py
├── eval.py
├── infer.py
├── requirements.txt
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   └── change_detector.py
│
├── losses/
│   └── combined.py
│
├── utils/
│   ├── metrics.py
│   ├── gradcam.py
│   └── visualize.py
│
├── checkpoints/
├── logs/
├── results/
└── rough/
```

---

## Requirements

- Python 3.11+
- PyTorch 2.3.1
- See `requirements.txt` for full dependency list

---

## Environment Setup

### Create Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / MacOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Arrange the dataset as follows:

```text
data/
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

File names must match across:
- pre-event
- post-event
- target

Update `config.yaml`:

```yaml
data:
  root: "/your/data/root"
```

---

## Label Reformulation

The original annotations contain four semantic categories. For binary EO--SAR change detection, all change-related categories are merged into a single foreground class.

| Original Class | Binary Label |
|---|---|
| Background / Intact | 0 |
| Damaged / Destroyed | 1 |

This reformulation converts the task into binary changed vs unchanged segmentation.

---

## Training

```bash
python train_efficientnet_b0.py
```

Outputs:
- Best model checkpoint
- Training history logs
- Loss / F1 / IoU curves
- Validation metrics

---

## Evaluation

### Test Evaluation

```bash
python eval.py \
    --config config.yaml \
    --weights checkpoints/efficientnet_b0/best_model.pth \
    --split test \
    --threshold 0.30 \
    --gradcam
```

### Validation Evaluation

```bash
python eval.py \
    --config config.yaml \
    --weights checkpoints/efficientnet_b0/best_model.pth \
    --split val \
    --threshold 0.30
```

Outputs generated inside `results/`:
- `confusion_matrix_test.png`
- `qualitative_test.png`
- `gradcam_test.png`
- `metrics_test.json`

---

## Inference on New Image Pairs

```bash
python infer.py \
    --config config.yaml \
    --weights checkpoints/efficientnet_b0/best_model.pth \
    --pre /path/to/pre_event.tif \
    --post /path/to/post_event.tif \
    --threshold 0.30 \
    --output_dir results/inference
```

Generated outputs:
- Binary change masks
- Probability maps
- EO overlay visualizations

---

## Final Results

| Split | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.2712 | 0.4267 | 0.2940 | 0.7778 |
| Test | 0.0887 | 0.1629 | 0.1207 | 0.2503 |

Optimal threshold:
`0.30`

---

## Cross-Scene Generalization

Validation and test scenes contain different EO--SAR distributions, leading to noticeable domain shift during evaluation. While the framework demonstrated stable localization behavior on validation scenes, test performance degraded under previously unseen SAR scattering patterns and structural layouts.

This highlights the difficulty of robust EO--SAR generalization under limited scene diversity and heterogeneous multimodal conditions.

---

## Generated Outputs

Evaluation automatically generates:
- Confusion matrices
- Qualitative segmentation visualizations
- GradCAM explainability maps
- Threshold-sensitive performance metrics
- Training dynamics plots (Loss, F1, IoU)

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Input | 5-channel EO--SAR fusion | Combines EO texture and SAR structural information |
| Backbone | EfficientNet-B0 | Lightweight and computationally efficient |
| Loss | BCE + Dice + Focal | Handles severe class imbalance |
| Threshold | 0.30 | Best precision-recall balance |
| Explainability | GradCAM++ + LayerCAM | Interpretable spatial attention analysis |
| Training | CPU-based local setup | Reproducible under constrained hardware |

---

## Model Weights

HuggingFace model link:

```text
https://huggingface.co/juggtimber/galaxeye-eo-sar-change-detection
```

Place downloaded weights at:

```text
hf_model/efficientnet_b0/best_model.pth
```

---

## References

- Tan and Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
- Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Lin et al. *Focal Loss for Dense Object Detection*. ICCV 2017.
- Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
- Chattopadhay et al. *Grad-CAM++: Improved Visual Explanations*. WACV 2018.

---

## Notes

This repository was developed as part of a Satellite AI Research Internship assessment focused on multimodal EO--SAR remote sensing analysis, explainable segmentation, and cross-scene change localization under noisy SAR conditions.
