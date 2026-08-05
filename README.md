# GeoLinker: SE(3)-Equivariant Diffusion for Molecular Linker Design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21806622.svg)](https://doi.org/10.5281/zenodo.21806622)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1-orange.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/rdkit-2025.09.6-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **GeoLinker**, an SE(3)-equivariant diffusion model for fragment-based molecular linker design, supporting four design variants from unconstrained to fully anchor- and size-constrained generation.

> **Given disconnected molecular fragments in 3D space, GeoLinker generates linker atoms that bridge them into a fully connected molecule. The framework unifies multiple conditioning modes including unconstrained, anchor-guided, size-conditioned, and fully conditioned generation within a single model.**

![Model Architecture](https://raw.githubusercontent.com/bucketio/img8/main/2026/04/25/1777135775057-af40a8f0-6b76-4a68-bf2d-674c43501f89.png)

---

## Overview

GeoLinker generates 3D linker structures that bridge two molecular fragments. Four design variants provide increasing levels of constraint:

| Variant | Description |
|----------|-------------|
| **Base** | Fragment geometry conditioning only |
| **Anchor** | Explicit attachment atoms as hard constraints |
| **Sized** | Target linker atom count via size embedding |
| **Both** | Anchor constraints + size conditioning |

---

# ⚙️ Installation

```bash
git clone https://github.com/srajalcodes/GeoLinker.git
cd GeoLinker

conda env create -f environment.yml
conda activate geolinker
```

---

# 🚀 Quick Start: Reproduce Paper Results

We designed this repository to be **fully reproducible**. You do **not** need to retrain the model or manually download datasets to reproduce the paper.

## 1. Download datasets and pretrained model

```bash
python scripts/download_data.py
```

This automatically downloads:

- Pretrained GeoLinker checkpoint
- ZINC dataset
- CASF-2016 dataset
- GEOM-Drugs dataset

from our Zenodo archive and places everything into the correct repository folders.

Resulting directory structure:

```text
datasets/
├── zinc/
├── casf/
└── geom/

checkpoints/
└── geolinker_best_unified.pt
```

---

## 2. Reproduce all paper results

```bash
python scripts/reproduce.py
```

This master script automatically:

- Generates linkers for all four GeoLinker variants
- Converts generated 3D molecules into valid SMILES
- Performs MMFF optimization
- Computes all reported evaluation metrics
- Produces the LaTeX tables reported in the manuscript

No additional commands are required.

---

# 🔬 Manual Usage

If you prefer to execute each stage separately, the following scripts can be used.

---

## Training from Scratch

GeoLinker uses a **unified training paradigm with dynamic constraint masking**, allowing a single model to support all four conditioning variants.

```bash
python geolinker/train.py \
    --train_path datasets/zinc/zinc_final_train.pt \
    --val_path datasets/zinc/zinc_final_val.pt \
    --epochs 150 \
    --batch_size 32 \
    --max_atoms_per_batch 650 \
    --lr 2e-4 \
    --save_dir checkpoints/
```

---

## Generation

Generate molecular linkers for any supported variant (`base`, `anchor`, `sized`, or `both`).

Example:

```bash
python scripts/generate.py \
    --variant both \
    --checkpoint checkpoints/geolinker_best_unified.pt \
    --fragments datasets/zinc/zinc_final_test.pt \
    --output_dir outputs/zinc_both \
    --n_samples 250 \
    --num_steps 50 \
    --strategy ddim
```

---

## Export Generated Molecules

Convert generated 3D coordinates into chemically valid SMILES using strict MMFF optimization and smart covalent bond perception.

```bash
python scripts/export_for_difflinker.py \
    --output_dir outputs/zinc_both \
    --test_path datasets/zinc/zinc_final_test.pt
```

---

## Evaluation

Compute standard molecular generation metrics.

```bash
python scripts/evaluate_with_difflinker.py \
    ZINC \
    outputs/zinc_both/generated_smiles.txt \
    datasets/zinc/zinc_final_train_linkers.smi \
    4 \
    True \
    None \
    datasets/zinc/wehi_pains.csv \
    diffusion
```

Reported metrics include:

- Validity
- Uniqueness
- Novelty
- QED
- Synthetic Accessibility (SA)
- Ring Statistics

---

# 📊 Results

### Table 1 — ZINC Test Set

**250 samples per fragment pair (DDIM, 50 diffusion steps)**

| Variant | Validity ↑ | Uniqueness ↑ | Novelty ↑ | QED ↑ | SA ↓ | Rings ↑ |
|----------|-----------:|-------------:|----------:|------:|-----:|--------:|
| GeoLinker-Base | 45.5 | 86.4 | 78.7 | 0.50 | 5.14 | 0.34 |
| GeoLinker-Anchor | 44.1 | 86.4 | 82.1 | 0.49 | 5.04 | 0.57 |
| GeoLinker-Sized | 45.3 | 88.7 | 81.9 | 0.49 | 5.05 | 0.63 |
| GeoLinker-Both | 44.8 | 87.5 | 80.8 | 0.49 | 5.02 | 0.54 |

Complete evaluation on **CASF-2016** and **GEOM-Drugs** is reported in the accompanying manuscript.

---

# 🧠 Model Architecture

**Backbone**

- Unified SE(3)-equivariant dual-stream architecture
- 6-layer Equivariant Graph Neural Network (EGNN)

**Conditioning**

- Anchor-aware cross-attention
- Learnable size embedding \(E \in \mathbb{R}^{21 \times 64}\)

**Diffusion**

- Continuous SDE with 1000 diffusion steps
- Coordinate tanh stabilization
- Zero-center-of-mass projection

**Physics-Informed Losses**

- Endpoint Distance Loss
- Non-Anchor Repulsion Loss
- Steric Clash Loss

**Post-processing Pipeline**

1. Connectivity perception
2. RDKit sanitization
3. Standardization
4. Charge neutralization
5. Fragment retention
6. Radical resolution
7. Steric clash resolution

---

# 📁 Repository Structure

```text
GeoLinker/
│
├── geolinker/          # Core model implementation
│   ├── models.py
│   ├── diffusion.py
│   ├── losses.py
│   └── train.py
│
├── scripts/            # Executable scripts
│   ├── download_data.py
│   ├── reproduce.py
│   ├── generate.py
│   ├── export_for_difflinker.py
│   └── evaluate_with_difflinker.py
│
├── src/                # DiffLinker evaluation utilities
│
├── configs/
│   └── default.yaml
│
├── datasets/           # Automatically populated from Zenodo
│
├── checkpoints/        # Automatically populated from Zenodo
│
├── environment.yml
│
└── README.md
```

