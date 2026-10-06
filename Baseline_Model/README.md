# [Baseline Model] SRMamba-T: Hyperspectral Image Super-Resolution

> **Note**: This directory contains the **Baseline Model** implementation (`SRMamba-T`) adapted for hyperspectral image super-resolution, serving as the benchmark for our proposed multi-modal architecture, **FusMamba-SR v2**. For the primary multi-modal model, please refer to the root [README.md](../README.md).

**Architecture**: Hybrid Mamba-Transformer network for Single Image Super-Resolution  
**Datasets**: PaviaC (102 bands) / PaviaU (103 bands)  
**Training strategy**: ×3 → ×2 → ×4 (continual learning)  
**Paper Reference**: *"SRMamba-T: Exploring the hybrid Mamba-Transformer network for Single Image Super-Resolution"* (Neurocomputing 2025)

---

## 📁 Repository Structure

```
├── Code/                          # Training & testing scripts
│   ├── srmamba_t_kaggle_notebook.py       # Full training notebook (PaviaU, Kaggle T4x2)
│   ├── srmamba_t_kaggle_paviac.py         # Full training notebook (PaviaC, Kaggle T4x2)
│   ├── srmamba_t_colab_test.py            # Testing script for Colab
│   └── srmamba_t_testing_notebook.py      # Testing notebook
├── Models/                        # Pretrained model weights
│   ├── PaviaC/
│   │   ├── srmamba_t_x2_best.pth
│   │   ├── srmamba_t_x3_best.pth
│   │   └── srmamba_t_x4_best.pth
│   ├── PaviaU/
│   │   ├── srmamba_t_x2_best.pth
│   │   ├── srmamba_t_x3_best.pth
│   │   └── srmamba_t_x4_best.pth
│   └── daeu_paviaC.pth            # DAEU model for PaviaC
├── Results/                       # Super-resolution results & metrics
│   └── PaviaC/
│       ├── README.md              # Detailed per-patch metrics
│       ├── paviac_x2_results.png  # ×2 visual comparisons
│       └── paviac_x3_results.png  # ×3 visual comparisons
├── PPTs/                          # Presentation materials
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ with CUDA support
- NumPy, SciPy

### Training on Kaggle
1. Upload your dataset ZIP (e.g., `PaviaC_Data.zip`) as a Kaggle Dataset
2. Create a new Notebook and attach the dataset
3. Enable GPU: **Settings → Accelerator → GPU T4 x2**
4. Copy cells from `Code/srmamba_t_kaggle_notebook.py` into notebook cells
5. Run cells in order

### Testing
Use `Code/srmamba_t_testing_notebook.py` or `Code/srmamba_t_colab_test.py` with the pretrained weights from `Models/`.

## 📊 Models

| Dataset | Scale | Model File | Size |
|---------|-------|------------|------|
| PaviaC  | ×2    | `Models/PaviaC/srmamba_t_x2_best.pth` | ~10 MB |
| PaviaC  | ×3    | `Models/PaviaC/srmamba_t_x3_best.pth` | ~11 MB |
| PaviaC  | ×4    | `Models/PaviaC/srmamba_t_x4_best.pth` | ~11 MB |
| PaviaU  | ×2    | `Models/PaviaU/srmamba_t_x2_best.pth` | ~10 MB |
| PaviaU  | ×3    | `Models/PaviaU/srmamba_t_x3_best.pth` | ~11 MB |
| PaviaU  | ×4    | `Models/PaviaU/srmamba_t_x4_best.pth` | ~11 MB |

## 📈 Results (PaviaC)

### Quantitative Results

| Scale | Avg PSNR (dB) | Avg SSIM |
|-------|---------------|----------|
| ×2    | **21.05**     | **0.4735** |
| ×3    | **20.59**     | **0.4403** |

### Per-Patch Breakdown

<details>
<summary>×2 Scale — Per-Patch Metrics</summary>

| Patch | PSNR (dB) | SSIM  |
|-------|-----------|-------|
| 1     | 19.6      | 0.439 |
| 2     | 19.3      | 0.437 |
| 3     | 25.3      | 0.601 |
| 4     | 22.7      | 0.463 |

</details>

<details>
<summary>×3 Scale — Per-Patch Metrics</summary>

| Patch | PSNR (dB) | SSIM  |
|-------|-----------|-------|
| 1     | 19.4      | 0.393 |
| 2     | 19.1      | 0.398 |
| 3     | 25.1      | 0.601 |
| 4     | 22.2      | 0.431 |

</details>

### Visual Comparisons (LR → SR → HR)

See detailed results with images in [`Results/PaviaC/`](Results/PaviaC/).

## 📄 License

This project is for academic/research purposes.
