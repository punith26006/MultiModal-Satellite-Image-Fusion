# SRMamba-T Results — PaviaC Dataset

## Summary Metrics

| Scale | Avg PSNR (dB) | Avg SSIM |
|-------|---------------|----------|
| ×2    | 21.05         | 0.4735   |
| ×3    | 20.59         | 0.4403   |

## Per-Patch Results (×3 Scale)

| Patch | PSNR (dB) | SSIM  |
|-------|-----------|-------|
| 1     | 19.4      | 0.393 |
| 2     | 19.1      | 0.398 |
| 3     | 25.1      | 0.601 |
| 4     | 22.2      | 0.431 |

## Per-Patch Results (×2 Scale)

| Patch | PSNR (dB) | SSIM  |
|-------|-----------|-------|
| 1     | 19.6      | 0.439 |
| 2     | 19.3      | 0.437 |
| 3     | 25.3      | 0.601 |
| 4     | 22.7      | 0.463 |

## Visual Comparisons

Each comparison shows: **LR (input)** → **SR (super-resolved)** → **HR (ground truth)**

### ×3 Super-Resolution
![SRMamba-T x3 PaviaC Results](paviac_x3_results.png)

### ×2 Super-Resolution
![SRMamba-T x2 PaviaC Results](paviac_x2_results.png)

> **Note**: Save the result images as `paviac_x3_results.png` and `paviac_x2_results.png` in this directory.
