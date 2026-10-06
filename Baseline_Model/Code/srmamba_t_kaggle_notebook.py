# =============================================================================
# SRMamba-T: Hyperspectral Image Super-Resolution
# =============================================================================
# Paper: "SRMamba-T: Exploring the hybrid Mamba-Transformer network for
#         Single Image Super-Resolution"
#
# Adapted for hyperspectral super-resolution on PaviaC / PaviaU datasets.
# Designed to run on Kaggle with T4x2 GPUs.
#
# HOW TO USE ON KAGGLE:
#   1. Upload your ZIP (PaviaC_Data.zip) as a Kaggle Dataset
#   2. Create a new Notebook, attach the dataset
#   3. Enable GPU: Settings -> Accelerator -> GPU T4 x2
#   4. Copy each CELL section below into a separate Kaggle cell
#   5. Run cells in order (Shift+Enter)
#
# Each cell is marked with:  # %% [markdown]  or  # %%
# =============================================================================


# %% [markdown]
# # SRMamba-T: Hyperspectral Image Super-Resolution
# **Architecture**: Hybrid Mamba-Transformer (from paper)
# **Dataset**: PaviaC (102 bands) / PaviaU (103 bands)
# **Training order**: ×3 → ×2 → ×4 (continual learning)


# %%
# =============================================================================
# CELL 1: IMPORTS & GPU CHECK
# =============================================================================
import os
import gc
import glob
import math
import time
import zipfile
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
else:
    print("WARNING: No GPU found! Training will be very slow.")


# %%
# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================
# *** CHANGE THESE PATHS TO MATCH YOUR KAGGLE DATASET ***
CONFIG = {
    # --- Data Settings ---
    'dataset_name': 'PaviaU',
    'data_dir': '/kaggle/input/datasets/vujjapunithsai/btpdataset/PaviaU_Data',
    'in_channels': 103,      # 102 for PaviaC, 103 for PaviaU
    'max_val': 10000.0,      # Normalization constant (data range ~0-8000)

    # --- Model Architecture ---
    'embed_dim': 48,         # Feature dimension (must be divisible by 2,4,num_heads)
    'num_heads': 6,          # Attention heads
    'blocks_per_layer': 4,   # Blocks per Mamba/Transformer layer (L in paper)
    'ssm_state_dim': 16,     # State space model hidden dimension (N in paper)
    'cab_reduction': 8,      # Channel attention reduction ratio

    # --- Training ---
    'batch_size': 4,         # Reduced from 8 to prevent memory crash
    'epochs_per_scale': 30,  # 30 epochs per scale for better PSNR
    'lr': 2e-4,              # Initial learning rate
    'lambda_freq': 0.1,      # Frequency loss weight
    'grad_clip': 1.0,        # Gradient clipping
    'num_workers': 0,        # Set to 0 to save memory
    'use_amp': True,         # Mixed precision (faster on T4)

    # --- Training Order ---
    'train_scales': [3, 2, 4],  # Scale factors in training order

    # --- Output ---
    'save_dir': '/kaggle/working/checkpoints',
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
print("Configuration loaded!")
print(f"  Dataset: {CONFIG['dataset_name']} ({CONFIG['in_channels']} bands)")
print(f"  Embed dim: {CONFIG['embed_dim']}, Heads: {CONFIG['num_heads']}")
print(f"  Blocks/layer: {CONFIG['blocks_per_layer']}, SSM state: {CONFIG['ssm_state_dim']}")
print(f"  Batch size: {CONFIG['batch_size']}, Epochs/scale: {CONFIG['epochs_per_scale']}")
print(f"  Training scales: {CONFIG['train_scales']}")


# %%
# =============================================================================
# CELL 3: DATA UTILITIES & DATASET
# =============================================================================

def find_data_dir(base_dir):
    """Auto-detect the actual data directory containing HR/, LR_2/, etc."""
    # Direct structure: base_dir/HR/
    if os.path.isdir(os.path.join(base_dir, 'HR')):
        return base_dir
    # One level deeper: base_dir/SomeName/HR/
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, 'HR')):
            return path
    # Check for ZIP files that need extraction
    for f in os.listdir(base_dir):
        if f.endswith('.zip'):
            print(f"Found ZIP: {f}, extracting...")
            with zipfile.ZipFile(os.path.join(base_dir, f), 'r') as z:
                z.extractall(base_dir)
            return find_data_dir(base_dir)  # Retry after extraction
    raise FileNotFoundError(
        f"Could not find HR/ directory in {base_dir}.\n"
        f"Contents: {os.listdir(base_dir)}"
    )


class HyperspectralSRDataset(Dataset):
    """
    Dataset for hyperspectral image super-resolution.
    Loads paired LR/HR .npy patches from disk.
    """

    def __init__(self, data_dir, scale, max_val=10000.0, augment=True):
        self.scale = scale
        self.max_val = max_val
        self.augment = augment

        # Find actual data directory
        actual_dir = find_data_dir(data_dir)

        lr_dir = os.path.join(actual_dir, f'LR_{scale}')
        hr_dir = os.path.join(actual_dir, 'HR')

        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR directory not found: {lr_dir}")
        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")

        # Match LR and HR files by filename
        lr_files = {os.path.basename(f): f
                    for f in sorted(glob.glob(os.path.join(lr_dir, '*.npy')))}
        hr_files = {os.path.basename(f): f
                    for f in sorted(glob.glob(os.path.join(hr_dir, '*.npy')))}

        common = sorted(set(lr_files.keys()) & set(hr_files.keys()))
        self.pairs = [(lr_files[n], hr_files[n]) for n in common]

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching LR/HR pairs found!\n"
                f"  LR dir ({lr_dir}): {len(lr_files)} files\n"
                f"  HR dir ({hr_dir}): {len(hr_files)} files"
            )

        # Print info
        sample_lr = np.load(self.pairs[0][0])
        sample_hr = np.load(self.pairs[0][1])
        print(f"Dataset loaded: scale ×{scale}")
        print(f"  Pairs: {len(self.pairs)}")
        print(f"  LR shape: {sample_lr.shape} | HR shape: {sample_hr.shape}")
        print(f"  dtype: {sample_lr.dtype} | range: [{sample_lr.min()}, {sample_lr.max()}]")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr = np.load(self.pairs[idx][0]).astype(np.float32)
        hr = np.load(self.pairs[idx][1]).astype(np.float32)

        # Normalize to [0, 1]
        lr = lr / self.max_val
        hr = hr / self.max_val

        # (H, W, C) -> (C, H, W) for PyTorch
        lr = lr.transpose(2, 0, 1)
        hr = hr.transpose(2, 0, 1)

        # Data augmentation (applied identically to LR and HR)
        if self.augment:
            if np.random.random() > 0.5:  # Horizontal flip
                lr = lr[:, :, ::-1].copy()
                hr = hr[:, :, ::-1].copy()
            if np.random.random() > 0.5:  # Vertical flip
                lr = lr[:, ::-1, :].copy()
                hr = hr[:, ::-1, :].copy()
            k = np.random.randint(0, 4)    # Random 90° rotation
            if k > 0:
                lr = np.rot90(lr, k, axes=(1, 2)).copy()
                hr = np.rot90(hr, k, axes=(1, 2)).copy()

        return torch.from_numpy(lr), torch.from_numpy(hr)


# Quick test
print("\nTesting dataset loading...")
try:
    _test_ds = HyperspectralSRDataset(
        CONFIG['data_dir'], scale=3, max_val=CONFIG['max_val'], augment=False
    )
    _lr, _hr = _test_ds[0]
    print(f"  LR tensor: {_lr.shape} | HR tensor: {_hr.shape}")
    print(f"  LR range: [{_lr.min():.4f}, {_lr.max():.4f}]")
    del _test_ds, _lr, _hr
    print("  Dataset test PASSED ✓")
except Exception as e:
    print(f"  Dataset test FAILED: {e}")
    print("  Make sure your data_dir path is correct in CONFIG!")


# %%
# =============================================================================
# CELL 3.5: DATA PROFILING & EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Comprehensive analysis of the hyperspectral dataset BEFORE training.
# Covers: file inventory, shape checks, band-level statistics,
#         intensity distributions, spectral profiles, band correlations,
#         spatial structure, LR-HR comparisons, and anomaly detection.
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sp_stats

print("=" * 70)
print("  📊 DATA PROFILING & EXPLORATORY DATA ANALYSIS")
print("=" * 70)


def run_data_profiling(data_dir, max_val, scales=[2, 3, 4]):
    """
    Full EDA / profiling of the hyperspectral SR dataset.
    Analyses every .npy file in HR/ and LR_x/ directories.
    """
    actual_dir = find_data_dir(data_dir)

    # ------------------------------------------------------------------
    # 1) FILE INVENTORY
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  1️⃣  FILE INVENTORY")
    print("-" * 60)

    hr_dir = os.path.join(actual_dir, 'HR')
    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
    inventory_rows = [{'Directory': 'HR', 'File Count': len(hr_files)}]

    lr_dirs = {}
    for s in scales:
        lr_d = os.path.join(actual_dir, f'LR_{s}')
        if os.path.isdir(lr_d):
            files = sorted(glob.glob(os.path.join(lr_d, '*.npy')))
            lr_dirs[s] = files
            inventory_rows.append({'Directory': f'LR_{s}', 'File Count': len(files)})

    inv_df = pd.DataFrame(inventory_rows)
    print(inv_df.to_string(index=False))
    total_files = inv_df['File Count'].sum()
    print(f"\n  Total .npy files: {total_files}")

    # ------------------------------------------------------------------
    # 2) SHAPE & DTYPE CONSISTENCY CHECK
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  2️⃣  SHAPE & DTYPE CONSISTENCY CHECK")
    print("-" * 60)

    shape_rows = []

    # Check HR
    hr_shapes = set()
    hr_dtypes = set()
    for f in hr_files:
        arr = np.load(f)
        hr_shapes.add(arr.shape)
        hr_dtypes.add(str(arr.dtype))
    shape_rows.append({
        'Directory': 'HR',
        'Unique Shapes': ', '.join(str(s) for s in sorted(hr_shapes)),
        'Dtypes': ', '.join(sorted(hr_dtypes)),
        'Consistent': '✓' if len(hr_shapes) == 1 else '✗ MISMATCH',
    })

    for s, files in lr_dirs.items():
        lr_shapes = set()
        lr_dtypes = set()
        for f in files:
            arr = np.load(f)
            lr_shapes.add(arr.shape)
            lr_dtypes.add(str(arr.dtype))
        shape_rows.append({
            'Directory': f'LR_{s}',
            'Unique Shapes': ', '.join(str(s_) for s_ in sorted(lr_shapes)),
            'Dtypes': ', '.join(sorted(lr_dtypes)),
            'Consistent': '✓' if len(lr_shapes) == 1 else '✗ MISMATCH',
        })

    shape_df = pd.DataFrame(shape_rows)
    print(shape_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 3) HR BAND-LEVEL STATISTICS (full per-band profiling)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  3️⃣  HR BAND-LEVEL STATISTICS (sampled from all patches)")
    print("-" * 60)

    # Stack a representative sample of HR patches
    max_sample = min(len(hr_files), 200)
    sample_indices = np.linspace(0, len(hr_files) - 1, max_sample, dtype=int)
    sample_patches = [np.load(hr_files[i]) for i in sample_indices]
    stacked = np.concatenate([p.reshape(-1, p.shape[-1]) for p in sample_patches], axis=0)
    # stacked shape: (total_pixels, num_bands)
    num_bands = stacked.shape[1]

    band_stats_rows = []
    for b in range(num_bands):
        col = stacked[:, b]
        band_stats_rows.append({
            'Band': b,
            'Mean': f'{col.mean():.2f}',
            'Std': f'{col.std():.2f}',
            'Min': f'{col.min():.2f}',
            'Max': f'{col.max():.2f}',
            'Median': f'{np.median(col):.2f}',
            'Skewness': f'{sp_stats.skew(col):.3f}',
            'Kurtosis': f'{sp_stats.kurtosis(col):.3f}',
            'Zeros%': f'{100 * np.mean(col == 0):.1f}',
        })

    band_df = pd.DataFrame(band_stats_rows)
    # Print first 10, last 10 if many bands
    if num_bands > 25:
        print("  (Showing first 10 and last 10 bands)")
        print(band_df.head(10).to_string(index=False))
        print("  ...")
        print(band_df.tail(10).to_string(index=False))
    else:
        print(band_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4) GLOBAL DATA SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  4️⃣  GLOBAL DATA SUMMARY")
    print("-" * 60)

    global_summary = {
        'Total Pixels Sampled': f'{stacked.shape[0]:,}',
        'Number of Bands': num_bands,
        'Global Mean': f'{stacked.mean():.2f}',
        'Global Std': f'{stacked.std():.2f}',
        'Global Min': f'{stacked.min():.2f}',
        'Global Max': f'{stacked.max():.2f}',
        'Normalization Max Val': max_val,
        'Data Range / Max Val': f'{stacked.max() / max_val:.2%}',
        'Zero Pixels (%)': f'{100 * np.mean(stacked == 0):.2f}%',
        'Negative Pixels (%)': f'{100 * np.mean(stacked < 0):.2f}%',
    }
    for k, v in global_summary.items():
        print(f"  {k:<30s}: {v}")

    # ------------------------------------------------------------------
    # 5) VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  5️⃣  GENERATING VISUALIZATIONS...")
    print("-" * 60)

    fig = plt.figure(figsize=(22, 24))
    fig.suptitle(
        f"Dataset Profiling — {CONFIG['dataset_name']} "
        f"({num_bands} bands, {len(hr_files)} HR patches)",
        fontsize=16, fontweight='bold', y=0.98
    )

    # --- (a) Mean & Std per Band (Spectral Signature Profile) ---
    ax1 = fig.add_subplot(3, 3, 1)
    band_means = stacked.mean(axis=0)
    band_stds = stacked.std(axis=0)
    ax1.fill_between(range(num_bands), band_means - band_stds,
                     band_means + band_stds, alpha=0.3, color='steelblue')
    ax1.plot(range(num_bands), band_means, color='steelblue', lw=1.5)
    ax1.set_xlabel('Band Index')
    ax1.set_ylabel('Pixel Value')
    ax1.set_title('(a) Mean ± Std per Band')
    ax1.grid(True, alpha=0.3)

    # --- (b) Global Pixel Intensity Histogram ---
    ax2 = fig.add_subplot(3, 3, 2)
    flat_sample = stacked[:, ::max(1, num_bands // 10)].ravel()
    ax2.hist(flat_sample, bins=150, color='coral', edgecolor='none', alpha=0.8, density=True)
    ax2.axvline(x=stacked.mean(), color='darkred', ls='--', lw=1.5, label=f'Mean={stacked.mean():.0f}')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Density')
    ax2.set_title('(b) Global Pixel Intensity Distribution')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- (c) Per-Band Intensity Box Plot (sampled bands) ---
    ax3 = fig.add_subplot(3, 3, 3)
    band_indices_for_box = np.linspace(0, num_bands - 1, min(20, num_bands), dtype=int)
    box_data = [stacked[::max(1, stacked.shape[0] // 2000), b] for b in band_indices_for_box]
    bp = ax3.boxplot(box_data, labels=[str(b) for b in band_indices_for_box],
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
    ax3.set_xlabel('Band Index')
    ax3.set_ylabel('Pixel Value')
    ax3.set_title('(c) Band Intensity Box Plots')
    ax3.tick_params(axis='x', rotation=45, labelsize=7)
    ax3.grid(True, alpha=0.3)

    # --- (d) Spectral Signatures of Random Pixels ---
    ax4 = fig.add_subplot(3, 3, 4)
    n_signatures = 50
    sig_indices = np.random.choice(stacked.shape[0], n_signatures, replace=False)
    for idx in sig_indices:
        ax4.plot(range(num_bands), stacked[idx], alpha=0.15, color='teal', lw=0.8)
    ax4.plot(range(num_bands), band_means, color='red', lw=2.0, label='Mean Spectrum')
    ax4.set_xlabel('Band Index')
    ax4.set_ylabel('Pixel Value')
    ax4.set_title(f'(d) Spectral Signatures ({n_signatures} Random Pixels)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # --- (e) Band Correlation Heatmap ---
    ax5 = fig.add_subplot(3, 3, 5)
    # Subsample for faster correlation computation
    sub_pixels = stacked[::max(1, stacked.shape[0] // 5000)]
    corr_matrix = np.corrcoef(sub_pixels.T)
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xlabel('Band Index')
    ax5.set_ylabel('Band Index')
    ax5.set_title('(e) Inter-Band Correlation Matrix')
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # --- (f) Skewness & Kurtosis per Band ---
    ax6 = fig.add_subplot(3, 3, 6)
    skews = [sp_stats.skew(stacked[:, b]) for b in range(num_bands)]
    kurts = [sp_stats.kurtosis(stacked[:, b]) for b in range(num_bands)]
    ax6.plot(range(num_bands), skews, color='darkorange', lw=1.5, label='Skewness')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(range(num_bands), kurts, color='purple', lw=1.5, label='Kurtosis')
    ax6.set_xlabel('Band Index')
    ax6.set_ylabel('Skewness', color='darkorange')
    ax6_twin.set_ylabel('Kurtosis', color='purple')
    ax6.set_title('(f) Skewness & Kurtosis per Band')
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)

    # --- (g) Sample HR Patch Visualization (pseudo-RGB) ---
    ax7 = fig.add_subplot(3, 3, 7)
    rgb_bands = [min(60, num_bands - 1), min(30, num_bands - 1), min(10, num_bands - 1)]
    sample_hr = np.load(hr_files[0])
    rgb_img = sample_hr[:, :, rgb_bands].astype(np.float32)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    ax7.imshow(rgb_img)
    ax7.set_title(f'(g) Sample HR Patch (bands {rgb_bands})')
    ax7.axis('off')

    # --- (h) LR vs HR Comparison for each available scale ---
    ax8 = fig.add_subplot(3, 3, 8)
    lr_scale_info = []
    for s, files in lr_dirs.items():
        if len(files) > 0:
            lr_sample = np.load(files[0])
            lr_scale_info.append((s, lr_sample.shape))
    if lr_scale_info:
        text_lines = [f"HR shape: {sample_hr.shape}"]
        for s, shp in lr_scale_info:
            text_lines.append(f"LR x{s} shape: {shp}")
            text_lines.append(f"  Downscale factor: {sample_hr.shape[0] / shp[0]:.1f}x")
        ax8.text(0.1, 0.5, '\n'.join(text_lines), transform=ax8.transAxes,
                 fontsize=12, verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax8.set_title('(h) LR / HR Spatial Dimensions')
    ax8.axis('off')

    # --- (i) Band Energy Distribution ---
    ax9 = fig.add_subplot(3, 3, 9)
    band_energy = (stacked ** 2).mean(axis=0)
    band_energy_pct = 100 * band_energy / band_energy.sum()
    ax9.bar(range(num_bands), band_energy_pct, color='mediumseagreen', alpha=0.8)
    ax9.set_xlabel('Band Index')
    ax9.set_ylabel('Energy (%)')
    ax9.set_title('(i) Band Energy Distribution')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    profile_path = os.path.join(CONFIG['save_dir'], 'data_profiling.png')
    plt.savefig(profile_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Profile plot saved: {profile_path}")

    # ------------------------------------------------------------------
    # 6) ANOMALY / OUTLIER DETECTION
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  6️⃣  ANOMALY & OUTLIER DETECTION")
    print("-" * 60)

    anomaly_rows = []
    for i, f in enumerate(hr_files):
        arr = np.load(f)
        fname = os.path.basename(f)
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        has_neg = np.any(arr < 0)
        pct_zero = 100 * np.mean(arr == 0)
        val_max = arr.max()
        if has_nan or has_inf or has_neg or pct_zero > 50 or val_max > max_val * 1.5:
            anomaly_rows.append({
                'File': fname,
                'NaN': 'yes' if has_nan else '',
                'Inf': 'yes' if has_inf else '',
                'Negative': 'yes' if has_neg else '',
                'Zero%': f'{pct_zero:.1f}',
                'Max': f'{val_max:.1f}',
            })

    if anomaly_rows:
        anom_df = pd.DataFrame(anomaly_rows)
        print(f"  Found {len(anomaly_rows)} file(s) with potential issues:")
        print(anom_df.to_string(index=False))
    else:
        print("  No anomalies detected (no NaN, Inf, negatives, or extreme values)")

    # ------------------------------------------------------------------
    # 7) SUMMARY TABLE (pandas-style profiling)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  7️⃣  PANDAS-STYLE PROFILING SUMMARY")
    print("-" * 60)

    summary_df = pd.DataFrame({
        'Metric': [
            'Dataset Name', 'Number of Bands (Channels)',
            'HR Patch Count', 'HR Patch Shape',
            'Pixel Value Range', 'Global Mean', 'Global Std',
            'Normalization Constant', 'Data Utilization (%)',
            'Available LR Scales', 'Total File Count',
            'Data Quality', 'Estimated Memory (all HR, float32)',
        ],
        'Value': [
            CONFIG['dataset_name'], num_bands,
            len(hr_files), str(list(hr_shapes)[0]) if len(hr_shapes) == 1 else str(hr_shapes),
            f'[{stacked.min():.1f}, {stacked.max():.1f}]',
            f'{stacked.mean():.2f}', f'{stacked.std():.2f}',
            max_val, f'{100 * stacked.max() / max_val:.1f}%',
            ', '.join([f'x{s}' for s in lr_dirs.keys()]),
            total_files,
            'Clean' if not anomaly_rows else f'{len(anomaly_rows)} issue(s)',
            f'{len(hr_files) * np.prod(list(hr_shapes)[0]) * 4 / 1e9:.2f} GB' if len(hr_shapes) == 1 else 'Variable',
        ]
    })
    print(summary_df.to_string(index=False))

    # Cleanup
    del stacked, sample_patches, sub_pixels
    print("\n" + "=" * 70)
    print("  DATA PROFILING COMPLETE — Proceeding to model definition...")
    print("=" * 70)


# --- Run the profiling ---
try:
    run_data_profiling(CONFIG['data_dir'], CONFIG['max_val'])
except Exception as e:
    print(f"  Profiling could not complete: {e}")
    print("  This is non-blocking — training can still proceed.")
    print("  Make sure your data_dir path is correct in CONFIG!")

# Free memory after profiling
plt.close('all')
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Memory cleaned after profiling ✓")


# %%
# =============================================================================
# CELL 4: MODEL — Selective Scan & MDSSM (Mamba Core)
# =============================================================================

def selective_scan(x, delta, A, B, C, D_skip):
    """
    Pure PyTorch implementation of Mamba's S6 selective scan.
    No custom CUDA kernels needed — runs on any GPU.

    Args:
        x:      (B, T, D)  input sequence
        delta:  (B, T, D)  discretization timestep (positive, after softplus)
        A:      (D, N)     state transition matrix (negative values)
        B:      (B, T, N)  input-dependent projection
        C:      (B, T, N)  output-dependent projection
        D_skip: (D,)       skip connection parameter

    Returns:
        y: (B, T, D) output sequence
    """
    B_batch, T, D = x.shape
    N = A.shape[1]

    # Discretize continuous parameters
    # A_bar = exp(delta * A)  — state decay
    # B_bar * x = delta * B * x  — input contribution
    deltaA = torch.exp(
        delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )  # (B, T, D, N)
    deltaBx = (
        delta.unsqueeze(-1) *
        B.unsqueeze(2) *
        x.unsqueeze(-1)
    )  # (B, T, D, N)

    # Sequential scan (recurrence)
    ys = torch.zeros(B_batch, T, D, device=x.device, dtype=x.dtype)
    h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)

    for t in range(T):
        h = deltaA[:, t] * h + deltaBx[:, t]           # (B, D, N)
        ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)   # (B, D)

    # Skip connection (like residual)
    ys = ys + x * D_skip

    return ys


class SSMBlock(nn.Module):
    """Single-direction State Space Model block."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A: state transition (learnable, log-parameterized for stability)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0).repeat(d_model, 1)
            )
        )

        # Input-dependent projections (computed per token)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

        # Skip connection parameter
        self.D = nn.Parameter(torch.ones(d_model))

        # Initialize dt bias for reasonable timescales
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.1)

    def forward(self, x):
        """x: (B, T, d_model) → (B, T, d_model)"""
        A = -torch.exp(self.A_log)          # Negative for stability
        B = self.B_proj(x)                  # (B, T, d_state)
        C = self.C_proj(x)                  # (B, T, d_state)
        delta = F.softplus(self.dt_proj(x)) # (B, T, d_model), positive

        return selective_scan(x, delta, A, B, C, self.D)


class MDSSM(nn.Module):
    """
    Multi-Directional Selective Scan Module (from paper Fig. 4c).

    Splits channels into 4 groups, scans each in a different 2D direction:
      Dir 0: row-major (top-left → bottom-right)
      Dir 1: column-major (top-left → bottom-right, transposed)
      Dir 2: reverse row-major (bottom-right → top-left)
      Dir 3: reverse column-major (bottom-right → top-left, transposed)

    This captures multi-directional spatial dependencies with fewer
    parameters than replicating the full feature map (like 2D-SSM).
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        assert d_model % 4 == 0, f"d_model={d_model} must be divisible by 4"
        self.d_model = d_model
        self.d_group = d_model // 4

        # One independent SSM per scan direction
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(self.d_group, d_state) for _ in range(4)
        ])

    @staticmethod
    def _scan_2d(x, direction):
        """Flatten 2D spatial to 1D in specified direction.
        x: (B, C, H, W) → (B, H*W, C)
        """
        B, C, H, W = x.shape
        if direction == 0:    # row-major
            return x.reshape(B, C, -1).permute(0, 2, 1)
        elif direction == 1:  # column-major
            return x.permute(0, 1, 3, 2).reshape(B, C, -1).permute(0, 2, 1)
        elif direction == 2:  # reverse row-major
            return x.reshape(B, C, -1).flip(-1).permute(0, 2, 1)
        else:                 # reverse column-major
            return x.permute(0, 1, 3, 2).reshape(B, C, -1).flip(-1).permute(0, 2, 1)

    @staticmethod
    def _unscan_2d(x, H, W, direction):
        """Restore 2D spatial from 1D scan order.
        x: (B, H*W, C) → (B, C, H, W)
        """
        B, T, C = x.shape
        if direction == 0:
            return x.permute(0, 2, 1).reshape(B, C, H, W)
        elif direction == 1:
            return x.permute(0, 2, 1).reshape(B, C, W, H).permute(0, 1, 3, 2)
        elif direction == 2:
            return x.permute(0, 2, 1).flip(-1).reshape(B, C, H, W)
        else:
            return x.permute(0, 2, 1).flip(-1).reshape(B, C, W, H).permute(0, 1, 3, 2)

    def forward(self, x, H, W):
        """
        x: (B, T, d_model) token sequence, T = H*W
        Returns: (B, T, d_model)
        """
        B, T, D = x.shape

        # Split channels into 4 groups
        groups = x.chunk(4, dim=-1)   # each: (B, T, d_group)

        outputs = []
        for i, (group, ssm) in enumerate(zip(groups, self.ssm_blocks)):
            # Reshape tokens → 2D spatial
            g_2d = group.permute(0, 2, 1).reshape(B, self.d_group, H, W)

            # Flatten in scan direction i
            g_1d = self._scan_2d(g_2d, direction=i)

            # Apply SSM along the 1D sequence
            g_out = ssm(g_1d)

            # Unscan: restore original 2D spatial order
            g_2d_out = self._unscan_2d(g_out, H, W, direction=i)

            # Back to token format
            outputs.append(g_2d_out.reshape(B, self.d_group, -1).permute(0, 2, 1))

        return torch.cat(outputs, dim=-1)  # (B, T, d_model)


print("Selective Scan & MDSSM defined ✓")


# %%
# =============================================================================
# CELL 5: MODEL — MambaMixer, Self-Attention, CAB
# =============================================================================

class MambaMixer(nn.Module):
    """
    Asymmetric Mamba Mixer block (paper Eq. 8, Fig. 3a).

    Two parallel branches:
      Branch 1: Linear → Conv → SiLU  (preserves full context)
      Branch 2: Linear → Conv → SiLU → MDSSM  (multi-directional scan)
    Then concatenate and merge with a linear layer.
    """

    def __init__(self, embed_dim, ssm_state_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2

        # Branch 1 (context preservation — no MDSSM)
        self.linear1 = nn.Linear(embed_dim, self.half_dim)
        self.conv1 = nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1)

        # Branch 2 (multi-directional scanning)
        self.linear2 = nn.Linear(embed_dim, self.half_dim)
        self.conv2 = nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1)
        self.mdssm = MDSSM(self.half_dim, ssm_state_dim)

        # Merge branches
        self.merge = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, H, W):
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        # --- Branch 1 ---
        x1 = self.linear1(x)                                          # (B,T,D/2)
        x1 = x1.permute(0, 2, 1).reshape(B, self.half_dim, H, W)     # (B,D/2,H,W)
        x1 = F.silu(self.conv1(x1))
        x1 = x1.reshape(B, self.half_dim, T).permute(0, 2, 1)        # (B,T,D/2)

        # --- Branch 2 ---
        x2 = self.linear2(x)
        x2 = x2.permute(0, 2, 1).reshape(B, self.half_dim, H, W)
        x2 = F.silu(self.conv2(x2))
        x2 = x2.reshape(B, self.half_dim, T).permute(0, 2, 1)
        x2 = self.mdssm(x2, H, W)                                     # (B,T,D/2)

        # --- Merge ---
        out = self.merge(torch.cat([x1, x2], dim=-1))                 # (B,T,D)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    Full multi-head self-attention without windowing (paper Eq. 9).
    Uses PyTorch 2.0 efficient attention (FlashAttention when available).
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, H=None, W=None):
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, H, T, d_head)

        # Efficient attention (auto-selects FlashAttention if available)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            # Fallback for older PyTorch
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q * scale, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)

        x = attn_out.transpose(1, 2).reshape(B, T, D)
        x = self.proj(x)
        return x


class CAB(nn.Module):
    """
    Channel Attention Block (squeeze-and-excitation style).
    Learns per-channel importance weights.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)"""
        scale = x.mean(dim=[2, 3])                              # (B, C)
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(scale)))) # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


print("MambaMixer, Self-Attention, CAB defined ✓")


# %%
# =============================================================================
# CELL 6: MODEL — Blocks, Layers, FFM
# =============================================================================

class MambaBlock(nn.Module):
    """
    Single Mamba Block (paper Eq. 6-7):
      X → LayerNorm → MambaMixer → + S1·X → X_hat
      X_hat → LayerNorm → Conv → CAB → + S2·X_hat → X_out
    """

    def __init__(self, embed_dim, ssm_state_dim=16, cab_reduction=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mamba_mixer = MambaMixer(embed_dim, ssm_state_dim)
        self.s1 = nn.Parameter(torch.ones(embed_dim) * 0.1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.cab = CAB(embed_dim, cab_reduction)
        self.s2 = nn.Parameter(torch.ones(embed_dim) * 0.1)

    def forward(self, x, H, W):
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        # Part 1: MambaMixer + skip
        x_hat = self.mamba_mixer(self.norm1(x), H, W) + self.s1 * x

        # Part 2: Conv + CAB + skip
        z = self.norm2(x_hat)
        z = z.permute(0, 2, 1).reshape(B, D, H, W)   # → 2D
        z = self.cab(self.conv(z))
        z = z.reshape(B, D, T).permute(0, 2, 1)       # → 1D tokens

        return z + self.s2 * x_hat


class AttentionBlock(nn.Module):
    """
    Single Attention Block (same structure as MambaBlock, with MHSA):
      X → LayerNorm → MHSA → + S1·X → X_hat
      X_hat → LayerNorm → Conv → CAB → + S2·X_hat → X_out
    """

    def __init__(self, embed_dim, num_heads, cab_reduction=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.s1 = nn.Parameter(torch.ones(embed_dim) * 0.1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.cab = CAB(embed_dim, cab_reduction)
        self.s2 = nn.Parameter(torch.ones(embed_dim) * 0.1)

    def forward(self, x, H, W):
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        # Part 1: Self-Attention + skip
        x_hat = self.attention(self.norm1(x)) + self.s1 * x

        # Part 2: Conv + CAB + skip
        z = self.norm2(x_hat)
        z = z.permute(0, 2, 1).reshape(B, D, H, W)
        z = self.cab(self.conv(z))
        z = z.reshape(B, D, T).permute(0, 2, 1)

        return z + self.s2 * x_hat


class MambaLayer(nn.Module):
    """
    Mamba Layer (Residual Group) — used in the first N/2 encoders.

    Structure (from paper Section 3.3):
      input → [MambaBlock₁ → MambaBlock₂ → ... → MambaBlock_L] → Conv3×3 → + input

    The layer-level 3×3 Conv and residual skip ensure stable gradient
    flow and act as a local feature refinement after the Mamba blocks.
    """

    def __init__(self, embed_dim, num_blocks, ssm_state_dim=16, cab_reduction=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, ssm_state_dim, cab_reduction)
            for _ in range(num_blocks)
        ])
        # Layer-level 3×3 Conv (paper: "followed by a 3×3 convolution")
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x, H, W):
        B, T, D = x.shape
        residual = x  # Layer-level residual

        # Pass through all L Mamba Blocks
        for block in self.blocks:
            x = block(x, H, W)

        # Layer-level Conv (operates in 2D spatial domain)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)    # tokens → 2D
        x = self.conv(x)
        x = x.reshape(B, D, T).permute(0, 2, 1)        # 2D → tokens

        # Layer-level residual skip
        return x + residual


class TransformerLayer(nn.Module):
    """
    Transformer Layer (Residual Group) — used in later encoders + decoder.

    Structure (from paper Section 3.3):
      input → [AttnBlock₁ → AttnBlock₂ → ... → AttnBlock_L] → Conv3×3 → + input

    Same Residual Group pattern as MambaLayer, but with Attention blocks.
    """

    def __init__(self, embed_dim, num_blocks, num_heads, cab_reduction=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, cab_reduction)
            for _ in range(num_blocks)
        ])
        # Layer-level 3×3 Conv (paper: "followed by a 3×3 convolution")
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x, H, W):
        B, T, D = x.shape
        residual = x  # Layer-level residual

        # Pass through all L Attention Blocks
        for block in self.blocks:
            x = block(x, H, W)

        # Layer-level Conv (operates in 2D spatial domain)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)
        x = self.conv(x)
        x = x.reshape(B, D, T).permute(0, 2, 1)

        # Layer-level residual skip
        return x + residual


class FFM(nn.Module):
    """
    Feature Fusion Module (paper Section 3.4, Fig. 2 bottleneck).

    Aggregates hierarchical features from all encoders + shallow features:
      Concat → DepthwiseConv → SiLU → PointwiseConv → LayerNorm
    """

    def __init__(self, embed_dim, num_sources):
        super().__init__()
        total_dim = num_sources * embed_dim

        self.dw_conv = nn.Conv2d(total_dim, total_dim, 3, 1, 1, groups=total_dim)
        self.act = nn.SiLU()
        self.pw_conv = nn.Conv2d(total_dim, embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feature_list, H, W):
        """
        feature_list: list of (B, T, D) tensors
        Returns: (B, T, D)
        """
        B = feature_list[0].shape[0]

        # Concat along feature dim → (B, T, num_sources * D)
        x = torch.cat(feature_list, dim=-1)
        total_dim = x.shape[-1]

        # Tokens → 2D spatial
        x = x.permute(0, 2, 1).reshape(B, total_dim, H, W)

        # Process
        x = self.act(self.dw_conv(x))
        x = self.pw_conv(x)  # (B, D, H, W)

        # 2D → tokens + normalize
        x = x.reshape(B, -1, H * W).permute(0, 2, 1)  # (B, T, D)
        x = self.norm(x)

        return x


print("Blocks, Layers, FFM defined ✓")


# %%
# =============================================================================
# CELL 7: MODEL — Full SRMamba-T
# =============================================================================

class ReconstructionHead(nn.Module):
    """
    Upsampling head for SR reconstruction.
    Uses PixelShuffle for spatial upsampling + final Conv for channel mapping.

    For ×2: Conv→PS(2)→Conv
    For ×3: Conv→PS(3)→Conv
    For ×4: Conv→PS(2)→Conv→PS(2)→Conv  (progressive)
    """

    def __init__(self, embed_dim, out_channels, scale):
        super().__init__()
        self.scale = scale

        layers = []
        if scale == 2:
            layers += [
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ]
        elif scale == 3:
            layers += [
                nn.Conv2d(embed_dim, embed_dim * 9, 3, 1, 1),
                nn.PixelShuffle(3),
            ]
        elif scale == 4:
            layers += [
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ]
        else:
            raise ValueError(f"Unsupported scale: {scale}")

        # Final conv: embed_dim channels → spectral channels
        layers.append(nn.Conv2d(embed_dim, out_channels, 3, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SRMambaT(nn.Module):
    """
    SRMamba-T: Hybrid Mamba-Transformer for Hyperspectral Super-Resolution.

    Architecture (3 stages):
      1. Shallow Feature Extraction: 3×3 Conv
      2. Deep Feature Extraction:
         - Encoder 1 (Mamba Layer)
         - Encoder 2 (Mamba Layer)
         - Encoder 3 (Transformer Layer)
         - FFM Bottleneck (fuses shallow + all encoder features)
         - Decoder (Transformer Layer)
      3. Reconstruction: Conv + PixelShuffle
    """

    def __init__(self, in_channels, embed_dim, scale, num_heads,
                 blocks_per_layer, ssm_state_dim, cab_reduction):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale

        # === Stage 1: Shallow Feature Extraction ===
        self.shallow_conv = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # === Stage 2: Deep Feature Extraction ===
        # N=4 layers: 2 Mamba (enc) + 1 Transformer (enc) + 1 Transformer (dec)
        self.mamba_enc1 = MambaLayer(
            embed_dim, blocks_per_layer, ssm_state_dim, cab_reduction
        )
        self.mamba_enc2 = MambaLayer(
            embed_dim, blocks_per_layer, ssm_state_dim, cab_reduction
        )
        self.trans_enc = TransformerLayer(
            embed_dim, blocks_per_layer, num_heads, cab_reduction
        )

        # FFM: fuses 3 encoder outputs + shallow features = 4 sources
        self.ffm = FFM(embed_dim, num_sources=4)

        # Decoder
        self.trans_dec = TransformerLayer(
            embed_dim, blocks_per_layer, num_heads, cab_reduction
        )

        # === Stage 3: Reconstruction ===
        self.reconstruction = ReconstructionHead(embed_dim, in_channels, scale)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_backbone_state_dict(self):
        """Get all weights EXCEPT the reconstruction head."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith('reconstruction')}

    def load_backbone_weights(self, backbone_dict):
        """Load backbone weights, ignore reconstruction head."""
        model_dict = self.state_dict()
        loaded = 0
        for k, v in backbone_dict.items():
            if k in model_dict and not k.startswith('reconstruction'):
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1
        self.load_state_dict(model_dict)
        print(f"  Loaded {loaded} backbone weight tensors")

    def forward(self, x):
        B, C, H, W = x.shape

        # Stage 1: Shallow features
        f_s = self.shallow_conv(x)                          # (B, D, H, W)

        # Patch embedding: 2D → 1D tokens
        f_t = f_s.flatten(2).transpose(1, 2)                # (B, T, D)

        # Stage 2: Deep feature extraction
        f_enc1 = self.mamba_enc1(f_t, H, W)                 # Encoder 1
        f_enc2 = self.mamba_enc2(f_enc1, H, W)              # Encoder 2
        f_enc3 = self.trans_enc(f_enc2, H, W)               # Encoder 3

        # FFM: fuse shallow + encoder features
        f_s_tokens = f_s.flatten(2).transpose(1, 2)
        f_fused = self.ffm([f_s_tokens, f_enc1, f_enc2, f_enc3], H, W)

        # Decoder
        f_d = self.trans_dec(f_fused, H, W)

        # Patch unembedding: 1D → 2D
        f_d_2d = f_d.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Residual connection with shallow features
        f = f_s + f_d_2d

        # Stage 3: Reconstruction (upsample)
        out = self.reconstruction(f)

        return out


def create_model(config, scale):
    """Create a fresh SRMamba-T model for the given scale."""
    model = SRMambaT(
        in_channels=config['in_channels'],
        embed_dim=config['embed_dim'],
        scale=scale,
        num_heads=config['num_heads'],
        blocks_per_layer=config['blocks_per_layer'],
        ssm_state_dim=config['ssm_state_dim'],
        cab_reduction=config['cab_reduction'],
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created — scale ×{scale}")
    print(f"  Total params:     {total:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    return model


# Quick test: build model and run a dummy forward pass
print("\nTesting model...")
_model = create_model(CONFIG, scale=3)
_x = torch.randn(1, CONFIG['in_channels'], 21, 21)
with torch.no_grad():
    _y = _model(_x)
print(f"  Input:  {_x.shape}")
print(f"  Output: {_y.shape}")
expected_h = 21 * 3  # = 63 (PixelShuffle for ×3)
print(f"  Expected output H: {expected_h} (PixelShuffle), got: {_y.shape[2]}")
del _model, _x, _y
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("  Model test PASSED ✓")


# %%
# =============================================================================
# CELL 8: LOSS FUNCTION & METRICS
# =============================================================================

class DualDomainLoss(nn.Module):
    """
    Dual-domain loss from paper (Eq. 5):
      L = MSE_pixel + λ · MSE_frequency

    Combines pixel-space accuracy with frequency-space sharpness.
    """

    def __init__(self, lambda_freq=0.1):
        super().__init__()
        self.lambda_freq = lambda_freq

    def forward(self, sr, hr):
        # Pixel domain
        pixel_loss = F.mse_loss(sr, hr)

        # Frequency domain (FFT magnitude)
        sr_fft = torch.fft.rfft2(sr, dim=(-2, -1))
        hr_fft = torch.fft.rfft2(hr, dim=(-2, -1))
        freq_loss = F.mse_loss(torch.abs(sr_fft), torch.abs(hr_fft))

        return pixel_loss + self.lambda_freq * freq_loss


def compute_psnr(sr, hr, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio (dB)."""
    mse = F.mse_loss(sr, hr).item()
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction
    return 10.0 * math.log10(max_val ** 2 / mse)


print("Loss & Metrics defined ✓")


# %%
# =============================================================================
# CELL 9: TRAINING & VALIDATION FUNCTIONS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, config):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (lr, hr) in enumerate(loader):
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if config['use_amp'] and scaler is not None:
            with torch.cuda.amp.autocast():
                sr = model(lr)
                if sr.shape != hr.shape:
                    sr = F.interpolate(sr, size=hr.shape[-2:],
                                       mode='bilinear', align_corners=False)
                loss = criterion(sr, hr)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            sr = model(lr)
            if sr.shape != hr.shape:
                sr = F.interpolate(sr, size=hr.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss = criterion(sr, hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

        total_loss += loss.item()

        # Progress log every 50 batches
        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            avg = total_loss / (i + 1)
            print(f"    Batch {i+1:>4d}/{n_batches} | Loss: {loss.item():.6f} | Avg: {avg:.6f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, config):
    """Validate and return average PSNR (dB)."""
    model.eval()
    total_psnr = 0.0
    count = 0

    for lr, hr in loader:
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        if config['use_amp']:
            with torch.cuda.amp.autocast():
                sr = model(lr)
        else:
            sr = model(lr)

        if sr.shape != hr.shape:
            sr = F.interpolate(sr, size=hr.shape[-2:],
                               mode='bilinear', align_corners=False)
        sr = sr.clamp(0, 1)

        for j in range(sr.shape[0]):
            total_psnr += compute_psnr(sr[j:j+1], hr[j:j+1])
            count += 1

    return total_psnr / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, epoch, scale, config, path):
    """Save full checkpoint (model + optimizer + scheduler state)."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        'model_state_dict': m.state_dict(),
        'backbone_state_dict': m.get_backbone_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'scale': scale,
        'config': {k: v for k, v in config.items()},
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"    💾 Saved: {path} ({size_mb:.1f} MB)")


def train_scale(scale, config, prev_checkpoint_path=None, resume_checkpoint_path=None):
    """
    Full training pipeline for one scale factor.

    Args:
        scale: upsampling factor (2, 3, or 4)
        config: CONFIG dict
        prev_checkpoint_path: path to previous scale's checkpoint for
                              continual training (loads backbone weights)
        resume_checkpoint_path: path to resume training from a saved checkpoint
                                (loads full model + optimizer + scheduler state)

    Returns:
        checkpoint_path: path to the saved final checkpoint
    """
    print("\n" + "=" * 70)
    print(f"  TRAINING — Scale ×{scale} | {config['dataset_name']}")
    print("=" * 70)

    # --- Create model ---
    model = create_model(config, scale)

    # --- Load backbone from previous scale ---
    if prev_checkpoint_path:
        # Try direct path first
        ckpt_path = prev_checkpoint_path
        if not os.path.exists(ckpt_path):
            # Search for the checkpoint file in /kaggle/input/
            print(f"\n  ⚠️ Checkpoint not found at: {ckpt_path}")
            print(f"  🔍 Searching for checkpoint in /kaggle/input/...")
            found_files = glob.glob('/kaggle/input/**/*.pth', recursive=True)
            print(f"  Found .pth files: {found_files}")
            # Use the first .pth file found
            if found_files:
                ckpt_path = found_files[0]
                print(f"  ✅ Using: {ckpt_path}")
            else:
                ckpt_path = None
                print(f"  ❌ No checkpoint found. Training from scratch.")
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n  Loading backbone from previous scale: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            print(f"  Checkpoint keys: {list(ckpt.keys())}")
            if 'backbone_state_dict' in ckpt:
                model.load_backbone_weights(ckpt['backbone_state_dict'])
                print(f"  ✅ Backbone weights loaded successfully!")
            else:
                print(f"  ⚠️ No backbone_state_dict in checkpoint. Training from scratch.")
            del ckpt
            gc.collect()
        else:
            print("\n  Training from scratch (checkpoint not found)")
    else:
        print("\n  Training from scratch (no previous checkpoint)")

    # --- Multi-GPU ---
    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Dataset ---
    dataset = HyperspectralSRDataset(
        config['data_dir'], scale, config['max_val'], augment=True
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )

    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # --- Optimizer / Scheduler / Loss ---
    optimizer = Adam(model.parameters(), lr=config['lr'],
                     betas=(0.9, 0.99), eps=1e-8)
    # MultiStepLR: halve LR at milestones (paper Section 4.1)
    # Milestones at ~60%, 80%, 90%, 95% of total epochs (MambaIR convention)
    total_epochs = config['epochs_per_scale']
    milestones = [
        int(total_epochs * 0.6),
        int(total_epochs * 0.8),
        int(total_epochs * 0.9),
        int(total_epochs * 0.95),
    ]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    criterion = DualDomainLoss(lambda_freq=config['lambda_freq'])
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    # --- Paths ---
    final_path = os.path.join(config['save_dir'], f"srmamba_t_x{scale}_final.pth")
    best_path = os.path.join(config['save_dir'], f"srmamba_t_x{scale}_best.pth")

    best_psnr = 0.0
    start_epoch = 0

    # --- Resume from checkpoint if provided ---
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"\n  🔄 Resuming from checkpoint: {resume_checkpoint_path}")
        ckpt = torch.load(resume_checkpoint_path, map_location='cpu')
        m = model.module if isinstance(model, nn.DataParallel) else model
        m.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # Reset optimizer LR and scheduler for extended training
        for pg in optimizer.param_groups:
            pg['lr'] = config['lr']
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        print(f"  ✅ Resumed! Starting from epoch {start_epoch + 1}/{total_epochs}")
        print(f"  🔄 LR reset to {config['lr']:.1e} (fresh schedule for extended training)")
        del ckpt
        gc.collect()

    print(f"\n  Starting training: epochs {start_epoch + 1} to {total_epochs}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.1e}, milestones: {milestones} (MultiStepLR)")
    print("-" * 70)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, config
        )

        # Validate
        val_psnr = validate(model, val_loader, config)

        # Clean memory every epoch to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()

        # Step LR
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Log
        marker = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, scheduler, epoch, scale, config, best_path)
            marker = " ★ BEST"

        print(
            f"  Epoch {epoch+1:>3d}/{total_epochs} | "
            f"Loss: {train_loss:.6f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"LR: {lr_now:.2e} | "
            f"Time: {elapsed:.0f}s{marker}"
        )

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, total_epochs - 1,
                    scale, config, final_path)

    print("-" * 70)
    print(f"  ✅ Scale ×{scale} complete!")
    print(f"     Best PSNR: {best_psnr:.2f} dB")
    print(f"     Final:  {final_path}")
    print(f"     Best:   {best_path}")

    # Cleanup GPU memory
    del model, optimizer, scheduler, criterion, scaler
    del train_loader, val_loader
    torch.cuda.empty_cache()

    return final_path


print("Training functions defined ✓")


# %%
# =============================================================================
# CELL 10: SKIP ×3 (already done - 30 epochs, 18.32 dB)
# =============================================================================
print("🚀 Training pipeline: resume ×4 only (PaviaU)")
print(f"   Dataset: {CONFIG['dataset_name']} | Data: {CONFIG['data_dir']}")

ckpt_x3 = '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x3_best.pth'
print(f"\n  ✅ ×3: Done (30 epochs, PSNR 18.32 dB)")


# %%
# =============================================================================
# CELL 11: SKIP ×2 (already done - 30 epochs)
# =============================================================================
ckpt_x2 = '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x2_best.pth'
print(f"  ✅ ×2: Done (30 epochs, PSNR 18.66 dB)")


# %%
# =============================================================================
# CELL 12: RESUME ×4 TRAINING
# =============================================================================
ckpt_x4_resume = '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x4_best.pth'
ckpt_x4 = train_scale(
    scale=4,
    config=CONFIG,
    prev_checkpoint_path=ckpt_x2,
    resume_checkpoint_path=ckpt_x4_resume
)

print("\n" + "=" * 70)
print("🎉 ALL TRAINING COMPLETE!")
print("=" * 70)
print(f"  Checkpoints saved in: {CONFIG['save_dir']}")
for f in sorted(os.listdir(CONFIG['save_dir'])):
    size = os.path.getsize(os.path.join(CONFIG['save_dir'], f)) / 1e6
    print(f"    {f} ({size:.1f} MB)")


# %%
# =============================================================================
# CELL 13: EVALUATION & VISUALIZATION
# =============================================================================
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt


def evaluate_and_visualize(checkpoint_path, config, scale, num_samples=5):
    """Load a trained model, evaluate, and visualize LR/SR/HR comparisons."""
    print(f"\n{'='*50}")
    print(f"  Evaluating scale ×{scale}")
    print(f"{'='*50}")

    # Load model
    model = create_model(config, scale)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Full dataset (no augmentation)
    dataset = HyperspectralSRDataset(
        config['data_dir'], scale, config['max_val'], augment=False
    )

    # Compute PSNR on entire dataset
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=False, num_workers=2)
    avg_psnr = validate(model, loader, config)
    print(f"  Average PSNR on full dataset: {avg_psnr:.2f} dB")

    # Visualize random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    # Select 3 spectral bands for pseudo-RGB visualization
    # Adjust these indices based on your spectral range
    rgb_bands = [60, 30, 10]  # Red-ish, Green-ish, Blue-ish

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        lr, hr = dataset[idx]
        lr_in = lr.unsqueeze(0).to(device)

        with torch.no_grad():
            if config['use_amp']:
                with torch.cuda.amp.autocast():
                    sr = model(lr_in)
            else:
                sr = model(lr_in)

        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:],
                               mode='bilinear', align_corners=False)
        sr = sr.clamp(0, 1).cpu()

        # Compute per-sample PSNR
        psnr = compute_psnr(sr[0:1], hr.unsqueeze(0))

        # Extract pseudo-RGB for display
        def to_rgb(tensor, bands):
            img = tensor[bands].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        lr_rgb = to_rgb(lr, rgb_bands)
        sr_rgb = to_rgb(sr[0], rgb_bands)
        hr_rgb = to_rgb(hr, rgb_bands)

        axes[row, 0].imshow(lr_rgb)
        axes[row, 0].set_title(f'LR ({lr.shape[1]}×{lr.shape[2]})', fontsize=11)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(sr_rgb)
        axes[row, 1].set_title(f'SR ×{scale} — PSNR: {psnr:.1f} dB', fontsize=11)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(hr_rgb)
        axes[row, 2].set_title(f'HR ({hr.shape[1]}×{hr.shape[2]})', fontsize=11)
        axes[row, 2].axis('off')

    plt.suptitle(
        f'SRMamba-T ×{scale} | {config["dataset_name"]} | Avg PSNR: {avg_psnr:.2f} dB',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    save_path = os.path.join(config['save_dir'], f'results_x{scale}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Results saved: {save_path}")

    del model
    torch.cuda.empty_cache()
    return avg_psnr


# --- Run evaluation for all scales ---
results = {}
for scale in CONFIG['train_scales']:
    best_ckpt = os.path.join(CONFIG['save_dir'], f'srmamba_t_x{scale}_best.pth')
    if os.path.exists(best_ckpt):
        psnr = evaluate_and_visualize(best_ckpt, CONFIG, scale, num_samples=5)
        results[f'×{scale}'] = psnr
    else:
        print(f"  Checkpoint not found for ×{scale}: {best_ckpt}")

print("\n" + "=" * 50)
print("  FINAL RESULTS SUMMARY")
print("=" * 50)
for scale_str, psnr in results.items():
    print(f"  {scale_str}: {psnr:.2f} dB PSNR")


# %%
# =============================================================================
# CELL 14: DOWNLOAD MODELS (Run this to save models from Kaggle)
# =============================================================================
# After training, your checkpoints are in /kaggle/working/checkpoints/
# Kaggle auto-saves everything in /kaggle/working/ as "Output"
#
# To download: click "Save Version" → "Save & Run All" → wait for it to
# complete → go to your notebook → click "Output" tab → download files
#
# Or copy to a specific location:
import shutil
output_dir = '/kaggle/working/final_models'
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(CONFIG['save_dir']):
    if f.endswith('.pth'):
        src = os.path.join(CONFIG['save_dir'], f)
        dst = os.path.join(output_dir, f)
        shutil.copy2(src, dst)
        print(f"  Copied: {f}")
print(f"\nAll models saved to: {output_dir}")
print("Download from Kaggle: Notebook → Output tab → Download")


# %%
# =============================================================================
# CELL 15: TRAIN ON PaviaU DATASET
# =============================================================================
# Now we train the same SRMamba-T architecture on PaviaU (103 bands).
# All previous functions (train_scale, evaluate_and_visualize, etc.) are reused.
# Checkpoints are saved separately in checkpoints_paviaU/.
# =============================================================================

print("\n" + "=" * 70)
print("  🚀 NOW TRAINING ON PaviaU DATASET (103 bands)")
print("=" * 70)

# Save PaviaC results before overwriting CONFIG
paviac_results = dict(results)

# Update CONFIG for PaviaU
CONFIG['dataset_name'] = 'PaviaU'
CONFIG['in_channels'] = 103              # PaviaU has 103 bands (PaviaC had 102)
CONFIG['data_dir'] = '/kaggle/input/datasets/vujjapunithsai/btpdataset/PaviaU_Data'
CONFIG['save_dir'] = '/kaggle/working/checkpoints_paviaU'
os.makedirs(CONFIG['save_dir'], exist_ok=True)

print(f"  Dataset: {CONFIG['dataset_name']} ({CONFIG['in_channels']} bands)")
print(f"  Data dir: {CONFIG['data_dir']}")
print(f"  Save dir: {CONFIG['save_dir']}")

# Train all 3 scales: ×3 → ×2 → ×4 (same order as PaviaC)
ckpt_u_x3 = train_scale(scale=3, config=CONFIG, prev_checkpoint_path=None)
ckpt_u_x2 = train_scale(scale=2, config=CONFIG, prev_checkpoint_path=ckpt_u_x3)
ckpt_u_x4 = train_scale(scale=4, config=CONFIG, prev_checkpoint_path=ckpt_u_x2)

print("\n" + "=" * 70)
print("  🎉 PaviaU TRAINING COMPLETE!")
print("=" * 70)

# Evaluate PaviaU
paviau_results = {}
for scale in CONFIG['train_scales']:
    best_ckpt = os.path.join(CONFIG['save_dir'], f'srmamba_t_x{scale}_best.pth')
    if os.path.exists(best_ckpt):
        psnr = evaluate_and_visualize(best_ckpt, CONFIG, scale, num_samples=5)
        paviau_results[f'×{scale}'] = psnr
    else:
        print(f"  Checkpoint not found for ×{scale}: {best_ckpt}")

# --- Final Summary: Both Datasets ---
print("\n" + "=" * 70)
print("  🏆 FINAL RESULTS — BOTH DATASETS")
print("=" * 70)
print("\n  PaviaC (102 bands):")
for s, p in paviac_results.items():
    print(f"    {s}: {p:.2f} dB PSNR")
print("\n  PaviaU (103 bands):")
for s, p in paviau_results.items():
    print(f"    {s}: {p:.2f} dB PSNR")
print("=" * 70)


# %%
# =============================================================================
# CELL 16: DOWNLOAD ALL MODELS (Both PaviaC + PaviaU)
# =============================================================================
output_dir_all = '/kaggle/working/final_models_all'
os.makedirs(output_dir_all, exist_ok=True)

# Copy PaviaC checkpoints
paviac_ckpt_dir = '/kaggle/working/checkpoints'
if os.path.isdir(paviac_ckpt_dir):
    for f in os.listdir(paviac_ckpt_dir):
        if f.endswith('.pth') or f.endswith('.png'):
            src = os.path.join(paviac_ckpt_dir, f)
            dst = os.path.join(output_dir_all, f'PaviaC_{f}')
            shutil.copy2(src, dst)
            print(f"  Copied: PaviaC_{f}")

# Copy PaviaU checkpoints
paviau_ckpt_dir = '/kaggle/working/checkpoints_paviaU'
if os.path.isdir(paviau_ckpt_dir):
    for f in os.listdir(paviau_ckpt_dir):
        if f.endswith('.pth') or f.endswith('.png'):
            src = os.path.join(paviau_ckpt_dir, f)
            dst = os.path.join(output_dir_all, f'PaviaU_{f}')
            shutil.copy2(src, dst)
            print(f"  Copied: PaviaU_{f}")

print(f"\nAll models saved to: {output_dir_all}")
print("Download from Kaggle: Notebook → Output tab → Download")
