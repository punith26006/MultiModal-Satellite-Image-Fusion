# %% [markdown]
# # SRMamba-T — Model Testing & Evaluation
# Tests all trained models and generates comparison images for BTP report.

# %%
# =============================================================================
# CELL 1: IMPORTS & SETUP
# =============================================================================
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %%
# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================
EVAL_CONFIG = {
    'datasets': {
        'PaviaC': {
            'data_dir': '/kaggle/input/datasets/vujjapunithsai/btpdataset/PaviaC_Data',
            'in_channels': 102,
            'models': {
                2: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviac/srmamba_t_x2_best.pth',
                3: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviac/srmamba_t_x3_best.pth',
                4: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviac/srmamba_t_x4_best.pth',
            }
        },
        'PaviaU': {
            'data_dir': '/kaggle/input/datasets/vujjapunithsai/btpdataset/PaviaU_Data',
            'in_channels': 103,
            'models': {
                2: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x2_best.pth',
                3: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x3_best.pth',
                4: '/kaggle/input/datasets/vujjapunithsai/btpcheckpoint-paviau/srmamba_t_x4_best.pth',
            }
        }
    },
    'max_val': 10000.0,
    'embed_dim': 48,
    'num_heads': 6,
    'blocks_per_layer': 4,
    'ssm_state_dim': 16,
    'cab_reduction': 8,
    'use_amp': True,
    'save_dir': '/kaggle/working/results',
}

os.makedirs(EVAL_CONFIG['save_dir'], exist_ok=True)
print("Evaluation config loaded!")


# %%
# =============================================================================
# CELL 3: DATASET CLASS
# =============================================================================
class HyperspectralSRDataset(Dataset):
    def __init__(self, data_dir, scale, max_val=10000.0):
        super().__init__()
        self.hr_dir = os.path.join(data_dir, 'HR')
        self.lr_dir = os.path.join(data_dir, f'LR_{scale}')
        self.max_val = max_val
        self.hr_files = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.npy')])
        self.lr_files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.npy')])
        assert len(self.hr_files) == len(self.lr_files)

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr = np.load(os.path.join(self.hr_dir, self.hr_files[idx])).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, self.lr_files[idx])).astype(np.float32)
        hr = hr / self.max_val
        lr = lr / self.max_val
        hr = torch.from_numpy(hr).permute(2, 0, 1)
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        return lr, hr

print("Dataset class defined")


# %%
# =============================================================================
# CELL 4: MODEL ARCHITECTURE
# =============================================================================
def selective_scan(u, delta, A, B, C, D=None):
    B_batch, L, D_in = u.shape
    N = A.shape[1]
    deltaA = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A))
    deltaB_u = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)
    x = torch.zeros(B_batch, D_in, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = torch.einsum('b d n, b n -> b d', x, C[:, i])
        ys.append(y)
    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)
    return y


class MDSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_dirs = 4
        self.in_proj = nn.Linear(d_model, d_model * 2 * self.n_dirs)
        self.A_log = nn.Parameter(torch.randn(self.n_dirs, d_model, d_state))
        self.D = nn.Parameter(torch.ones(self.n_dirs, d_model))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_model * self.n_dirs, d_model)

    def forward(self, x):
        B_batch, C, H, W = x.shape
        L = H * W
        def create_scan_sequences(x_flat):
            seqs = [x_flat]
            seqs.append(torch.flip(x_flat, dims=[1]))
            x_2d = x_flat.view(B_batch, H, W, C)
            vert = x_2d.permute(0, 2, 1, 3).reshape(B_batch, L, C)
            seqs.append(vert)
            seqs.append(torch.flip(vert, dims=[1]))
            return seqs
        x_flat = x.permute(0, 2, 3, 1).reshape(B_batch, L, C)
        scan_seqs = create_scan_sequences(x_flat)
        outputs = []
        for d in range(self.n_dirs):
            seq = scan_seqs[d]
            zx = self.in_proj(seq).chunk(2 * self.n_dirs, dim=-1)
            z = zx[d * 2]
            x_d = zx[d * 2 + 1]
            A = -torch.exp(self.A_log[d])
            delta = F.softplus(self.delta_proj(x_d))
            B_param = self.B_proj(x_d)
            C_param = self.C_proj(x_d)
            y = selective_scan(x_d, delta, A, B_param, C_param, self.D[d])
            y = y * F.silu(z)
            if d == 1:
                y = torch.flip(y, dims=[1])
            elif d == 2:
                y = y.view(B_batch, W, H, self.d_model).permute(0, 2, 1, 3).reshape(B_batch, L, self.d_model)
            elif d == 3:
                y = torch.flip(y, dims=[1])
                y = y.view(B_batch, W, H, self.d_model).permute(0, 2, 1, 3).reshape(B_batch, L, self.d_model)
            outputs.append(y)
        y_cat = torch.cat(outputs, dim=-1)
        out = self.out_proj(y_cat)
        return out.permute(0, 2, 1).view(B_batch, C, H, W)


class CAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.GELU(), nn.Conv2d(channels, channels, 3, 1, 1))
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1), nn.GELU(), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid())

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        return x + res


class MambaBlock(nn.Module):
    def __init__(self, dim, ssm_state_dim=16, cab_reduction=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mdssm = MDSSM(dim, ssm_state_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cab = CAB(dim, cab_reduction)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm1(x_flat).permute(0, 2, 1).view(B, C, H, W)
        x = x + self.mdssm(x_norm)
        x_flat2 = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm2 = self.norm2(x_flat2).permute(0, 2, 1).view(B, C, H, W)
        x = x + self.cab(x_norm2)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class MambaLayer(nn.Module):
    def __init__(self, dim, n_blocks, ssm_state_dim, cab_reduction):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(dim, ssm_state_dim, cab_reduction) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class FFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1))

    def forward(self, mamba_feat, attn_feat):
        return self.conv(torch.cat([mamba_feat, attn_feat], dim=1))


class SRMambaT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, blocks_per_layer, ssm_state_dim, cab_reduction, scale):
        super().__init__()
        self.scale = scale
        self.shallow_feat = nn.Sequential(nn.Conv2d(in_channels, embed_dim, 3, 1, 1), nn.GELU(), nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
        self.mamba_layers = nn.ModuleList([MambaLayer(embed_dim, blocks_per_layer, ssm_state_dim, cab_reduction) for _ in range(2)])
        self.sa_layers = nn.ModuleList([SelfAttentionBlock(embed_dim, num_heads) for _ in range(2)])
        self.ffms = nn.ModuleList([FFM(embed_dim) for _ in range(2)])
        self.conv_before_up = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        if scale in [2, 3]:
            self.upsample = nn.Sequential(nn.Conv2d(embed_dim, in_channels * scale * scale, 3, 1, 1), nn.PixelShuffle(scale))
        elif scale == 4:
            self.upsample = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(embed_dim, in_channels * 4, 3, 1, 1), nn.PixelShuffle(2))

    def forward(self, x):
        shallow = self.shallow_feat(x)
        deep = shallow
        for i in range(2):
            mamba_out = self.mamba_layers[i](deep)
            sa_out = self.sa_layers[i](deep)
            deep = self.ffms[i](mamba_out, sa_out) + deep
        deep = self.conv_before_up(deep + shallow)
        return self.upsample(deep)


def create_model(in_ch, config, scale):
    return SRMambaT(in_ch, config['embed_dim'], config['num_heads'], config['blocks_per_layer'], config['ssm_state_dim'], config['cab_reduction'], scale)

print("Model architecture defined")


# %%
# =============================================================================
# CELL 5: METRICS
# =============================================================================
def compute_psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse).item()

def compute_ssim(sr, hr):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_sr = F.avg_pool2d(sr, 3, 1, 1)
    mu_hr = F.avg_pool2d(hr, 3, 1, 1)
    sigma_sr = F.avg_pool2d(sr ** 2, 3, 1, 1) - mu_sr ** 2
    sigma_hr = F.avg_pool2d(hr ** 2, 3, 1, 1) - mu_hr ** 2
    sigma_srhr = F.avg_pool2d(sr * hr, 3, 1, 1) - mu_sr * mu_hr
    ssim_map = ((2 * mu_sr * mu_hr + C1) * (2 * sigma_srhr + C2)) / ((mu_sr ** 2 + mu_hr ** 2 + C1) * (sigma_sr + sigma_hr + C2))
    return ssim_map.mean().item()

print("Metrics defined")


# %%
# =============================================================================
# CELL 6: EVALUATE AND VISUALIZE
# =============================================================================
def evaluate_model(dataset_name, dataset_cfg, scale, config):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {dataset_name} x{scale}")
    print(f"{'='*60}")

    model_path = dataset_cfg['models'][scale]
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None, None

    in_ch = dataset_cfg['in_channels']
    model = create_model(in_ch, config, scale)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')})")
    del ckpt

    dataset = HyperspectralSRDataset(dataset_cfg['data_dir'], scale, config['max_val'])
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"  Dataset: {len(dataset)} patches")

    all_psnr, all_ssim = [], []
    with torch.no_grad():
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            with torch.amp.autocast('cuda', enabled=config['use_amp']):
                sr_b = model(lr_b)
            if sr_b.shape[-2:] != hr_b.shape[-2:]:
                sr_b = F.interpolate(sr_b, size=hr_b.shape[-2:], mode='bilinear', align_corners=False)
            sr_b = sr_b.clamp(0, 1)
            for i in range(sr_b.shape[0]):
                all_psnr.append(compute_psnr(sr_b[i:i+1], hr_b[i:i+1]))
                all_ssim.append(compute_ssim(sr_b[i:i+1], hr_b[i:i+1]))

    avg_psnr, avg_ssim = np.mean(all_psnr), np.mean(all_ssim)
    print(f"  PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    # Visualize 5 samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(5, len(dataset)), replace=False)
    rgb_bands = [60, 30, 10]
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))

    for row, idx in enumerate(indices):
        lr, hr = dataset[idx]
        lr_in = lr.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=config['use_amp']):
                sr = model(lr_in)
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        sr = sr.clamp(0, 1).cpu()
        psnr = compute_psnr(sr[0:1], hr.unsqueeze(0))
        ssim_val = compute_ssim(sr[0:1], hr.unsqueeze(0))

        def to_rgb(tensor, bands):
            img = tensor[bands].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img

        lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode='nearest')[0]
        axes[row, 0].imshow(to_rgb(lr_up, rgb_bands))
        axes[row, 0].set_title(f'LR ({lr.shape[1]}x{lr.shape[2]})', fontsize=12)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(to_rgb(sr[0], rgb_bands))
        axes[row, 1].set_title(f'SR x{scale} | PSNR:{psnr:.1f} | SSIM:{ssim_val:.3f}', fontsize=11)
        axes[row, 1].axis('off')
        axes[row, 2].imshow(to_rgb(hr, rgb_bands))
        axes[row, 2].set_title(f'HR ({hr.shape[1]}x{hr.shape[2]})', fontsize=12)
        axes[row, 2].axis('off')

    plt.suptitle(f'SRMamba-T x{scale} | {dataset_name} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config['save_dir'], f'{dataset_name}_x{scale}_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_path}")
    del model; torch.cuda.empty_cache(); gc.collect()
    return avg_psnr, avg_ssim

print("Evaluation function defined")


# %%
# =============================================================================
# CELL 7: RUN ALL EVALUATIONS
# =============================================================================
results = {}
for ds_name, ds_cfg in EVAL_CONFIG['datasets'].items():
    results[ds_name] = {}
    for scale in [2, 3, 4]:
        psnr, ssim = evaluate_model(ds_name, ds_cfg, scale, EVAL_CONFIG)
        if psnr is not None:
            results[ds_name][f'x{scale}'] = {'PSNR': psnr, 'SSIM': ssim}


# %%
# =============================================================================
# CELL 8: FINAL RESULTS TABLE
# =============================================================================
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Dataset':<10} {'Scale':<8} {'PSNR (dB)':<12} {'SSIM':<10}")
print("-" * 40)
for ds_name, ds_results in results.items():
    for scale_str, metrics in ds_results.items():
        print(f"{ds_name:<10} {scale_str:<8} {metrics['PSNR']:<12.2f} {metrics['SSIM']:<10.4f}")
print("\n  ALL EVALUATIONS COMPLETE!")


# %%
# =============================================================================
# CELL 9: COPY TO OUTPUT
# =============================================================================
import shutil
output_dir = '/kaggle/working/final_results'
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(EVAL_CONFIG['save_dir']):
    src = os.path.join(EVAL_CONFIG['save_dir'], f)
    dst = os.path.join(output_dir, f)
    shutil.copy2(src, dst)
    print(f"  Copied: {f}")
print(f"\nDownload from Output tab!")
