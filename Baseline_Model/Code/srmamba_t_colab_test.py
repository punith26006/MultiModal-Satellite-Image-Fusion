# %% [markdown]
# # SRMamba-T — Testing on Google Colab
# Test PaviaC models and generate comparison images

# %%
# CELL 1: MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')

# %%
# CELL 2: SETUP
import os, zipfile

DRIVE_BASE = '/content/drive/MyDrive'
MODEL_DIR = f'{DRIVE_BASE}/BTP_Models/PaviaC'
DATA_ZIP = f'{DRIVE_BASE}/BTP_Data/PaviaC_Data.zip'
DATA_DIR = '/content'  # Files extract directly here

if not os.path.exists(os.path.join(DATA_DIR, 'HR')):
    print("Unzipping dataset...")
    with zipfile.ZipFile(DATA_ZIP, 'r') as z:
        z.extractall('/content/')
    print("Done!")

for d in ['HR', 'LR_2', 'LR_3', 'LR_4']:
    path = os.path.join(DATA_DIR, d)
    if os.path.exists(path):
        print(f"  {d}: {len(os.listdir(path))} files")

# %%
# CELL 3: IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# CELL 4: DATASET
class HyperspectralSRDataset(Dataset):
    def __init__(self, data_dir, scale, max_val=10000.0):
        self.hr_dir = os.path.join(data_dir, 'HR')
        self.lr_dir = os.path.join(data_dir, f'LR_{scale}')
        self.max_val = max_val
        self.hr_files = sorted([f for f in os.listdir(self.hr_dir) if f.endswith('.npy')])
        self.lr_files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.npy')])
    def __len__(self): return len(self.hr_files)
    def __getitem__(self, idx):
        hr = np.load(os.path.join(self.hr_dir, self.hr_files[idx])).astype(np.float32) / self.max_val
        lr = np.load(os.path.join(self.lr_dir, self.lr_files[idx])).astype(np.float32) / self.max_val
        return torch.from_numpy(lr).permute(2, 0, 1), torch.from_numpy(hr).permute(2, 0, 1)
print("Dataset ready")

# %%
# CELL 5: MODEL ARCHITECTURE (exact copy from training notebook)

def selective_scan(x, delta, A, B, C, D_skip):
    B_batch, T, D = x.shape
    N = A.shape[1]
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    deltaBx = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    ys = torch.zeros(B_batch, T, D, device=x.device, dtype=x.dtype)
    h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
    for t in range(T):
        h = deltaA[:, t] * h + deltaBx[:, t]
        ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)
    ys = ys + x * D_skip
    return ys


class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.D = nn.Parameter(torch.ones(d_model))
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.1)
    def forward(self, x):
        A = -torch.exp(self.A_log)
        B = self.B_proj(x)
        C = self.C_proj(x)
        delta = F.softplus(self.dt_proj(x))
        return selective_scan(x, delta, A, B, C, self.D)


class MDSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        assert d_model % 4 == 0
        self.d_model = d_model
        self.d_group = d_model // 4
        self.ssm_blocks = nn.ModuleList([SSMBlock(self.d_group, d_state) for _ in range(4)])

    @staticmethod
    def _scan_2d(x, direction):
        B, C, H, W = x.shape
        if direction == 0: return x.reshape(B, C, -1).permute(0, 2, 1)
        elif direction == 1: return x.permute(0, 1, 3, 2).reshape(B, C, -1).permute(0, 2, 1)
        elif direction == 2: return x.reshape(B, C, -1).flip(-1).permute(0, 2, 1)
        else: return x.permute(0, 1, 3, 2).reshape(B, C, -1).flip(-1).permute(0, 2, 1)

    @staticmethod
    def _unscan_2d(x, H, W, direction):
        B, T, C = x.shape
        if direction == 0: return x.permute(0, 2, 1).reshape(B, C, H, W)
        elif direction == 1: return x.permute(0, 2, 1).reshape(B, C, W, H).permute(0, 1, 3, 2)
        elif direction == 2: return x.permute(0, 2, 1).flip(-1).reshape(B, C, H, W)
        else: return x.permute(0, 2, 1).flip(-1).reshape(B, C, W, H).permute(0, 1, 3, 2)

    def forward(self, x, H, W):
        B, T, D = x.shape
        groups = x.chunk(4, dim=-1)
        outputs = []
        for i, (group, ssm) in enumerate(zip(groups, self.ssm_blocks)):
            g_2d = group.permute(0, 2, 1).reshape(B, self.d_group, H, W)
            g_1d = self._scan_2d(g_2d, direction=i)
            g_out = ssm(g_1d)
            g_2d_out = self._unscan_2d(g_out, H, W, direction=i)
            outputs.append(g_2d_out.reshape(B, self.d_group, -1).permute(0, 2, 1))
        return torch.cat(outputs, dim=-1)


class MambaMixer(nn.Module):
    def __init__(self, embed_dim, ssm_state_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.half_dim = embed_dim // 2
        self.linear1 = nn.Linear(embed_dim, self.half_dim)
        self.conv1 = nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1)
        self.linear2 = nn.Linear(embed_dim, self.half_dim)
        self.conv2 = nn.Conv2d(self.half_dim, self.half_dim, 3, 1, 1)
        self.mdssm = MDSSM(self.half_dim, ssm_state_dim)
        self.merge = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, H, W):
        B, T, D = x.shape
        x1 = self.linear1(x)
        x1 = x1.permute(0, 2, 1).reshape(B, self.half_dim, H, W)
        x1 = F.silu(self.conv1(x1))
        x1 = x1.reshape(B, self.half_dim, T).permute(0, 2, 1)
        x2 = self.linear2(x)
        x2 = x2.permute(0, 2, 1).reshape(B, self.half_dim, H, W)
        x2 = F.silu(self.conv2(x2))
        x2 = x2.reshape(B, self.half_dim, T).permute(0, 2, 1)
        x2 = self.mdssm(x2, H, W)
        return self.merge(torch.cat([x1, x2], dim=-1))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, H=None, W=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q * scale, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn, v)
        x = attn_out.transpose(1, 2).reshape(B, T, D)
        return self.proj(x)


class CAB(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        scale = x.mean(dim=[2, 3])
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(scale))))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class MambaBlock(nn.Module):
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
        B, T, D = x.shape
        x_hat = self.mamba_mixer(self.norm1(x), H, W) + self.s1 * x
        z = self.norm2(x_hat)
        z = z.permute(0, 2, 1).reshape(B, D, H, W)
        z = self.cab(self.conv(z))
        z = z.reshape(B, D, T).permute(0, 2, 1)
        return z + self.s2 * x_hat


class AttentionBlock(nn.Module):
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
        B, T, D = x.shape
        x_hat = self.attention(self.norm1(x)) + self.s1 * x
        z = self.norm2(x_hat)
        z = z.permute(0, 2, 1).reshape(B, D, H, W)
        z = self.cab(self.conv(z))
        z = z.reshape(B, D, T).permute(0, 2, 1)
        return z + self.s2 * x_hat


class MambaLayer(nn.Module):
    def __init__(self, embed_dim, num_blocks, ssm_state_dim=16, cab_reduction=8):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(embed_dim, ssm_state_dim, cab_reduction) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x, H, W):
        B, T, D = x.shape
        residual = x
        for block in self.blocks:
            x = block(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)
        x = self.conv(x)
        x = x.reshape(B, D, T).permute(0, 2, 1)
        return x + residual


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, cab_reduction=8):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, num_heads, cab_reduction) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x, H, W):
        B, T, D = x.shape
        residual = x
        for block in self.blocks:
            x = block(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)
        x = self.conv(x)
        x = x.reshape(B, D, T).permute(0, 2, 1)
        return x + residual


class FFM(nn.Module):
    def __init__(self, embed_dim, num_sources):
        super().__init__()
        total_dim = num_sources * embed_dim
        self.dw_conv = nn.Conv2d(total_dim, total_dim, 3, 1, 1, groups=total_dim)
        self.act = nn.SiLU()
        self.pw_conv = nn.Conv2d(total_dim, embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feature_list, H, W):
        B = feature_list[0].shape[0]
        x = torch.cat(feature_list, dim=-1)
        total_dim = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, total_dim, H, W)
        x = self.act(self.dw_conv(x))
        x = self.pw_conv(x)
        x = x.reshape(B, -1, H * W).permute(0, 2, 1)
        return self.norm(x)


class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim, out_channels, scale):
        super().__init__()
        layers = []
        if scale == 2:
            layers += [nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(embed_dim, embed_dim * 9, 3, 1, 1), nn.PixelShuffle(3)]
        elif scale == 4:
            layers += [nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2),
                       nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1), nn.PixelShuffle(2)]
        layers.append(nn.Conv2d(embed_dim, out_channels, 3, 1, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class SRMambaT(nn.Module):
    def __init__(self, in_channels, embed_dim, scale, num_heads, blocks_per_layer, ssm_state_dim, cab_reduction):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.shallow_conv = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.mamba_enc1 = MambaLayer(embed_dim, blocks_per_layer, ssm_state_dim, cab_reduction)
        self.mamba_enc2 = MambaLayer(embed_dim, blocks_per_layer, ssm_state_dim, cab_reduction)
        self.trans_enc = TransformerLayer(embed_dim, blocks_per_layer, num_heads, cab_reduction)
        self.ffm = FFM(embed_dim, num_sources=4)
        self.trans_dec = TransformerLayer(embed_dim, blocks_per_layer, num_heads, cab_reduction)
        self.reconstruction = ReconstructionHead(embed_dim, in_channels, scale)

    def forward(self, x):
        B, C, H, W = x.shape
        f_s = self.shallow_conv(x)
        f_t = f_s.flatten(2).transpose(1, 2)
        f_enc1 = self.mamba_enc1(f_t, H, W)
        f_enc2 = self.mamba_enc2(f_enc1, H, W)
        f_enc3 = self.trans_enc(f_enc2, H, W)
        f_s_tokens = f_s.flatten(2).transpose(1, 2)
        f_fused = self.ffm([f_s_tokens, f_enc1, f_enc2, f_enc3], H, W)
        f_d = self.trans_dec(f_fused, H, W)
        f_d_2d = f_d.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        f = f_s + f_d_2d
        return self.reconstruction(f)

print("Model architecture ready (exact match with training)")


# %%
# CELL 6: METRICS
def compute_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(1.0 / mse).item() if mse > 0 else float('inf')

def compute_ssim(sr, hr):
    C1, C2 = 0.01**2, 0.03**2
    mu_s, mu_h = F.avg_pool2d(sr, 3, 1, 1), F.avg_pool2d(hr, 3, 1, 1)
    sig_s = F.avg_pool2d(sr**2, 3, 1, 1) - mu_s**2
    sig_h = F.avg_pool2d(hr**2, 3, 1, 1) - mu_h**2
    sig_sh = F.avg_pool2d(sr*hr, 3, 1, 1) - mu_s*mu_h
    return ((2*mu_s*mu_h+C1)*(2*sig_sh+C2)/((mu_s**2+mu_h**2+C1)*(sig_s+sig_h+C2))).mean().item()

print("Metrics ready")


# %%
# CELL 7: TEST ALL 3 PAVIAC MODELS
IN_CHANNELS = 102
EMBED_DIM = 48
NUM_HEADS = 6
BLOCKS_PER_LAYER = 4
SSM_STATE_DIM = 16
CAB_REDUCTION = 8
SCALES = [2, 3, 4]
RESULTS = {}

for scale in SCALES:
    print(f"\n{'='*60}")
    print(f"  Testing PaviaC x{scale}")
    print(f"{'='*60}")

    model_path = os.path.join(MODEL_DIR, f'srmamba_t_x{scale}_best.pth')
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        continue

    model = SRMambaT(IN_CHANNELS, EMBED_DIM, scale, NUM_HEADS, BLOCKS_PER_LAYER, SSM_STATE_DIM, CAB_REDUCTION).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')})")

    dataset = HyperspectralSRDataset(DATA_DIR, scale)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"  Patches: {len(dataset)}")

    all_psnr, all_ssim = [], []
    with torch.no_grad():
        for lr_b, hr_b in loader:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            sr_b = model(lr_b)
            if sr_b.shape[-2:] != hr_b.shape[-2:]:
                sr_b = F.interpolate(sr_b, size=hr_b.shape[-2:], mode='bilinear', align_corners=False)
            sr_b = sr_b.clamp(0, 1)
            for i in range(sr_b.shape[0]):
                all_psnr.append(compute_psnr(sr_b[i:i+1], hr_b[i:i+1]))
                all_ssim.append(compute_ssim(sr_b[i:i+1], hr_b[i:i+1]))

    avg_psnr, avg_ssim = np.mean(all_psnr), np.mean(all_ssim)
    RESULTS[f'x{scale}'] = {'PSNR': avg_psnr, 'SSIM': avg_ssim}
    print(f"  PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

    # Visualize 5 samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), 5, replace=False)
    rgb_bands = [60, 30, 10]
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    for row, idx in enumerate(indices):
        lr, hr = dataset[idx]
        with torch.no_grad():
            sr = model(lr.unsqueeze(0).to(device))
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        sr = sr.clamp(0, 1).cpu()
        p = compute_psnr(sr[0:1], hr.unsqueeze(0))
        s = compute_ssim(sr[0:1], hr.unsqueeze(0))

        def to_rgb(t, bands):
            img = t[bands].permute(1, 2, 0).numpy()
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        lr_up = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode='nearest')[0]
        axes[row, 0].imshow(to_rgb(lr_up, rgb_bands)); axes[row, 0].set_title(f'LR ({lr.shape[1]}x{lr.shape[2]})', fontsize=12); axes[row, 0].axis('off')
        axes[row, 1].imshow(to_rgb(sr[0], rgb_bands)); axes[row, 1].set_title(f'SR x{scale} PSNR:{p:.1f} SSIM:{s:.3f}', fontsize=11); axes[row, 1].axis('off')
        axes[row, 2].imshow(to_rgb(hr, rgb_bands)); axes[row, 2].set_title(f'HR ({hr.shape[1]}x{hr.shape[2]})', fontsize=12); axes[row, 2].axis('off')

    plt.suptitle(f'SRMamba-T x{scale} | PaviaC | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/content/PaviaC_x{scale}_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: PaviaC_x{scale}_results.png")
    del model; torch.cuda.empty_cache(); gc.collect()


# %%
# CELL 8: RESULTS
print("\n" + "="*50)
print("  PAVIAC RESULTS")
print("="*50)
print(f"{'Scale':<8} {'PSNR':<12} {'SSIM':<10}")
print("-"*30)
for s, m in RESULTS.items():
    print(f"{s:<8} {m['PSNR']:<12.2f} {m['SSIM']:<10.4f}")
print("\nDone! Download images from Files panel (left sidebar)")
