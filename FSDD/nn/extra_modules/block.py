# SAR Ship Detection Framework with Frequency-Spatial Dual-Domain Enhancement
# Paper-faithful implementation of FSEM, MSFE and SOEP modules.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..modules.conv import Conv
from ..modules.block import C2f
from ..modules.head import RTDETRDecoder


class EdgeEnhancer(nn.Module):
    """Fixed-Scharr edge magnitude branch (Sec. FSEM, Spatial Feature Extraction).

    E = sqrt(Gx^2 + Gy^2 + eps), with the paper's kernel convention:
    Kx = [[-3,0,3],[-10,0,10],[-3,0,3]], Ky = [[-3,-10,-3],[0,0,0],[3,10,3]].
    Depthwise cross-correlation, unit stride, one-pixel zero padding.
    """

    def __init__(self, in_channels):
        super().__init__()
        kernel_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
        ).view(1, 1, 3, 3).expand(in_channels, 1, 3, 3).contiguous()
        kernel_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32
        ).view(1, 1, 3, 3).expand(in_channels, 1, 3, 3).contiguous()

        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                groups=in_channels, bias=False)
        with torch.no_grad():
            self.conv_x.weight.copy_(kernel_x)
            self.conv_y.weight.copy_(kernel_y)
        self.conv_x.weight.requires_grad_(False)
        self.conv_y.weight.requires_grad_(False)
        self.eps = 1e-6

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx * gx + gy * gy + self.eps)


class FSEM(nn.Module):
    """Frequency-Spatial Enhancement Module (Sec. FSEM).

    Spatial branch:  S = f_conv2( f_conv1(E) + X )
    Frequency branch: Xf = sqrt( Re(IFFT(F'_r + jF'_i))^2 + Im(...)^2 + eps )
                      F_out = f_conv(Xf)
    Fusion:          Y = f_final( S + F_out )   (1x1 conv)
    """

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.c1 = c1
        self.c2 = c2
        self.edge_enhancer = EdgeEnhancer(c1)
        self.spatial_proj1 = Conv(c1, c1, 3, 1)      # f_conv1
        self.spatial_proj2 = Conv(c1, c1, 3, 1)      # f_conv2
        self.freq_transform = Conv(c1 * 2, c1 * 2, 3, 1)  # f_fftconv
        self.freq_proj = Conv(c1, c1, 3, 1)          # f_conv
        self.fusion = Conv(c1, c2, 1, 1)             # f_final (1x1)
        self.eps = 1e-6

    def forward(self, x):
        c = self.c1
        # --- spatial branch ---
        e = self.edge_enhancer(x)
        s = self.spatial_proj2(self.spatial_proj1(e) + x)

        # --- frequency branch ---
        x_hat = torch.fft.fft2(x, norm='ortho')          # complex [B, c, H, W]
        x_r = torch.real(x_hat)
        x_i = torch.imag(x_hat)
        f = torch.cat((x_r, x_i), dim=1)                  # [B, 2c, H, W]
        f_p = self.freq_transform(f)                     # [B, 2c, H, W]
        f_pr, f_pi = torch.split(f_p, [c, c], dim=1)
        f_hat = torch.complex(f_pr, f_pi)
        x_bar = torch.fft.ifft2(f_hat, norm='ortho')      # complex
        x_f = torch.sqrt(torch.real(x_bar) ** 2 + torch.imag(x_bar) ** 2 + self.eps)
        f_out = self.freq_proj(x_f)

        # --- dual-domain fusion ---
        return self.fusion(s + f_out)


class DynamicTanhNorm(nn.Module):
    """Standardized Dynamic-Tanh (Sec. MSFE, DyT).

    x_hat = (x - mu_c) / sqrt(Var_c(x) + eps)
    DyT(x) = tanh(gamma * x_hat + beta)
    mu_c, Var_c computed over the channel dimension at each spatial location.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):                       # x: [B, N, C]
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)
        return torch.tanh(self.gamma * x_hat + self.beta)


class ScatteringSaliency(nn.Module):
    """Psi_sca(Z) = Flatten( sigmoid( DWConv3x3( Reshape(Z) ) ) )."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, H, W):                # z: [B, N, C]
        B, N, C = z.shape
        x = z.transpose(1, 2).reshape(B, C, H, W)
        x = self.sigmoid(self.dwconv(x))
        return x.flatten(2).transpose(1, 2)    # [B, N, C]


class PositionalEmbedding(nn.Module):
    """Fixed 2D sin-cos positional embedding (Sec. MSFE).

    p(x,y) = [sin(x*w_k), cos(x*w_k), sin(y*w_k), cos(y*w_k)]_{k=0}^{d-1},
    d = C/4, w_k = 1 / temp^{k/d}.  Returns [1, N, C].
    """

    def __init__(self, temperature=10000.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, H, W, C, device, dtype):
        d = C // 4
        omega = 1.0 / (self.temperature ** (torch.arange(d, device=device, dtype=dtype) / d))
        y_pos = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)   # [H,1]
        x_pos = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)  # [W,1]
        y_emb = torch.cat([torch.sin(y_pos * omega), torch.cos(y_pos * omega)], dim=-1)  # [H, 2d]
        x_emb = torch.cat([torch.sin(x_pos * omega), torch.cos(x_pos * omega)], dim=-1)  # [W, 2d]
        # p(x,y) interleaves x and y parts -> [H, W, C]
        pos = torch.cat([
            x_emb.unsqueeze(0).expand(H, W, 2 * d),
            y_emb.unsqueeze(1).expand(H, W, 2 * d),
        ], dim=-1)                                          # [H, W, 4d] == C
        return pos.reshape(H * W, C).unsqueeze(0)            # [1, N, C]


class ScatteringAwarePolarizedAttention(nn.Module):
    """SAPA (Sec. MSFE).

    Q = Psi_sca(Z) * (Z Wq + P),  K = Psi_sca(Z) * (Z Wk + P),  V = Z Wv
    SAPA(Q,K,V) = phi(Q)(phi(K)^T V) / ( phi(Q)(phi(K)^T 1_N) + eps )
    with phi(x) = ReLU(x)^2.  Efficient O(N) form.
    """

    def __init__(self, dim, heads=8, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        self.psi_sca = ScatteringSaliency(dim)
        self.pos_emb = PositionalEmbedding()

    def forward(self, z, H, W):                # z: [B, N, C]
        B, N, C = z.shape
        P = self.pos_emb(H, W, C, z.device, z.dtype)
        psi = self.psi_sca(z, H, W)            # [B, N, C]

        q = psi * (self.W_q(z) + P)
        k = psi * (self.W_k(z) + P)
        v = self.W_v(z)

        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)   # [B, h, N, d]
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)   # [B, h, N, d]
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        phi_q = torch.relu(q) ** 2
        phi_k = torch.relu(k) ** 2

        # KV = phi(K)^T V  -> [B, h, d, d]
        kv = torch.einsum('bhnd,bhnv->bhdv', phi_k, v)
        # phi(K)^T 1_N = sum_n phi_k -> [B, h, d]
        kv_denom = phi_k.sum(dim=2)

        num = torch.einsum('bhnd,bhdv->bhnv', phi_q, kv)             # [B, h, N, d]
        denom = torch.einsum('bhnd,bhd->bhn', phi_q, kv_denom)       # [B, h, N]
        out = num / (denom.unsqueeze(-1) + self.eps)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.W_out(out)


class EdgeEnhancedDecomposedFFN(nn.Module):
    """EDFFN (Sec. MSFE): F(U) = delta( DWConv(U W_e) ) W_p.

    Residual is applied ONCE in the MSFE forward pass, not here.
    """

    def __init__(self, dim, hidden=None, expansion_ratio=4):
        super().__init__()
        hidden = hidden if hidden is not None else dim * expansion_ratio
        self.W_e = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.W_p = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, u, H, W):                # u: [B, N, C]
        B, N, C = u.shape
        hidden = self.W_e(u)                                          # [B, N, rC]
        x = hidden.transpose(1, 2).reshape(B, -1, H, W)             # [B, rC, H, W]
        x = self.act(self.dwconv(x))                                # GELU(DWConv(...))
        x = x.flatten(2).transpose(1, 2)                           # [B, N, rC]
        return self.W_p(x)                                          # [B, N, C]


class MonaReweighting(nn.Module):
    """Mona-inspired channel gate (Sec. MSFE).

    g = GAP(U);  h(U) = W2 delta(W1 g);  MonaGate(U) = U * sigma(h(U))
    W1: C -> C/r_m, W2: C/r_m -> C, r_m = 4.
    Operates on token layout [B, N, C] (GAP over N).
    """

    def __init__(self, dim, r_m=4):
        super().__init__()
        hidden = max(dim // r_m, 1)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, u):                      # u: [B, N, C]
        g = u.mean(dim=1, keepdim=True)        # GAP -> [B, 1, C]
        h = self.fc2(self.act(self.fc1(g)))     # [B, 1, C]
        gate = self.sigmoid(h)
        return u * gate


class MSFE(nn.Module):
    """Multi-Stage Feature Enhancement (Sec. MSFE), AIFI-compatible I/O [B,C,H,W].

    U1 = MonaGate1( DyT( X + SAPA(X) ) )
    U2 = U1 + EDFFN(U1)
    Y  = MonaGate2( DyT(U2) )
    """

    def __init__(self, c1, cm=None, num_heads=8, **kwargs):
        super().__init__()
        cm = cm if cm is not None else c1 * 4
        self.dim = c1
        self.heads = num_heads
        self.sapa = ScatteringAwarePolarizedAttention(c1, num_heads)
        self.dyt = DynamicTanhNorm(c1)
        self.mona1 = MonaReweighting(c1)
        self.edffn = EdgeEnhancedDecomposedFFN(c1, cm)
        self.mona2 = MonaReweighting(c1)

    def forward(self, x):                      # x: [B, C, H, W]
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)       # [B, N, C]
        u1 = self.mona1(self.dyt(z + self.sapa(z, H, W)))
        u2 = u1 + self.edffn(u1, H, W)
        y = self.mona2(self.dyt(u2))           # [B, N, C]
        return y.transpose(1, 2).reshape(B, C, H, W).contiguous()


class SPDConv(nn.Module):
    """Space-to-Depth Convolution (Sec. SOEP).

    S2(P) = Concat of 4 spatial offsets -> [4C, H/2, W/2];
    Conv1x1(4C -> C3).
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv(in_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)            # [B, 4C, H/2, W/2]
        return self.conv(x)


class OmniKernel(nn.Module):
    """Global-Large-Local representation with DDFA (Sec. SOEP, OmniKernel).

    Y = X + phi_1x1(X) + phi_1xk(X) + phi_kx1(X) + phi_kxk(X) + X_fgm
    where X_fgm = DDFA( SCA(X) ),  SCA(X) = sigma(Ws GAP(X)) * X,
    DDFA = FCA + FGM (frequency-channel attention + frequency gating).
    """

    def __init__(self, dim, k=31, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        pad = k // 2
        # SCA
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ws = nn.Conv2d(dim, dim, 1, 1)
        self.sigmoid = nn.Sigmoid()
        # FCA: learnable per-channel complex frequency response Wf
        self.wf_real = nn.Parameter(torch.zeros(dim))
        self.wf_imag = nn.Parameter(torch.zeros(dim))
        # FGM
        self.psi1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.psi2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        # spatial depthwise branches
        self.dw_1x1 = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.dw_1xk = nn.Conv2d(dim, dim, kernel_size=(1, k), padding=(0, pad), groups=dim)
        self.dw_kx1 = nn.Conv2d(dim, dim, kernel_size=(k, 1), padding=(pad, 0), groups=dim)
        self.dw_kxk = nn.Conv2d(dim, dim, kernel_size=k, padding=pad, groups=dim)

    def forward(self, x):                      # x: [B, C, H, W]
        # SCA
        sca = self.sigmoid(self.ws(self.gap(x))) * x
        # FCA
        x_hat = torch.fft.fft2(sca, norm='ortho')
        wf = torch.complex(self.wf_real, self.wf_imag).view(1, -1, 1, 1)
        x_fca = torch.real(torch.fft.ifft2(x_hat * wf, norm='ortho'))
        # FGM
        x1 = self.psi1(x_fca)
        x2 = self.psi2(x_fca)
        spec = torch.fft.fft2(x1, norm='ortho') * torch.fft.fft2(x2, norm='ortho')
        x_fgm = self.alpha * torch.abs(torch.fft.ifft2(spec, norm='ortho')) + self.beta * x_fca
        # spatial branches + residual
        y = (x + self.dw_1x1(x) + self.dw_1xk(x) +
             self.dw_kx1(x) + self.dw_kxk(x) + x_fgm)
        return y


class CSP_FeatureFusion(nn.Module):
    """SOEP final feature fusion (Sec. SOEP).

    Split(C3/2, C3/2); P3_out = Conv1x1( Concat[ OmniKernel(half), identity ] ).
    """

    def __init__(self, dim, e=0.5):
        super().__init__()
        self.split_ratio = e
        c = int(dim * e)
        self.input_proj = Conv(dim, dim, 1, 1)
        self.output_proj = Conv(dim, dim, 1, 1)
        self.fusion_module = OmniKernel(c)

    def forward(self, x):
        projected = self.input_proj(x)
        c = int(projected.size(1) * self.split_ratio)
        fusion_branch, identity_branch = torch.split(
            projected, [c, projected.size(1) - c], dim=1)
        fused = self.fusion_module(fusion_branch)
        return self.output_proj(torch.cat((fused, identity_branch), dim=1))


# ---- aliases required by the model configs ----
MultiScaleFeatureFusion = OmniKernel
DyT = DynamicTanhNorm
SAPA = ScatteringAwarePolarizedAttention
EDFFN = EdgeEnhancedDecomposedFFN
Mona = MonaReweighting
CSPOmniKernel = CSP_FeatureFusion
Decoder = RTDETRDecoder
