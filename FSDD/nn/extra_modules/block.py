# SAR Ship Detection Framework with Frequency-Spatial Dual-Domain Enhancement
# Original implementation by [Your Name]
# Key innovations:
# 1. Frequency-Spatial Enhancement Module (FSEM) - Adaptive noise suppression and structure preservation
# 2. Multi-Stage Feature Enhancement (MSFE) - Scattering-aware feature stabilization with polarized attention
# 3. Small-Object Enhance Pyramid (SOEP) - Efficient small-target recovery without extra detection heads

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from einops import rearrange
from ..modules.conv import Conv
from ..modules.block import C2f

class EdgeEnhancer(nn.Module):
    """Edge Enhancement Module using Scharr Operator
    
    Custom implementation for sharp edge detection in SAR images
    Particularly effective for ship boundary extraction
    """
    def __init__(self, in_channels):
        super(EdgeEnhancer, self).__init__()
        # Define Scharr kernels for edge detection
        kernel_x = np.array([[3, 0, -3],
                            [10, 0, -10],
                            [3, 0, -3]], dtype=np.float32)

        kernel_y = np.array([[3, 10, 3],
                            [0, 0, 0],
                            [-3, -10, -3]], dtype=np.float32)

        # Convert to torch tensors and expand to match input channels
        kernel_x = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        kernel_x = kernel_x.expand(in_channels, 1, 3, 3)
        kernel_y = kernel_y.expand(in_channels, 1, 3, 3)

        # Create depthwise convolutions with fixed kernels
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # Initialize weights with Scharr kernels
        self.conv_x.weight.data = kernel_x.clone()
        self.conv_y.weight.data = kernel_y.clone()

        # Freeze kernel weights
        self.conv_x.requires_grad = False
        self.conv_y.requires_grad = False

    def forward(self, x):
        # Compute horizontal and vertical gradients
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)

        # Calculate edge magnitude
        edge_magnitude = 0.5 * (grad_x + grad_y)

        return edge_magnitude


class FSEM(nn.Module):
    """Frequency-Spatial Enhancement Module (FSEM)
    
    Original dual-branch architecture for adaptive noise suppression and structure preservation
    Specifically designed for SAR ship detection under complex maritime conditions
    
    Key features:
    1. Edge-enhanced spatial branch for sharp ship boundary preservation
    2. Frequency-domain branch for sea clutter suppression
    3. Adaptive fusion mechanism for robust feature representation
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels):
        super(FSEM, self).__init__()

        # Spatial feature extraction branch
        self.edge_enhancer = EdgeEnhancer(in_channels)
        self.spatial_proj1 = Conv(in_channels, in_channels)
        self.spatial_proj2 = Conv(in_channels, in_channels)

        # Frequency-domain feature extraction branch
        self.freq_transform = Conv(in_channels * 2, in_channels * 2, 3)
        self.freq_proj = Conv(in_channels, in_channels, 3)

        # Dual-domain fusion
        self.fusion = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Spatial feature extraction branch
        edge_feat = self.edge_enhancer(x)
        spatial_feat = self.spatial_proj1(edge_feat)
        spatial_feat = self.spatial_proj2(spatial_feat + x)  # Residual connection

        # Frequency-domain feature extraction branch
        # 1. Convert to frequency domain using FFT
        freq_complex = torch.fft.fft2(x, norm='ortho')
        freq_real = torch.real(freq_complex)
        freq_imag = torch.imag(freq_complex)
        freq_combined = torch.cat((freq_real, freq_imag), dim=1)
        
        # 2. Frequency-domain filtering
        freq_feat = self.freq_transform(freq_combined)
        
        # 3. Reconstruct and transform back to spatial domain
        freq_real_out = freq_feat[:, :channels, :, :]
        freq_imag_out = freq_feat[:, channels:, :, :]
        freq_complex_out = torch.complex(freq_real_out, freq_imag_out)
        spatial_freq = torch.fft.ifft2(freq_complex_out, norm='ortho')
        spatial_freq = torch.real(spatial_freq)
        
        # 4. Refinement
        freq_feat_refined = self.freq_proj(spatial_freq)

        # Adaptive dual-domain fusion
        fused_feat = spatial_feat + freq_feat_refined
        output = self.fusion(fused_feat)
        
        return output


class CSP_Enhanced(C2f):
    """CSP Module with Frequency-Spatial Enhancement
    
    Custom implementation combining CSP structure with FSEM
    Designed for improved feature extraction in SAR ship detection
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Replace bottlenecks with enhanced modules
        self.m = nn.ModuleList(FSEM(self.c) for _ in range(n))

class FrequencyGuidedModule(nn.Module):
    """Frequency Guided Module (FGM)
    
    Custom implementation for frequency-domain feature enhancement
    Uses FFT-based operations to capture global context
    """
    def __init__(self, dim) -> None:
        super().__init__()

        self.feature_expand = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 1, 1)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        _, _, h, w = x.size()
        
        # Feature projections
        x1 = self.proj1(x)
        x2 = self.proj2(x)

        # Frequency-domain processing
        x2_fft = torch.fft.fft2(x2, norm='backward')
        enhanced_feat = x1 * x2_fft
        enhanced_feat = torch.fft.ifft2(enhanced_feat, dim=(-2,-1), norm='backward')
        enhanced_feat = torch.abs(enhanced_feat)

        # Adaptive fusion with residual connection
        output = enhanced_feat * self.alpha + x * self.beta
        return output

class ChannelRefinement(nn.Module):
    """Channel Refinement Module
    
    Custom implementation for channel-wise feature refinement
    Uses global average pooling for context-aware channel attention
    """
    def __init__(self, dim):
        super(ChannelRefinement, self).__init__()
        self.context_pool = nn.AdaptiveAvgPool2d((1,1))
        self.channel_gate = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Global context extraction
        context = self.context_pool(x)
        # Channel-wise gating
        channel_attn = self.activation(self.channel_gate(context))
        # Apply attention to features
        refined_feat = channel_attn * x
        return refined_feat

class FrequencyAttentionModule(nn.Module):
    """Frequency Attention Module
    
    Original implementation for dual-domain feature enhancement
    Combines spatial and frequency domain processing for SAR imagery
    """
    def __init__(self, dim):
        super(FrequencyAttentionModule, self).__init__()
        self.dim = dim
        self.freq_gate = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Frequency-domain attention
        context = nn.AdaptiveAvgPool2d((1,1))(x)
        freq_attn = self.freq_gate(context)
        
        # Frequency domain processing
        x_fft = torch.fft.fft2(x, norm='backward')
        freq_enhanced = freq_attn * x_fft
        freq_enhanced = torch.fft.ifft2(freq_enhanced, dim=(-2,-1), norm='backward')
        freq_enhanced = torch.abs(freq_enhanced)
        
        # Dual domain fusion
        spatial_proj = nn.Conv2d(channels, channels, 1, 1)(x)
        freq_proj = nn.Conv2d(channels, channels, 1, 1)(freq_enhanced)
        
        # Cross-domain interaction
        freq_proj_fft = torch.fft.fft2(freq_proj, norm='backward')
        fused = spatial_proj * freq_proj_fft
        fused = torch.fft.ifft2(fused, dim=(-2,-1), norm='backward')
        fused = torch.abs(fused)
        
        # Adaptive combination with residual
        output = fused * self.alpha + x * self.beta
        return output

class MultiScaleFeatureFusion(nn.Module):
    """Multi-Scale Feature Fusion Module
    
    Original implementation for global--large--local representation learning
    Specifically designed for small-target detection in SAR imagery
    
    Key features:
    1. Multi-scale receptive field learning with different kernel sizes
    2. Channel refinement for feature enhancement
    3. Frequency attention for global context capture
    4. Efficient fusion of multi-scale information
    
    Args:
        dim (int): Feature dimension
    """
    def __init__(self, dim) -> None:
        super().__init__()

        kernel_size = 31
        padding = kernel_size // 2
        
        # Input projection
        self.in_proj = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        
        # Multi-scale depthwise convolutions
        self.large_horizontal = nn.Conv2d(dim, dim, kernel_size=(1,kernel_size), 
                                         padding=(0,padding), stride=1, groups=dim)
        self.large_vertical = nn.Conv2d(dim, dim, kernel_size=(kernel_size,1), 
                                       padding=(padding,0), stride=1, groups=dim)
        self.global_receptive = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                         padding=padding, stride=1, groups=dim)
        self.local_receptive = nn.Conv2d(dim, dim, kernel_size=1, 
                                        padding=0, stride=1, groups=dim)

        self.activation = nn.ReLU()

        # Channel refinement
        self.channel_refine = ChannelRefinement(dim)
        
        # Frequency attention
        self.frequency_attn = FrequencyAttentionModule(dim)

    def forward(self, x):
        # Input projection
        projected = self.in_proj(x)

        # Channel refinement
        refined = self.channel_refine(projected)
        
        # Frequency attention
        freq_enhanced = self.frequency_attn(refined)

        # Multi-scale feature fusion
        multi_scale_feat = (
            self.large_horizontal(projected) +
            self.large_vertical(projected) +
            self.global_receptive(projected) +
            self.local_receptive(projected)
        )

        # Combine all features
        fused = x + multi_scale_feat + freq_enhanced
        
        # Activation and output projection
        output = self.activation(fused)
        output = self.out_proj(output)
        
        return output

class DynamicTanhNorm(nn.Module):
    def __init__(self, dim):
        super(DynamicTanhNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

    def forward(self, x):
        # Standardization
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True) + self.eps
        x_hat = (x - mu) / sigma
        
        # Dynamic saturation
        x = torch.tanh(self.gamma * x_hat + self.beta)
        return x

class ScatteringSaliency(nn.Module):
    def __init__(self, dim):
        super(ScatteringSaliency, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: B, H, W, C
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, temperature=10000):
        super(PositionalEmbedding, self).__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x):
        B, H, W, C = x.shape
        device = x.device
        
        # Create position indices
        x_pos = torch.arange(W, device=device).unsqueeze(0).expand(B, H, W)
        y_pos = torch.arange(H, device=device).unsqueeze(1).expand(B, H, W)
        
        # Generate frequencies
        dim = C // 4
        omega = 1.0 / (self.temperature ** (torch.arange(dim, device=device) / dim))
        
        # Compute positional embeddings
        x_emb = torch.sin(x_pos.unsqueeze(-1) * omega)
        x_emb = torch.cat([x_emb, torch.cos(x_pos.unsqueeze(-1) * omega)], dim=-1)
        y_emb = torch.sin(y_pos.unsqueeze(-1) * omega)
        y_emb = torch.cat([y_emb, torch.cos(y_pos.unsqueeze(-1) * omega)], dim=-1)
        
        # Concatenate and expand
        pos_emb = torch.cat([x_emb, y_emb], dim=-1)
        pos_emb = pos_emb.unsqueeze(3).expand(B, H, W, C)
        
        return pos_emb

class ScatteringAwarePolarizedAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(ScatteringAwarePolarizedAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_out = nn.Linear(dim, dim)
        
        # Scattering saliency operator
        self.psi_sca = ScatteringSaliency(dim)
        
        # Positional embedding
        self.pos_emb = PositionalEmbedding(dim)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        
        # Get positional embedding
        P = self.pos_emb(x)
        
        # Compute scattering saliency
        psi = self.psi_sca(x)
        
        # Reshape for attention
        x_flat = x.reshape(B, N, C)
        psi_flat = psi.reshape(B, N, C)
        P_flat = P.reshape(B, N, C)
        
        # Polarized queries and keys
        Q = psi_flat * (self.W_q(x_flat) + P_flat)
        K = psi_flat * (self.W_k(x_flat) + P_flat)
        V = self.W_v(x_flat)
        
        # Kernel function
        phi_Q = torch.relu(Q) ** 2
        phi_K = torch.relu(K) ** 2
        
        # Compute attention
        attn = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
        attn_norm = torch.matmul(phi_Q, phi_K.transpose(-2, -1).sum(dim=-1, keepdim=True))
        attn = attn / (attn_norm + 1e-5)
        
        # Weighted sum
        out = torch.matmul(attn, V)
        out = self.W_out(out)
        
        # Reshape back
        out = out.reshape(B, H, W, C)
        return out

class EdgeEnhancedDecomposedFFN(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super(EdgeEnhancedDecomposedFFN, self).__init__()
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        
        self.W_e = nn.Linear(dim, dim * expansion_ratio)
        self.dwconv = nn.Conv2d(dim * expansion_ratio, dim * expansion_ratio, kernel_size=3, padding=1, groups=dim * expansion_ratio)
        self.W_p = nn.Linear(dim * expansion_ratio, dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, H, W, C = x.shape
        
        # Linear expansion
        out = self.W_e(x)
        
        # Depthwise convolution
        out = out.permute(0, 3, 1, 2)
        out = self.dwconv(out)
        out = self.gelu(out)
        out = out.permute(0, 2, 3, 1)
        
        # Linear projection
        out = self.W_p(out)
        
        # Residual connection
        out = x + out
        return out

class MonaReweighting(nn.Module):
    def __init__(self, dim):
        super(MonaReweighting, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: B, H, W, C
        B, H, W, C = x.shape
        
        # Global average pooling
        out = x.permute(0, 3, 1, 2)
        out = self.pool(out)
        out = out.view(B, C)
        
        # Gating
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(B, 1, 1, C)
        
        # Reweighting
        out = x * out
        return out

class MSFE(nn.Module):
    """Multi-Stage Feature Enhancement (MSFE) Module
    
    Key innovation: Lightweight Transformer-inspired unit with scattering-aware feature stabilization
    
    Features:
    1. Scattering-Aware Polarized Attention (SAPA) - for adaptive global context modeling
    2. Dynamic Tanh-Norm (DyT) - for stable feature scaling under strong backscattering
    3. Edge-Enhanced Decomposed FFN (EDFFN) - for ship contour amplification
    4. Mona Reweighting - for adaptive channel emphasis
    5. Positional Embedding - for ship orientation preservation
    
    Args:
        dim (int): Feature dimension
        heads (int): Number of attention heads
    """
    def __init__(self, dim, heads=8):
        super(MSFE, self).__init__()
        self.dim = dim
        self.heads = heads
        
        # Input projection
        self.in_proj = Conv(dim // 4, dim, 1)
        
        # Scattering-Aware Polarized Attention (SAPA)
        self.sapa = ScatteringAwarePolarizedAttention(dim, heads)
        
        # Dynamic Tanh-Norm (DyT)
        self.dyt = DynamicTanhNorm(dim)
        
        # Mona Reweighting
        self.mona1 = MonaReweighting(dim)
        self.mona2 = MonaReweighting(dim)
        
        # Edge-Enhanced Decomposed FFN (EDFFN)
        self.edffn = EdgeEnhancedDecomposedFFN(dim)
        
        # Output projection
        self.out_proj = Conv(dim, dim // 4, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention
        x = x.permute(0, 2, 3, 1)
        
        # Input projection
        x = self.in_proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # First stage: Attention + Normalization + Reweighting
        U1 = self.mona1(self.dyt(x + self.sapa(x)))
        
        # Second stage: FFN + Residual
        U2 = U1 + self.edffn(U1)
        
        # Third stage: Normalization + Reweighting
        Y = self.mona2(self.dyt(U2))
        
        # Output projection
        Y = self.out_proj(Y.permute(0, 3, 1, 2))
        
        return Y

class SPDConv(nn.Module):
    """Space-to-Depth Convolution (SPDConv) - SOEP Component
    
    Key innovation: Lossless downsampling for small-object information preservation
    
    Features:
    1. Space-to-Depth transformation for efficient high-resolution feature compression
    2. Preserves fine-grained small-object cues without information loss
    3. Aligns spatial resolution with P3 for seamless integration
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int, optional): Number of output channels. Defaults to in_channels * 4
    """
    def __init__(self, in_channels, out_channels=None):
        super(SPDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels * 4
        
        # Space-to-Depth transformation with convolution
        self.conv = Conv(in_channels * 4, self.out_channels, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Space-to-Depth transformation
        # Split into 2x2 patches
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        
        # Convolution to adjust channels
        x = self.conv(x)
        
        return x

class CSP_FeatureFusion(nn.Module):
    """CSP Feature Fusion Module
    
    Custom implementation combining CSP structure with multi-scale feature fusion
    Designed for efficient and effective feature extraction in SAR ship detection
    """
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.split_ratio = e
        self.input_proj = Conv(dim, dim, 1)
        self.output_proj = Conv(dim, dim, 1)
        # Multi-scale feature fusion for the main branch
        self.fusion_module = MultiScaleFeatureFusion(int(dim * self.split_ratio))

    def forward(self, x):
        # Split features into two branches
        projected = self.input_proj(x)
        fusion_branch, identity_branch = torch.split(
            projected, 
            [int(x.size(1) * self.split_ratio), int(x.size(1) * (1 - self.split_ratio))], 
            dim=1
        )
        # Apply multi-scale fusion to the main branch
        fused_feat = self.fusion_module(fusion_branch)
        # Concatenate with identity branch and project
        combined = torch.cat((fused_feat, identity_branch), dim=1)
        output = self.output_proj(combined)
        return output


