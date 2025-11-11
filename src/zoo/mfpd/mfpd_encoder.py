"""
HybridEncoder + Light-Mamba + ASSA integration

"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import get_activation

__all__ = ["HybridEncoder"]

# -------------------------

# ASSA: Adaptive Sparse Self-Attention

# -------------------------

class ASSA(nn.Module):

    def __init__(self, dim, sparsity_threshold=0.1):
        super(ASSA, self).__init__()
        self.dim = dim
        self.sparsity_threshold = sparsity_threshold
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1)
        k = self.key(x).view(B, C, -1)
        v = self.value(x).view(B, C, -1)


        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn_dense = self.softmax(attn)


        mask = F.relu(attn, inplace=False)
        mask = (mask ** 2)
        max_val = mask.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        mask = mask / max_val
        sparse_mask = (mask > self.sparsity_threshold).float()
        attn_sparse = attn * sparse_mask
        attn_sparse = self.softmax(attn_sparse)


        fused_attn = self.alpha * attn_sparse + (1 - self.alpha) * attn_dense
        out = torch.bmm(fused_attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = self.gamma * out + x
        return out

# -------------------------

# Light MambaBlock（Mamba-like）

# -------------------------

class MambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=31, expansion=2, dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion = expansion

        self.norm = nn.LayerNorm(dim)

        self.dw_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim,
            bias=False,
        )

        hidden_dim = int(dim * expansion)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.proj = nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_seq)
        x_conv = self.dw_conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x_mlp = self.mlp(x_conv)
        out_seq = x_seq + x_mlp
        out = out_seq.transpose(1, 2).reshape(b, c, h, w)
        return out


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in, ch_out, kernel_size, stride, g, padding, bias,
        )

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.act(y)

class ELAN(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=2, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output

# -------------------------

# HybridEncoder (MambaBlock + ASSA)

# -------------------------

@register()
class HybridEncoder(nn.Module):
    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,

        use_mamba=True,
        mamba_kernel_size=31,
        mamba_expansion=2,
        mamba_dropout=0.0,

        use_assa=True,
        assa_sparsity_threshold=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        self.use_mamba = use_mamba
        self.mamba_kernel_size = mamba_kernel_size
        self.mamba_expansion = mamba_expansion
        self.mamba_dropout = mamba_dropout

        self.use_assa = use_assa
        self.assa_sparsity_threshold = assa_sparsity_threshold

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )
            self.input_proj.append(proj)

        # encoder transformer (Optional)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.mamba_after_fpn = nn.ModuleList() if self.use_mamba else None
        self.assa_after_fpn = nn.ModuleList() if self.use_assa else None

        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )
            if self.use_mamba:
                self.mamba_after_fpn.append(
                    MambaBlock(hidden_dim, kernel_size=self.mamba_kernel_size,
                               expansion=self.mamba_expansion, dropout=self.mamba_dropout)
                )
            if self.use_assa:
                self.assa_after_fpn.append(ASSA(hidden_dim, sparsity_threshold=self.assa_sparsity_threshold))

        # bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.mamba_after_pan = nn.ModuleList() if self.use_mamba else None
        self.assa_after_pan = nn.ModuleList() if self.use_assa else None

        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    SCDown(hidden_dim, hidden_dim, 3, 2),
                )
            )
            self.pan_blocks.append(
                RepNCSPELAN4(
                    hidden_dim * 2,
                    hidden_dim,
                    hidden_dim * 2,
                    round(expansion * hidden_dim // 2),
                    round(3 * depth_mult),
                )
            )
            if self.use_mamba:
                self.mamba_after_pan.append(
                    MambaBlock(hidden_dim, kernel_size=self.mamba_kernel_size,
                               expansion=self.mamba_expansion, dropout=self.mamba_dropout)
                )
            if self.use_assa:
                self.assa_after_pan.append(ASSA(hidden_dim, sparsity_threshold=self.assa_sparsity_threshold))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (embed_dim % 4 == 0), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # ----- encoder (Optional)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:3+1]  # -> (H, W)
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # [B, HW, C]
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)
                memory: torch.Tensor = self.encoder[i]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # ----- Top-Down FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            fpn_i = len(self.in_channels) - 1 - idx
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            feat_high = self.lateral_convs[fpn_i](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fpn_out = self.fpn_blocks[fpn_i]

            # Mamba -> ASSA
            if self.use_mamba:
                fpn_out = self.mamba_after_fpn[fpn_i](fpn_out)
            if self.use_assa:
                fpn_out = self.assa_after_fpn[fpn_i](fpn_out)

            inner_outs.insert(0, fpn_out)

        # ----- Bottom-Up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsample_convs[idx](feat_low)
            pan_out = self.pan_blocks[idx]

            if self.use_mamba:
                pan_out = self.mamba_after_pan[idx](pan_out)
            if self.use_assa:
                pan_out = self.assa_after_pan[idx](pan_out)

            outs.append(pan_out)

        return outs