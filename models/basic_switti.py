import math
import warnings

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.functional import scaled_dot_product_attention  # q, k, v: BHLc

from models.helpers import DropPath
from models.rope import apply_rotary_emb

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    print("Using FlashAttention fused MLP")
except ImportError:
    print("FlashAttention fused MLP not available")
    fused_mlp_func = None

# this file only provides the blocks used in Switti transformer
__all__ = ["FFN", "SwiGLUFFN", "RMSNorm", "AdaLNSelfCrossAttn", "AdaLNBeforeHead"]


try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

            """
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.

            """
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            """
            Forward pass through the RMSNorm layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

            """
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: float = 8 / 3,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            ff_mult (float, optional): Custom multiplier for hidden dimension. Defaults to 4.
        """
        super().__init__()
        hidden_dim = int(dim * ff_mult)

        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.fused_mlp_func = None
        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # @torch.compile
    def _forward_silu_gating(self, x_gate: torch.Tensor, x_up: torch.Tensor):
        return F.silu(x_gate) * x_up

    def forward(self, x: torch.Tensor):
        return self.down_proj(
            self._forward_silu_gating(self.gate_proj(x), self.up_proj(x))
        )

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        context_dim: int = 2048,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert attn_drop == 0.0

        self.num_heads, self.head_dim = (
            num_heads,
            embed_dim // num_heads,
        )
        self.qk_norm = qk_norm
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, embed_dim * 2, bias=True)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop = attn_drop

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, context, context_attn_bias=None, freqs_cis=None, replace_map=None):
        B, L, C = x.shape
        context_B, context_L, context_C = context.shape
        assert B == context_B

        q = self.to_q(x).view(B, L, -1)  # BLD
        if self.qk_norm:
            q = self.q_norm(q)

        q = q.view(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # BHLc

        if self.cached_k is None:
            # not using caches or first scale inference
            kv = self.to_kv(context).view(B, context_L, 2, -1)  # qkv: BL3D
            k, v = kv.permute(2, 0, 1, 3).unbind(dim=0)  # q or k or v: BLHD

            if self.qk_norm:
                k = self.k_norm(k)

            k = k.view(B, context_L, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)  # BHLc

            v = v.view(B, context_L, self.num_heads, self.head_dim)
            v = v.permute(0, 2, 1, 3)  # BHLc

            if self.caching:
                self.cached_k = k
                self.cached_v = v
        else:
            k = self.cached_k
            v = self.cached_v
            # When using caching, the query `q` is for the current token (length 1),
            # but the key and value are the cached ones (length L_cached + 1).
            # The manual calculation below handles this shape mismatch correctly.
            # q shape: (B, H, 1, head_dim)
            # k shape: (B, H, L_kv, head_dim)
        if replace_map is None:
            # --- Manual Attention Calculation to get weights ---
            # Calculate QK^T / sqrt(d_k)
            # q shape: (B, H, L_q, head_dim) where L_q is the current sequence length
            # k shape: (B, H, L_kv, head_dim) where L_kv is the context sequence length (or cached length)
            # k.transpose(-2, -1) shape: (B, H, head_dim, L_kv)
            # attn_weights shape: (B, H, L_q, L_kv)
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale

            # Add attention bias (masking) if provided
            if context_attn_bias is not None:
                # context_attn_bias shape: (B, 1, 1, context_L) or (B, 1, L, context_L) if query length varies
                # Ensure the bias shape is compatible with attn_weights (B, H, L_q, L_kv)
                # The original code reshapes context_attn_bias to "b 1 1 j" -> (B, 1, 1, context_L)
                # This seems intended for the query sequence length (L) to broadcast over.
                # Given q is (B, H, L, head_dim) and k is (B, H, context_L, head_dim),
                # attn_weights is (B, H, L, context_L).
                # So context_attn_bias should broadcast correctly if it's (B, 1, 1, context_L) or (B, 1, L, context_L)
                context_attn_bias = rearrange(context_attn_bias, "b j -> b 1 1 j") # Original reshape
                attn_weights = attn_weights + context_attn_bias

            # Apply softmax to get attention probabilities
            # This is the attention map you want
            attn_map = F.softmax(attn_weights, dim=-1)
        else:
            attn_map = replace_map.cuda()
            attn_weights = []


        # Apply dropout to attention map (if in training mode and dropout > 0)
        dropout_p = self.attn_drop if self.training else 0.0
        if dropout_p > 0:
             attn_map = F.dropout(attn_map, p=dropout_p, training=self.training)

        # Multiply by Values
        # attn_map shape: (B, H, L_q, L_kv)
        # v shape: (B, H, L_kv, head_dim)
        # out shape: (B, H, L_q, head_dim)
        out = attn_map @ v
        # --- End Manual Attention Calculation ---

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, C)

        out = self.proj_drop(self.proj(out))

        # Return both the output and the attention map
        return {
            "out": out,
            "attn_map": attn_map,
            "attn_weights": attn_weights,
            "v": v,
        }

class SelfAttention(nn.Module):
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )
        self.qk_norm = qk_norm
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop = attn_drop

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias, freqs_cis: torch.Tensor = None, replace_map=None, randomness_map=None):
        B, L, C = x.shape
        qkv = self.to_qkv(x).view(B, L, 3, -1)
        q, k, v = qkv.permute(2, 0, 1, 3).unbind(dim=0)  # q or k or v: BLD

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.view(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # BHLc
        k = k.view(B, L, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # BHLc
        v = v.view(B, L, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # BHLc
        dim_cat = 2

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis=freqs_cis)
            k = apply_rotary_emb(k, freqs_cis=freqs_cis)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

            # --- Manual Attention Calculation to get weights ---
            # Calculate QK^T / sqrt(d_k)
            # q shape: (B, H, L, head_dim)
            # k shape: (B, H, L_kv, head_dim) - L_kv can be L or L_cached + L
            # k.transpose(-2, -1) shape: (B, H, head_dim, L_kv)
            # attn_weights shape: (B, H, L, L_kv)
        if replace_map is None:
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale

            # Add attention bias (masking) if provided
            if attn_bias is not None:   
                # attn_bias is typically a lower triangular mask for self-attention
                # or other masking strategies. Ensure its shape is compatible.
                # attn_bias shape: (B, 1, L, L_kv) or (1, 1, L, L_kv)
                attn_weights = attn_weights + attn_bias

            # Apply softmax to get attention probabilities
            # This is the attention map you want
            attn_map = F.softmax(attn_weights, dim=-1)
        else:
            attn_map = replace_map.cuda()
            attn_weights = []
        if randomness_map is not None:
            
            attn_map = attn_map  + randomness_map


        # Apply dropout to attention map (if in training mode and dropout > 0)
        dropout_p = self.attn_drop if self.training else 0.0
        if dropout_p > 0:
             attn_map = F.dropout(attn_map, p=dropout_p, training=self.training)

        # Multiply by Values
        # attn_map shape: (B, H, L, L_kv)
        # v shape: (B, H, L_kv, head_dim)
        # out shape: (B, H, L, head_dim)
        out = attn_map @ v
        # --- End Manual Attention Calculation ---


        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, C)

        out = self.proj_drop(self.proj(out))

        # Return both the output and the attention map
        return {
            'out': out,
            'attn_map': attn_map,
            'attn_weights': attn_weights ,
            'v': v,
        }


    def extra_repr(self) -> str:
        return f"attn_l2_norm={self.qk_norm}"


class AdaLNSelfCrossAttn(nn.Module):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        qk_norm=False,
        context_dim=None,
        use_swiglu_ffn=False,
        norm_eps=1e-6,
        use_crop_cond=False,
    ):
        super().__init__()
        assert attn_drop == 0.0
        assert qk_norm

        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
        )

        if context_dim:
            self.cross_attn = CrossAttention(
                embed_dim=embed_dim,
                context_dim=context_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_norm=qk_norm,
            )
        else:
            self.cross_attn = None

        if use_swiglu_ffn:
            self.ffn = SwiGLUFFN(dim=embed_dim)
        else:
            self.ffn = FFN(
                in_features=embed_dim,
                hidden_features=round(embed_dim * mlp_ratio),
                drop=drop,
            )

        self.self_attention_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.self_attention_norm2 = RMSNorm(embed_dim, eps=norm_eps)
        self.cross_attention_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.cross_attention_norm2 = RMSNorm(embed_dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(embed_dim, eps=norm_eps)

        self.attention_y_norm = RMSNorm(context_dim, eps=norm_eps)

        # AdaLN
        lin = nn.Linear(cond_dim, 6 * embed_dim)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None
        
        self.use_crop_cond = use_crop_cond
        if use_crop_cond:
            self.crop_cond_scales = nn.Parameter(torch.zeros(1, cond_dim))

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        crop_cond=None,
        context=None,
        context_attn_bias=None,
        freqs_cis=None,
        replace_cross_map=None,
        replace_self_map=None,  
        randomness_map=None
    ):  # C: embed_dim, D: cond_dim

        if self.use_crop_cond:
            assert crop_cond is not None
            cond_BD = cond_BD + self.crop_cond_scales * crop_cond

        gamma1, gamma2, scale1, scale2, shift1, shift2 = (
            self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        )

        # Call self-attention and get output and attention map
        self_attn_out = self.attn(
             self.self_attention_norm1(x).mul(scale1.add(1)).add(shift1),
             attn_bias=attn_bias,
             freqs_cis=freqs_cis,
             replace_map=replace_self_map,
                randomness_map=randomness_map,  # Pass randomness map if available
        )

        x = x + self.self_attention_norm2(self_attn_out['out']).mul(gamma1)

        cross_attn_map = None # Initialize cross-attention map to None
        if context is not None:
            # Call cross-attention and get output and attention map
            cross_attn_out = self.cross_attn(
                 self.cross_attention_norm1(x),
                 self.attention_y_norm(context),
                 context_attn_bias=context_attn_bias,
                 freqs_cis=freqs_cis,
                    replace_map=replace_cross_map,
            )
            x = x + self.cross_attention_norm2(cross_attn_out['out'])

        x = x + self.ffn_norm2(
             self.ffn(self.ffn_norm1(x).mul(scale2.add(1)).add(shift2))
        ).mul(gamma2)

        # Return the block output and the attention maps
        return {
            "x": x,
            "self_attn_map": self_attn_out["attn_map"],
            "cross_attn_map": cross_attn_out["attn_map"] if cross_attn_out else None
        }
            # "self_attn_weights": self_attn_out["attn_weights"],
            # "cross_attn_weights": cross_attn_out["attn_weights"] if cross_attn_out else None,
            # "v": self_attn_out["v"],    
            # "v_cross": cross_attn_out["v"] if cross_attn_out else None,
        # }

class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
