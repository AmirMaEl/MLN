import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist
from torch import nn as nn
from torch.nn import functional as F

import dist

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ["VectorQuantizer2"]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        beta: float = 0.25,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList([(
                    Phi(Cvae, quant_resi)
                    if abs(quant_resi) > 1e-6
                    else nn.Identity()
                ) for _ in range(share_quant_resi)])
            )
        self.r = 0
        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(
                -abs(eini) / self.vocab_size, abs(eini) / self.vocab_size
            )

    def extra_repr(self) -> str:
        return f"{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}"

    # ===================== `forward` is only used in VAE training =====================
    def forward(
        self, f_BChw: torch.Tensor, ret_usages=False
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(
                self.vocab_size, dtype=torch.float, device=f_BChw.device
            )
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(
                        rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                        dim=1,
                    )
                else:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized():
                        handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = (
                    F.interpolate(
                        self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                        size=(H, W),
                        mode="bicubic",
                    ).contiguous()
                    if (si != SN - 1)
                    else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                )
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        margin = (
            # tdist.get_world_size()
            1
            * (f_BChw.numel() / f_BChw.shape[1])
            / self.vocab_size
            * 0.08
        )
        # margin = pn*pn / 100
        if ret_usages:
            usages = [
                (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                for si, pn in enumerate(self.v_patch_nums)
            ]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(
        self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(
                B,
                self.Cvae,
                self.v_patch_nums[0],
                self.v_patch_nums[0],
                dtype=torch.float32,
            )
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw


    @torch.no_grad()
    def f_to_idxBl_or_fhat(
        self,
        f_BChw: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        noise_std: Optional[float] = None,
        step = None,
        residual: bool = False,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        fhats = []
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []
        # noise_std = .02
        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)[:step] if step is not None else len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            # find the nearest embedding
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            if noise_std is not None:
                z_NC = math.sqrt(1 - noise_std ** 2) * z_NC + torch.randn_like(z_NC) * noise_std
                
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(
                    z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
                f_wo_last = f_hat.clone()
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )
            fhats.append(f_hat.clone())
        # f_hat = self.iterative_soft_project(f_BChw, f_hat, max_iter=5, tau=0.2, alpha=1, tol=1e-4)#512
        # f_hat = self.iterative_soft_project(f_BChw, f_hat, max_iter=3, tau=0.8, alpha=1, tol=1e-4)#1024
        # f_hat, f_rest = self.soft_project_residual(f_hat, f_rest, tau=0.02, alpha=1)
        # f_hat, f_rest = self.soft_project_residual(f_hat, f_rest, tau=0.2, alpha=1)
        # f_h = torch.zeros_like(f_rest)
        # if self.r == 0:
        #     self.r=1
        #     f_h+= self.f_to_idxBl_or_fhat(f_rest, True)['f_hat']
        
        return {
            'idx_list': f_hat_or_idx_Bl,
            'f_hat': f_hat,
            'f_rest': f_rest,
            'fhats': fhats,
            'f_BChw': f_BChw,
        }

    def iterative_soft_project(
        self,
        f_BChw: torch.Tensor,
        f_hat_init: torch.Tensor,
        max_iter: int = 8,
        tau: float = 0.2,
        alpha: float =1,
        tol: float = 1e-4,
        mask = None,
    ) -> torch.Tensor:
        """
        Iteratively refine reconstruction by projecting residual into codebook span.

        Args:
            f_BChw (Tensor): Original encoded features, [B, C, H, W]
            f_hat_init (Tensor): Initial reconstruction, [B, C, H, W]
            max_iter (int): Maximum refinement steps
            tau (float): Softmax temperature
            alpha (float): Step size for correction
            tol (float): Stop early if residual energy is below this threshold

        Returns:
            f_hat (Tensor): Refined reconstruction
        """
        f_hat = f_hat_init.clone()
        final = f_hat_init.clone()
        B, C, H, W = f_hat.shape
        E = self.embedding.weight.data  # (V, C)
        if mask is None:
            mask = torch.ones_like(f_hat)
        else:
            mask = torch.nn.functional.interpolate(mask, size=(H,W), mode='bilinear', align_corners=False).float()

        for it in range(max_iter):
            # Residual
            f_rest = f_BChw - f_hat
            res_norm = f_rest.pow(2).mean().sqrt().item()

            if res_norm < tol:
                break  # early stopping

            # Flatten residual
            z = f_rest.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

            # Soft assignment to codebook
            sims = z @ E.T  # (N, V)
            weights = F.softmax(sims / tau, dim=-1)  # (N, V)
            z_proj = weights @ E  # (N, C)

            # Reshape back
            f_rest_proj = z_proj.view(B, H, W, C).permute(0, 3, 1, 2)

            # Update reconstruction
            f_hat = f_hat + alpha * f_rest_proj 
            final = final + alpha * f_rest_proj * (1-mask)

        return final

    def soft_project_residual(
        self,
        f_hat: torch.Tensor,
        f_rest: torch.Tensor,
        tau: float = 0.1,
        alpha: float = 0.4,
    ) -> torch.Tensor:
        """
        Project residual into the codebook span with soft assignment.

        Args:
            f_hat (Tensor): Current reconstruction, shape [B, C, H, W]
            f_rest (Tensor): Residual (input - f_hat so far), shape [B, C, H, W]
            tau (float): Softmax temperature, controls hardness of assignment
            alpha (float): Scale factor for adding residual correction

        Returns:
            f_hat_corrected (Tensor): Corrected reconstruction, same shape as f_hat
        """
        B, C, H, W = f_rest.shape
        device = f_rest.device

        # Flatten residual
        z = f_rest.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

        # Codebook matrix
        E = self.embedding.weight.data  # (V, C)

        # Similarity scores (dot product)
        sims = z @ E.T  # (N, V)

        # Softmax over vocab
        weights = F.softmax(sims / tau, dim=-1)  # (N, V)

        # Weighted average projection
        z_proj = weights @ E  # (N, C)

        # Reshape back
        f_rest_proj = z_proj.view(B, H, W, C).permute(0, 3, 1, 2)

        # Corrected reconstruction
        f_hat_corrected = f_hat + alpha * f_rest_proj

        return f_hat_corrected, f_rest_proj


    
    def f_to_idxBl_and_frest(
        self,
        f_BChw: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        **kwargs,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        # return: [idx_Bl for si in [0, SN - 2], h_BChw for SN - 1]
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        idx_Bl_and_frest: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.data.square(), dim=1, keepdim=False
            )
            d_no_grad.addmm_(
                z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
            )  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(
                self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                size=(H, W),
                mode="bicubic",
            ).contiguous()
            if si != SN - 1:
                # consistency
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
            f_rest.sub_(h_BChw)
            idx_Bl_and_frest.append(idx_N.reshape(B, ph * pw))

        h_BChw = f_rest
        f_hat = f_hat + h_BChw
        idx_Bl_and_frest.append(
            f_rest_wo_last_discrete.clone()
            .permute(0, 2, 3, 1)
            .reshape(-1, ph * pw, self.Cvae)
        )

        return {
            'idx_list': idx_Bl_and_frest,
            'f_hat': f_hat,
            'f_rest': f_rest,
        }


    @torch.no_grad()
    def f_to_idxBl_or_fhatnq(
        self,
        f_BChw: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        noise_std: Optional[float] = None,
        step = None,
        residual: bool = False,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []
        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)[:step] if step is not None else len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            # find the nearest embedding
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            if noise_std is not None:
                z_NC = math.sqrt(1 - noise_std ** 2) * z_NC + torch.randn_like(z_NC) * noise_std
                
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(
                    z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
                f_wo_last = f_hat.clone()
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )
        f_h = torch.zeros_like(f_rest)
        if self.r == 0:
            self.r=1
            f_h+= self.f_to_idxBl_or_fhat(f_rest, True)['f_hat']
        
        return {
            'idx_list': f_hat_or_idx_Bl,
            'f_hat': f_hat,
            'f_rest': f_hat,
        }

        return f_hat_or_idx_Bl
    def quantize_residual(self, f_rest):
        # quantize f_rest at the full resolution using the same embedding
        B, C, H, W = f_rest.shape
        rest_NC = f_rest.permute(0,2,3,1).reshape(-1, C)   # (B*H*W, C)
        if True:
            rest_NC = F.normalize(rest_NC, dim=-1)
            idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1)
            d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
            idx_N = torch.argmin(d_no_grad, dim=1)
        idx_Bhw = idx_N.view(B, H, W)
        h_BChw = self.embedding(idx_Bhw).permute(0,3,1,2).contiguous()
        h_BChw = self.quant_resi[-1](h_BChw)  # quant_resi for last scale
        return idx_Bhw, h_BChw

    def finit_to_fhat(
        self,
        f_BChw: torch.Tensor,
        f_init: torch.Tensor,
        step: int,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        noise_std: Optional[float] = None,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        patch_hws = patch_hws[:step]
        SN = len(patch_hws)
        
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            # find the nearest embedding
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
            )
            if noise_std is not None:
                z_NC = math.sqrt(1 - noise_std ** 2) * z_NC + torch.randn_like(z_NC) * noise_std
                
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(
                    z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )

        return {'list': f_hat_or_idx_Bl,
                'f_hat': f_hat,
                'f_rest': f_rest,}


    def idxBl_to_fhat(
        self,
        ms_idx_Bl: List[torch.Tensor],
        all_to_max_scale=True,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_idx_Bl[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        

    # ===================== idxBl_to_switti_input: only used in Switti training, for getting teacher-forcing input =====================
    def idxBl_to_switti_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si])
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        # cat BlCs to BLC, this should be float32
        return torch.cat(next_scales, dim=1) if len(next_scales) else None

    def idxBl_to_switti_input_custom(self, gt_ms_idx_Bl: List[torch.Tensor],generate_to=5) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(generate_to):
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si])
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1] if si < SN - 1 else self.v_patch_nums[-1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        # cat BlCs to BLC, this should be float32
        return torch.cat(next_scales, dim=1) if len(next_scales) else None


    # ===================== get_next_autoregressive_input: only used in Switti inference, for getting next step's input =====================
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in Switti inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)]
            h=h(
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return h, f_hat, F.interpolate(
                f_hat,
                size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                mode="area",
                # mode="nearest",
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            # h 
            f_hat.add_(h)
            return h, f_hat, f_hat
    def get_prev_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in Switti inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            # h = self.quant_resi[si / (SN - 1)](
            h = (F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                mode="area",
                # mode="nearest",
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat



    def get_next_autoregressive_input_custom(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in Switti inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = (
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                mode="area",
            )
        else:
            # h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(
            self.resi_ratio
        )


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(
            np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()
        )

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"