"""MLN (ours): Masked Logit Nudging with adaptive nudging, on Switti -- mask-free, training-free editing.

1. Edit mask: the last scale is regenerated once with the source and once with the target
   prompt; the difference of their per-word cross-attention maps is thresholded at its
   q-quantile (1 = edit).
2. Edit: scales < s are copied from the source; every later scale is sampled from the target
   prompt (source prompt as CFG negative) with its logits nudged toward the source token,
       z <- z + (1 - TV)^gamma * (beta (1 - M) + alpha_k M) * (onehot(source) - softmax(z)),
   where TV is the per-token disagreement between the target- and source-prompt predictions
   (normalized per scale): tokens the prompts agree on stay anchored to the source, tokens
   they disagree on are free to follow the target -- even if the mask missed them.
3. Quantization refinement: outside the mask, the residual between the source's continuous
   features and the result is projected onto the codebook and added back.
"""
import numpy as np
import torch
import torch.nn.functional as F

from methods.switti_backbone import Switti

SMOOTH = [12, 11.5, 11, 10, 9, 8, 6, 3, 1.5, 0.5]                     # nudge alpha_k over Switti-512's 10 scales
PRESETS = {
    512: dict(s=5, alphas=SMOOTH, cfg_off=9, q=0.80, refine_iters=5, refine_tau=0.2),
    1024: dict(s=7, alphas=list(np.interp(np.linspace(0, 9, 14), np.arange(10), SMOOTH)), cfg_off=13, q=0.63,
               refine_iters=3, refine_tau=0.8),
}
CFG, BETA, GAMMA = 8.0, 12.0, 1.5


class MLN:
    def __init__(self, reso=512):
        self.s, self.p = Switti(reso), PRESETS[reso]

    def edit_mask(self, image, c_src, c_tgt, source_prompt):
        """Cross-attention-difference mask on the final token grid, [1, 1, G, G], 1 = edit."""
        s, last = self.s, len(self.s.patch_nums) - 1
        s._encode(source_prompt)                                         # word list of the source prompt
        f = s.vae.img_to_fhat(image)[last - 1]                           # source codes of all coarser scales
        a, b = (s.word_attention(last, f.clone(), c) for c in (c_tgt, c_src))
        n = max(a.shape[0], b.shape[0])
        pad = lambda x: torch.cat([x, torch.zeros_like(x[:1]).repeat(n - x.shape[0], 1, 1, 1)]) if x.shape[0] < n else x
        d = (pad(b) - pad(a)).abs()
        d = (d / d.max()).mean(0)
        d = (d / d.max()).float()
        return (d > torch.quantile(d, self.p["q"])).float().unsqueeze(0).to(s.dtype)

    def refine(self, image, f_hat, mask):
        """Quantization refinement outside the mask (project the residual onto the codebook)."""
        E = self.s.vae.quantize.embedding.weight.data
        f = self.s.vae.quant_conv(self.s.vae.encoder(image))
        B, C, H, W = f_hat.shape
        keep = 1 - F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
        cur, out = f_hat.clone(), f_hat.clone()
        for _ in range(self.p["refine_iters"]):
            rest = f - cur
            if rest.pow(2).mean().sqrt().item() < 1e-4:
                break
            z = rest.permute(0, 2, 3, 1).reshape(-1, C)
            proj = (F.softmax(z @ E.T / self.p["refine_tau"], dim=-1) @ E).view(B, H, W, C).permute(0, 3, 1, 2)
            cur, out = cur + proj, out + proj * keep
        return out

    @torch.inference_mode()
    def edit(self, image, source_prompt, target_prompt, global_edit=False, seed=42):
        """image [1,3,R,R] in [-1,1] -> edited image [1,3,R,R] in [0,1].
        global_edit=True (e.g. style changes): the whole image is the edit region, no refinement."""
        s, p = self.s, self.p
        s.seed = seed
        image = image.to("cuda", s.dtype)
        c_tgt = s.encode_prompt(target_prompt, source_prompt)            # source prompt as CFG negative
        c_src = s.encode_prompt(source_prompt, target_prompt)
        grid = s.patch_nums[-1]
        mask = torch.ones(1, 1, grid, grid, device="cuda", dtype=s.dtype) if global_edit else \
            self.edit_mask(image, c_src, c_tgt, source_prompt)
        idx = s.vae.img_to_idxBl(image)                                   # source tokens per scale
        f = s.empty_fhat()
        for si in range(len(s.patch_nums)):
            h = s.token_embed(si, idx[si]) if si < p["s"] else \
                s.step(si, f.clone(), c_tgt, CFG, p["cfg_off"], idx[si], p["alphas"][si], BETA, mask, GAMMA)
            f = s.add_scale(si, f, h)
        if not global_edit:
            f = self.refine(image, f, mask)
        return (s.vae.fhat_to_img(f).float().clamp(-1, 1) + 1) / 2
