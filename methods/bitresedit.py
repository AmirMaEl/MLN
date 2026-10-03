"""BitResEdit on Infinity-2B -- training-free editing inside a given edit mask.

Every scale runs three prompts at once: target, null and source.
  1. Guidance in per-bit log-odds d = logit(bit=1) - logit(bit=0):
       d_cfg = d_T + (cfg - 1)(d_T - d_null)                 (classifier-free guidance)
       d_raw = d_cfg - eta_k * (d_S - d_T)    inside the mask (push away from the source)
     eta_k decays linearly to 0 over the scales, and d_raw is pulled back toward
     d_cfg until KL(Bern(d) || Bern(d_cfg)) <= 0.3 per bit (then |d - d_cfg| <= 5).
  2. Sample target bits and blend them with the source codes in continuous code space:
       codes += src_k + gate * (tgt_k - src_k)
     where gate = dilated, blurred edit mask (1 = edit).
Optional structure lock: for masks covering >= 50 % of the image, the first 3
scales keep the source codes (style / background edits keep the layout).
"""
import torch
import torch.nn.functional as F

from methods.infinity_backbone import Infinity, sample

CFG, ETA, KL_DELTA, L_INF = 4.0392, 2.3898, 0.3, 5.0


def logodds(lg):
    return lg[..., 1] - lg[..., 0]


def bernoulli_kl(d_p, d_q):
    lp1, lp0, lq1, lq0 = F.logsigmoid(d_p), F.logsigmoid(-d_p), F.logsigmoid(d_q), F.logsigmoid(-d_q)
    return (lp1.exp() * (lp1 - lq1) + lp0.exp() * (lp0 - lq0)).clamp_min(0.0)


def trust_region(d_raw, d_ref):
    """Largest step from d_ref toward d_raw with per-bit KL <= KL_DELTA (4 bisection steps)."""
    lo, hi = torch.zeros_like(d_raw), torch.ones_like(d_raw)
    for _ in range(4):
        mid = (lo + hi) / 2
        ok = bernoulli_kl(mid * d_raw + (1 - mid) * d_ref, d_ref) <= KL_DELTA
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    a = torch.where(bernoulli_kl(d_raw, d_ref) <= KL_DELTA, torch.ones_like(lo), lo)
    return d_ref + (a * d_raw + (1 - a) * d_ref - d_ref).clamp(-L_INF, L_INF)


def mask_gate(mask, size):
    """Edit mask -> soft gate on the final token grid: dilate 7, resize, blur (5, sigma 2), max = 1."""
    m = F.max_pool2d(mask.float()[None, None].cuda(), 7, stride=1, padding=3)
    m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
    g = torch.exp(-0.5 * ((torch.arange(5.0) - 2) / 2.0) ** 2)
    g = g / g.sum()
    g = (g[:, None] * g[None, :]).to(m)[None, None]
    m = F.conv2d(m, g, padding=2)
    return (m / m.amax().clamp(min=1e-6)).clamp(0, 1)


class BitResEdit:
    def __init__(self, reso=512, structure_lock=False):
        self.m = Infinity(reso)
        self.lock_scales = 3 if structure_lock else 0

    @torch.inference_mode()
    def edit(self, image, source_prompt, target_prompt, mask, seed=42):
        """image [1,3,R,R] in [-1,1], mask [R,R] (1 = edit) -> edited image [1,3,R,R] in [0,1]."""
        m, K = self.m, len(self.m.scales)
        rng = torch.Generator(device="cuda").manual_seed(seed)
        src_bits = m.source_bits(image.cuda())
        src_codes = [m.codes(b, si, upsample_last=True) for si, b in enumerate(src_bits)]
        gate = mask_gate(torch.as_tensor(mask), m.scales[-1][1:])
        lock = self.lock_scales if float((torch.as_tensor(mask) > 0.5).float().mean()) >= 0.5 else 0
        cond = m.text(target_prompt, "", source_prompt)
        x, summed = m.sos(cond), 0
        m.kv_cache(True)
        try:
            for si, (_, h, w) in enumerate(m.scales):
                lg = m.logits(x, cond, si)                      # [3, h*w*32, 2]: target, null, source
                region = F.interpolate(gate, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
                eta = ETA * (1 - si / (K - 1))
                if si >= lock and eta > 0 and region.amax() > 0:
                    d_T, d_N, d_S = (logodds(l.reshape(1, h * w, -1, 2)) for l in lg[:, None])
                    inside = (region.reshape(1, h * w, 1) > 0).float()
                    d_cfg = d_T + (CFG - 1) * (d_T - d_N)
                    d = trust_region(d_cfg - eta * inside * (d_S - d_T), d_cfg)
                    gen = torch.stack([torch.zeros_like(d), d], -1).reshape(1, -1, 2)
                else:
                    gen = lg[1:2] + CFG * (lg[:1] - lg[1:2])
                tgt_codes = m.codes(sample(gen, rng), si, upsample_last=True)
                g = 0.0 if si < lock else gate
                summed = summed + src_codes[si] + g * (tgt_codes - src_codes[si])
                if si < K - 1:
                    x = m.next_input(summed, si, batch=3)
        finally:
            m.kv_cache(False)
        return m.decode(summed)
