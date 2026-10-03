"""AREdit (Wang et al., 2025) on Infinity-2B -- mask-free, training-free editing.

1. Cache: run the source image's own bits through the model with the source
   prompt (CFG-guided) and keep every bit's probability P_src.
2. Edit, scale by scale with the target prompt:
     scales < gamma      keep the source bits,
     later scales        re-sample a bit only where the target makes the source's
                         own bit much less likely:  P_src(r) - P_tgt(r) > tau
                         (or, if an edit mask is given, every bit inside the mask).
3. Attention control: at coarse scales (grid <= 16), every cross-attention head's
   map for words shared by both prompts is replaced by its source-pass map.
"""
import difflib

import torch
import torch.nn as nn

from methods.infinity_backbone import Infinity, sample

CFG, GAMMA, TAU, ATTN_MAX_RES = 3.0, 2, 0.2, 16


class SharedWordAttention(nn.Module):
    """Wraps a cross-attention layer: records (mode 'record') or re-injects
    (mode 'inject') the per-head attention maps of shared words."""

    def __init__(self, base, state):
        super().__init__()
        self.base, self.state, self.saved = base, state, {}

    def forward(self, q, ca_kv):
        st = self.state
        if st["mode"] is None or (st["mode"] == "inject" and st["scale"] not in self.saved):
            return self.base(q, ca_kv)
        kv_compact, cu, _ = ca_kv
        B, Lq = q.shape[:2]
        H, D = self.base.num_heads, self.base.head_dim
        kv = nn.functional.linear(kv_compact, self.base.mat_kv.weight,
                                  torch.cat([self.base.zero_k_bias, self.base.v_bias])).view(-1, 2, H, D)
        lens = (cu[1:] - cu[:-1]).tolist()
        qp = self.base.mat_q(q).view(B, Lq, H, D).permute(0, 2, 1, 3)
        k = torch.zeros(B, max(lens), H, D, device=q.device, dtype=qp.dtype)
        for b, n in enumerate(lens):
            k[b, :n] = kv[cu[b]:cu[b + 1], 0]
        scores = qp.float() @ k.permute(0, 2, 1, 3).float().transpose(-2, -1) * float(self.base.scale)
        for b, n in enumerate(lens):
            scores[b, :, :, n:] = float("-inf")
        attn = scores.softmax(-1)
        if st["mode"] == "record" and st["scale"] in st["record_scales"]:
            self.saved[st["scale"]] = attn[0].clone()           # source prompt = batch row 0
        if st["mode"] == "inject":
            src = self.saved[st["scale"]].to(attn.dtype)
            for t_idx, s_idx in enumerate(st["align"]):     # target prompt = batch row 0
                if t_idx >= lens[0]:
                    break
                if 0 <= s_idx < src.shape[-1]:
                    attn[0, :, :, t_idx] = src[:, :, s_idx]
            attn[0, :, :, :lens[0]] /= attn[0, :, :, :lens[0]].sum(-1, keepdim=True).clamp(min=1e-8)
        v = kv[:, 1]
        out = torch.stack([torch.einsum("hql,lhd->qhd", attn[b, :, :, :n].to(v.dtype), v[cu[b]:cu[b + 1]]).reshape(Lq, -1)
                           for b, n in enumerate(lens)])
        return self.base.proj_drop(self.base.proj(out))


class AREdit:
    def __init__(self, reso=512):
        from infinity.models.basic import CrossAttnBlock
        self.m = Infinity(reso)
        self.attn = {"mode": None, "scale": 0, "align": [], "record_scales": set()}
        for b in self.m.model.unregistered_blocks:
            if isinstance(b, CrossAttnBlock) and not isinstance(b.ca, SharedWordAttention):
                b.ca = SharedWordAttention(b.ca, self.attn)

    def _align(self, src, tgt):
        """Target-token -> source-token index for words the two prompts share (else -1)."""
        toks = lambda p: self.m.tok.convert_ids_to_tokens(self.m.tok(p, truncation=True, max_length=512).input_ids)
        norm = lambda t: t.lstrip("▁").lstrip("Ġ").lower()
        special = lambda t: norm(t) in {"</s>", "<s>", "<pad>"} or norm(t).startswith("<")
        s, t = toks(src), toks(tgt)
        out = [-1] * len(t)
        for tag, i1, _, j1, j2 in difflib.SequenceMatcher(a=[norm(x) for x in s], b=[norm(x) for x in t]).get_opcodes():
            if tag == "equal":
                for o, j in enumerate(range(j1, j2)):
                    if not special(s[i1 + o]) and not special(t[j]):
                        out[j] = i1 + o
        return out

    def _pass(self, cond, fixed_bits=None, p_src=None, rng=None, mask=None):
        """One multi-scale pass. With fixed_bits only: teacher-forced, returns P per scale.
        With p_src too: editing, returns the image."""
        m = self.m
        x, summed, probs = m.sos(cond), 0, []
        m.kv_cache(True)
        try:
            for si, (_, h, w) in enumerate(m.scales):
                self.attn["scale"] = si
                lg = m.logits(x, cond, si)
                lg = lg[1:] + CFG * (lg[:1] - lg[1:])                   # CFG against the null prompt
                p = lg.softmax(-1)
                r = fixed_bits[si].reshape(1, -1, 1).long()
                if p_src is None:                                       # cache pass
                    probs.append(p)
                    bits = fixed_bits[si].reshape(1, h * w, -1)
                else:                                                   # edit pass
                    if mask is None:
                        drop = p_src[si].gather(-1, r)[..., 0] - p.gather(-1, r)[..., 0]
                        edit = (drop > TAU).float().reshape(1, h * w, -1)
                    else:
                        mk = nn.functional.interpolate(mask[None, None], size=(h, w), mode="bilinear", align_corners=False)
                        edit = (mk[0, 0] > 0.5).float().reshape(1, h * w, 1).expand(1, h * w, r.shape[1] // (h * w))
                    if si < GAMMA:
                        edit = torch.zeros_like(edit)
                    new = sample(lg, rng).reshape(1, h * w, -1)
                    bits = edit * new + (1 - edit) * fixed_bits[si].reshape(1, h * w, -1)
                summed = summed + m.codes(bits, si)
                if si < len(m.scales) - 1:
                    x = m.next_input(summed, si, batch=2)
        finally:
            m.kv_cache(False)
        return probs if p_src is None else m.decode(summed)

    @torch.inference_mode()
    def edit(self, image, source_prompt, target_prompt, mask=None, seed=42):
        """image [1,3,R,R] in [-1,1], optional mask [R,R] (1 = edit) -> edited image [1,3,R,R] in [0,1]."""
        rng = torch.Generator(device="cuda").manual_seed(seed)
        bits = self.m.source_bits(image.cuda())
        self.attn.update(mode="record", record_scales={i for i, (_, h, w) in enumerate(self.m.scales) if max(h, w) <= ATTN_MAX_RES})
        for b in self.m.model.unregistered_blocks:
            if isinstance(getattr(b, "ca", None), SharedWordAttention):
                b.ca.saved = {}
        p_src = self._pass(self.m.text(source_prompt, ""), bits)
        self.attn.update(mode="inject", align=self._align(source_prompt, target_prompt))
        try:
            mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32).cuda()
            return self._pass(self.m.text(target_prompt, ""), bits, p_src, rng, mask)
        finally:
            self.attn["mode"] = None
