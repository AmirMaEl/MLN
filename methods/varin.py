"""VARIN (arXiv 2509.01984) on HART-0.7B (1024 px) -- noise inversion for next-scale AR editing.

1. Extract: teacher-force the source tokens with the source prompt. At each scale,
   LAI draws Gumbel noise q with argmax(q) = source token (all other codes truncated
   at least tau below it); keep n = q - logits.
2. Edit with the target prompt: scales < s copy the source tokens; scale t >= s picks
       argmax(logits_tgt + (1 - lam) g + lam n),   lam = (K-1-t) / (K-1-s),  g ~ Gumbel,
   so the cached source noise fades out toward the fine scales. The final scale is
   generated exactly as HART does (MaskGIT tokens + diffusion-head residual).

HART's model code (a modified HART, in hart/) and weights (weights/hart/, or $HART_WEIGHTS).
"""
import math

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from hart.params import WEIGHTS as HART_WEIGHTS

HART_LLM = HART_WEIGHTS / "hart-0.7b-1024px" / "llm"
HART_TEXT = HART_WEIGHTS / "Qwen2-VL-1.5B-Instruct"


def load_hart():
    """(HART transformer, Qwen2 text encoder, tokenizer)."""
    from hart.modules.models.transformer import HARTForT2I
    model = HARTForT2I(AutoConfig.from_pretrained(HART_LLM))
    model.load_state_dict(torch.load(HART_LLM / "ema_model.bin", map_location="cpu"), strict=False)
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(HART_TEXT)
    text = AutoModel.from_pretrained(HART_TEXT, torch_dtype=torch.float16).cuda().eval()
    return model, text, tok


def gumbel(like):
    return -torch.log(-torch.log(torch.rand_like(like).clamp_(1e-6, 1 - 1e-6)))


def lai_noise(true_idx, logits, tau):
    """Location-aware argmax inversion: noise n with argmax(logits + n) == true_idx."""
    l_true = logits.gather(-1, true_idx[..., None])
    q_true = l_true + gumbel(l_true)                                  # Gumbel(l_true)
    u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
    q = logits - torch.logaddexp(logits - (q_true - tau), torch.log(-torch.log(u)))  # truncated at q_true - tau
    q = q.scatter(-1, true_idx[..., None], q_true)
    return q - logits


class VARIN:
    def __init__(self, s=6, tau=18.0, cfg=4.5, maskgit_iters=2):
        self.m, self.text, self.tok = load_hart()
        self.s, self.tau, self.cfg, self.iters = s, tau, cfg, maskgit_iters

    def encode(self, prompt):
        from hart.utils.tools import encode_prompts
        _, mask, pos, emb = encode_prompts([prompt], self.text, self.tok, 300)
        null_mask = torch.zeros_like(mask)
        null_mask[:, 0] = 1
        return (torch.cat([emb, torch.zeros_like(emb)]), torch.cat([pos, torch.zeros_like(pos)]),
                torch.cat([mask, null_mask]))                            # [prompt, null] for CFG

    def guided(self, x, cond_BD, ratio):
        lg, t = self.m.get_logits(x, cond_BD), self.cfg * ratio
        return (1 + t) * lg[:1] - t * lg[1:]

    @torch.inference_mode()
    def _pass(self, prompt, true_idx, noise=None, seed=42):
        """noise=None: extract and return the per-scale noise.  Else: edit and return the image."""
        m, K = self.m, len(self.m.patch_nums)
        rng = torch.Generator(device="cuda").manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)                                             # HART's MaskGIT order
        emb, pos, cmask = self.encode(prompt)
        cond_BD = m.context_embed(m.context_norm(emb))
        lvl_pos = m.lvl_embed(m.lvl_1L) + (m.pos_1LC if m.pos_1LC is not None else 0)
        x_in = cond_BD.expand(2, m.first_l, -1) + lvl_pos[:, :m.first_l]
        if m.pos_start is not None:
            x_in = x_in + m.pos_start.expand(2, m.first_l, -1)
        f_hat = cond_BD.new_zeros(1, m.Cvae, math.ceil(m.ratio * m.patch_nums[-1]), m.patch_nums[-1])

        def run(x, si, **kw):                                            # one scale through the transformer
            gss = m.shared_ada_lin(cond_BD)
            for b in m.blocks:
                x = b(x=x, cond_BD=gss, attn_bias=None, si=si, context_position_ids=pos, context_mask=cmask, **kw)
            return x

        cur_L, noises = 0, []
        for b in m.blocks:
            b.attn.kv_caching(True)
        try:
            for si, pn in enumerate(m.patch_nums[:-1]):                  # discrete scales
                cur_L += m.context_token if si == 0 else pn * math.ceil(m.ratio * pn)
                lg = self.guided(run(x_in, si), cond_BD, si / m.num_stages_minus_1)
                lg = lg[:, -1:] if si == 0 else lg
                if noise is None:
                    noises.append(lai_noise(true_idx[si], lg, self.tau))
                    idx = true_idx[si]
                elif si < self.s:
                    idx = true_idx[si]
                else:
                    lam = (K - 1 - si) / (K - 1 - self.s)
                    idx = (lg + (1 - lam) * gumbel(lg) + lam * noise[si]).argmax(-1)
                h = m.vae_quant_proxy[0].embedding(idx).transpose(1, 2).reshape(1, m.Cvae, pn, pn)
                f_hat, x_in = m.vae_quant_proxy[0].get_next_autoregressive_input_ratio(si, K, f_hat, h, patch_nums=m.patch_nums)
                npn = m.patch_nums[si + 1]
                nlen = npn * math.ceil(npn * m.ratio)
                x_in = (m.word_embed(x_in.view(1, m.Cvae, -1).transpose(1, 2)) + lvl_pos[:, cur_L:cur_L + nlen]).repeat(2, 1, 1)
            if noise is None:
                return noises

            # final scale: HART's MaskGIT (+ continuous residual), CFG ratio kept at (K-2)/(K-1)
            si, ratio, n = K - 1, (K - 2) / (K - 1), m.last_level_pns
            mask, tokens = torch.ones(1, n, device="cuda"), torch.zeros(1, n, m.Cvae, device="cuda")
            o = m.sample_orders(1)
            orders = (o if torch.is_tensor(o) else torch.as_tensor(np.array(o))).cuda().long()
            for step in range(self.iters):
                keep = max(1.0, min(float(mask.sum()) - 1, float(np.floor(n * np.cos(math.pi / 2 * (step + 1) / self.iters)))))
                mask_next = torch.zeros(1, n, device="cuda").scatter(-1, orders[:, :int(keep)], torch.ones(1, n, device="cuda")).bool()
                todo = mask.bool() if step == self.iters - 1 else mask.bool() ^ mask_next
                mask = mask_next.float()
                cur = torch.cat([todo, todo]).nonzero(as_tuple=True)
                x = run(x_in[cur].reshape(2, -1, m.C), si, m_maskgit=cur)
                lg = self.guided(x, cond_BD, ratio)
                v, _ = torch.topk(lg, 300, dim=-1)                       # top-300 sampling
                p = lg.masked_fill(lg < v[..., -1:], float("-inf")).softmax(-1)
                idx = torch.multinomial(p.view(-1, p.shape[-1]), 1, generator=rng).view(1, -1)
                h = m.vae_quant_proxy[0].embedding(idx)
                z = m.decoder_norm(x + m.word_embed(h).repeat(2, 1, 1))
                res = m.diffloss.sample(z=z.reshape(-1, z.shape[-1]), temperature=1.0, cfg=self.cfg * ratio)
                tokens[todo] = (h + res.reshape(2, -1, m.Cvae)[:1]).reshape(-1, m.Cvae)
            f_hat = f_hat + tokens.transpose(1, 2).reshape(f_hat.shape)
            return m.vae_proxy[0].fhat_to_img(f_hat).add(1).mul(0.5).clamp(0, 1)
        finally:
            for b in m.blocks:
                b.attn.kv_caching(False)

    def edit(self, image, source_prompt, target_prompt, seed=42):
        """image [1,3,1024,1024] in [-1,1] -> edited image [1,3,1024,1024] in [0,1]."""
        true_idx = self.m.vae_proxy[0].img_to_idxBl(image.cuda())
        noise = self._pass(source_prompt, true_idx, seed=seed)
        return self._pass(target_prompt, true_idx, noise, seed)
