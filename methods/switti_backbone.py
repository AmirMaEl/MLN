"""Switti (512 or 1024 px): loading and one teacher-forced scale step with MLN's (adaptive) logit nudging."""
import torch
import torch.nn.functional as F

from switti.clip import FrozenCLIPEmbedder
from switti.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from switti.switti import SwittiHF, get_crop_condition
from switti.vqvae import VQVAEHF

MODEL_IDS = {512: "yresearch/Switti", 1024: "yresearch/Switti-1024"}
VAE_ID = "yresearch/VQVAE-Switti"
TEXT_ENCODERS = ("openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
TOP_K, TOP_P, LAST_SCALE_TEMP = 400, 0.95, 0.1
STOPWORDS = {  # words the cross-attention mask ignores (as in the MLN release)
    "a</w>", "with</w>", "cup</w>", "of</w>", "on</w>", "photo</w>", "cat</w>", "standing</w>",
    "rocks</w>", "near</w>", "the</w>", "ocean</w>", "<|startoftext|>", "<|endoftext|>", ",</w>",
}


class Switti:
    def __init__(self, reso=512, dtype=torch.float32):
        self.reso, self.dtype, self.seed = reso, dtype, 42
        self.switti = SwittiHF.from_pretrained(MODEL_IDS[reso]).cuda().to(dtype).eval()
        self.vae = VQVAEHF.from_pretrained(VAE_ID, reso=reso).cuda().to(dtype).eval()
        self.vae.quantize = self.vae.quantize.to(dtype)
        self.clip = [FrozenCLIPEmbedder(name, device="cuda").to(dtype) for name in TEXT_ENCODERS]
        self.patch_nums = self.switti.patch_nums
        self.words = None                    # CLIP tokens of the last encoded prompt with > 2 words

    # ---- text -------------------------------------------------------------
    def _encode(self, prompt):
        encs = [c.encode([prompt]) for c in self.clip]
        tok = self.clip[0].tokenizer
        words = tok.convert_ids_to_tokens(tok([prompt])["input_ids"][0])
        if len(words) > 4:
            self.words = words
        return torch.concat([e.last_hidden_state for e in encs], dim=-1), encs[-1].pooler_output, encs[-1].attn_bias

    def encode_prompt(self, prompt, negative_prompt=""):
        """Batched [prompt; negative] condition: (context, pooled vector, attention bias)."""
        emb, pooled, bias = self._encode(prompt)
        B, L, H = emb.shape
        n_emb, n_pooled, n_bias = self._encode(negative_prompt)
        emb = torch.cat([emb, n_emb[:, :L].expand(B, L, H)], 0)
        pooled = torch.cat([pooled, n_pooled.expand(B, pooled.shape[1])], 0)
        bias = torch.cat([bias, n_bias[:, :L].expand(B, L)], 0)
        return emb, self.switti.text_pooler(pooled), bias

    # ---- tokens -----------------------------------------------------------
    def token_embed(self, si, idx):
        pn = self.patch_nums[si]
        return self.vae.quantize.embedding(idx).transpose(1, 2).reshape(idx.shape[0], self.switti.Cvae, pn, pn)

    def add_scale(self, si, f_hat, h):
        """f_hat += phi_si(up(h)) (VAR residual accumulation)."""
        return self.vae.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h)[1]

    def empty_fhat(self):
        p = self.patch_nums[-1]
        return torch.zeros(1, self.switti.Cvae, p, p, device="cuda", dtype=self.dtype)

    # ---- transformer ------------------------------------------------------
    def _forward(self, si, f_hat, cond, both, attn=None):
        """Transformer on scale si; both=False drops the negative branch. Appends cross-attention maps to attn."""
        sw, pn = self.switti, self.patch_nums[si]
        context, cond_BD, bias = cond
        x = F.interpolate(f_hat, size=(pn, pn), mode="area").view(1, sw.Cvae, -1).transpose(1, 2)
        x = (sw.word_embed(x) + sw.lvl_embed(sw.lvl_1L)[:, sw.levels[si]:sw.levels[si + 1]]).repeat(2, 1, 1)
        freqs = sw.freqs_cis[:, sw.levels[si]:sw.levels[si + 1]].repeat(1, 2, 1)
        crop = sw.crop_proj(sw.crop_embed(get_crop_condition(2 * [self.reso], 2 * [self.reso]).cuda().view(-1)).reshape(2, sw.D))
        if not both:
            x, context, bias, cond_BD, crop = x[:1], context[:1], bias[:1], cond_BD[:1], crop[:1]
        for b in sw.blocks:
            b.cross_attn.kv_caching(True)
        for b in sw.blocks:
            out = b(x=x, cond_BD=cond_BD, attn_bias=None, context=context, context_attn_bias=bias,
                    freqs_cis=freqs.cuda(), crop_cond=crop)
            x = out["x"]
            if attn is not None:
                attn.append(out["cross_attn_map"])
        for b in sw.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)
        return sw.get_logits(x, cond_BD)

    @torch.inference_mode()
    def word_attention(self, si, f_hat, cond):
        """Per-word cross-attention maps of the prompt branch at scale si: [words, 1, pn, pn]
        (blocks 3-26, heads averaged; stop words skipped)."""
        pn, maps = self.patch_nums[si], []
        self._forward(si, f_hat, cond, both=False, attn=maps)
        m = torch.cat(maps, 0)[:30].mean(1)[3:27].mean(0).unsqueeze(0).permute(0, 2, 1)
        return torch.cat([m[:, i, :].reshape(1, 1, pn, pn) for i, w in enumerate(self.words[1:]) if w not in STOPWORDS], 0)

    @torch.inference_mode()
    def step(self, si, f_hat, cond, cfg, cfg_off, source_idx, alpha, beta, mask, gamma):
        """Scale si given f_hat (codes of all coarser scales), cond = [target; source] prompts.
        Logits are nudged toward the source tokens with strength beta outside the mask and
        alpha inside, both scaled by (1 - TV)^gamma, TV = per-token disagreement of the two
        prompts (normalized per scale). CFG (source as negative) and Gumbel-smoothed sampling
        while si < cfg_off. Returns token embeddings [1, C, pn, pn]."""
        sw, pn = self.switti, self.patch_nums[si]
        logits, ratio = self._forward(si, f_hat, cond, both=True), si / sw.num_stages_minus_1
        tv = 0.5 * (logits[:1].float().softmax(-1) - logits[1:].float().softmax(-1)).abs().sum(-1, keepdim=True)
        w = (1 - tv / tv.max().clamp(min=1e-6)).clamp(0, 1) ** gamma
        m = F.interpolate(mask, size=(pn, pn), mode="bilinear").view(1, -1, 1)

        def nudge(z):
            delta = F.one_hot(source_idx, num_classes=z.shape[-1]).float() - z.softmax(-1)
            return z + (beta * w * (1 - m) + alpha * w * m) * delta

        guided = 2 <= si < cfg_off
        if guided:
            t = cfg * ratio
            logits = (1 + t) * nudge(logits[:1]) - t * logits[1:]
        else:
            logits = nudge(logits[:1]) / LAST_SCALE_TEMP

        rng, emb = sw.rng, self.vae.quantize.embedding
        rng.manual_seed(self.seed)
        if guided:
            soft = gumbel_softmax_with_rng(logits.mul(1 + ratio), tau=max(0.27 * (1 - ratio * 0.95), 0.005), hard=False,
                                           dim=-1, rng=rng, seed=self.seed).to(self.dtype)
            h = soft @ emb.weight.unsqueeze(0)
        else:
            h = emb(sample_with_top_k_top_p_(logits, rng=rng, top_k=TOP_K, top_p=TOP_P, num_samples=1, seed=self.seed)[:, :, 0])
        return h.transpose(1, 2).reshape(1, sw.Cvae, pn, pn)
