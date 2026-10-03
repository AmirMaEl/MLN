"""Infinity-2B: loading and the few operations AREdit and BitResEdit need."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

from infinity.models.bsq_vae.vae import vae_model
from infinity.models.infinity import Infinity as InfinityModel
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

WEIGHTS = Path(os.environ.get("INFINITY_WEIGHTS", Path(__file__).resolve().parent.parent / "weights"))
PN = {512: "0.25M", 1024: "1M"}                                  # Infinity's names for the two resolutions


def load_infinity(reso):
    """(transformer, VAE, T5 encoder, T5 tokenizer) for 512 or 1024 px."""
    tok = AutoTokenizer.from_pretrained(WEIGHTS / "flan-t5-xl", legacy=True)
    tok.model_max_length = 512
    t5 = T5EncoderModel.from_pretrained(WEIGHTS / "flan-t5-xl", torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    vae = vae_model(str(WEIGHTS / "infinity_vae_d32_reg.pth"), "dynamic", 32, 2 ** 32, patch_size=16,
                    encoder_ch_mult=[1, 2, 4, 4, 4], decoder_ch_mult=[1, 2, 4, 4, 4], test_mode=True).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        model = InfinityModel(
            vae_local=vae, text_channels=2048, text_maxlen=512, shared_aln=True, raw_scale_schedule=None,
            checkpointing="full-block", customized_flash_attn=False, fused_norm=True, pad_to_multiplier=128,
            use_flex_attn=False, add_lvl_embeding_only_first_block=1, use_bit_label=1, rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2, pn=PN[reso], apply_spatial_patchify=0, inference_mode=True,
            train_h_div_w_list=[1.0], depth=32, embed_dim=2048, num_heads=16, drop_path_rate=0.1, mlp_ratio=4,
            block_chunks=8).cuda()
        for b in model.unregistered_blocks:
            b.bfloat16()
        model.eval().requires_grad_(False)
        model.load_state_dict(torch.load(WEIGHTS / "infinity_2b_reg.pth", map_location="cuda"))
    return model, vae, t5, tok


class Infinity:
    def __init__(self, reso=512):
        self.reso = reso
        self.model, self.vae, self.t5, self.tok = load_infinity(reso)
        tmpl = h_div_w_templates[np.argmin(np.abs(1.0 - h_div_w_templates))]
        self.scales = [(1, h, w) for _, h, w in dynamic_resolution_h_w[tmpl][PN[reso]]["scales"]]

    # ---- text -------------------------------------------------------------
    def _text1(self, prompt):
        t = self.tok(text=[prompt], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        ids, mask = t.input_ids.cuda(), t.attention_mask.cuda()
        n = int(mask.sum())
        kv = self.model.text_norm(self.t5(input_ids=ids, attention_mask=mask)["last_hidden_state"].float()[0, :n])
        cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        cond_BD = self.model.text_proj_for_sos((kv, cu, n))
        with torch.amp.autocast("cuda", enabled=False):
            gss = self.model.shared_ada_lin(cond_BD.float()).float()
        return cond_BD, gss, self.model.text_proj_for_ca(kv), n

    def text(self, *prompts):
        """Batched condition for prompts: (cond_BD, ada-LN input, cross-attn kv).
        Each prompt is encoded on its own and then concatenated."""
        parts = [self._text1(p) for p in prompts]
        lens = [p[3] for p in parts]
        cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32, device="cuda")
        return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
                (torch.cat([p[2] for p in parts]), cu, max(lens)))

    # ---- tokens -----------------------------------------------------------
    def source_bits(self, image):
        """Per-scale bit labels of the image, each [1, h, w, 32]."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h, _, _ = self.vae.encode_for_raw_features(image)
        return self.vae.quantizer(h, scale_schedule=self.scales)[2]

    def codes(self, bits, si, upsample_last=False):
        """Bit labels of scale si -> continuous codes upsampled to the final grid."""
        _, h, w = self.scales[si]
        c = self.vae.quantizer.lfq.indices_to_codes(bits.reshape(1, h, w, -1).unsqueeze(1), label_type="bit_label")
        if si == len(self.scales) - 1 and not upsample_last:
            return c
        return F.interpolate(c, size=self.scales[-1], mode=self.vae.quantizer.z_interplote_up)

    def next_input(self, summed, si, batch):
        """Transformer input for scale si+1 from the running code sum."""
        x = F.interpolate(summed, size=self.scales[si + 1], mode=self.vae.quantizer.z_interplote_up).squeeze(-3)
        x = x.reshape(1, x.shape[1], -1).permute(0, 2, 1)
        return self.model.word_embed(self.model.norm0_ve(x)).repeat(batch, 1, 1)

    def decode(self, summed):
        return ((self.vae.decode(summed.squeeze(-3)) + 1) / 2).clamp(0, 1)

    # ---- transformer ------------------------------------------------------
    def sos(self, cond):
        cond_BD = cond[0]
        return (cond_BD.unsqueeze(1) + self.model.pos_start.expand(cond_BD.shape[0], 1, -1)).to(torch.bfloat16)

    def kv_cache(self, on):
        from infinity.models.basic import CrossAttnBlock
        for b in self.model.unregistered_blocks:
            (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(on)

    def logits(self, x, cond, si):
        """One scale of the transformer -> per-bit logits [B, h*w*32, 2]."""
        m = self.model
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = m.add_lvl_embeding(x, si, self.scales)
            for b in m.unregistered_blocks:
                x = b(x=x, cond_BD=cond[1], ca_kv=cond[2], attn_bias_or_two_vector=None, attn_fn=None,
                      scale_schedule=self.scales, rope2d_freqs_grid=m.rope2d_freqs_grid, scale_ind=si)
        return m.get_logits(x, cond[0]).reshape(x.shape[0], -1, 2)


def sample(logits, rng, top_p=0.97):
    """Nucleus sampling of the 2-way bit logits [B, N, 2] -> bits [B, N]."""
    sl, si = torch.sort(logits, descending=True, dim=-1)
    drop = torch.cumsum(F.softmax(sl, -1), -1) > top_p
    drop[..., 0] = False
    drop[..., 1:] = drop[..., :-1].clone()
    logits = torch.where(drop.scatter(-1, si, drop), float("-inf"), logits)
    p = F.softmax(logits, -1)
    return torch.multinomial(p.view(-1, 2), 1, generator=rng).view(p.shape[:-1])
