import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
import torch.nn.functional as F
from mln_utils import save_data_batch, plot_data_batch, save_tensor_as_image, tensor_to_pil
from models.ste import SampleSTE
from tqdm import tqdm
import numpy as np
import torchvision.transforms as ts
import torch.nn as nn
import time

import cv2

class MaskNet(nn.Module):
    def __init__(self, cond_channels, codebook_size, hidden_dim=128):
        super().__init__()
        self.conv_in = nn.Conv2d(cond_channels + codebook_size, hidden_dim, 3, padding=1)
        self.conv_mid = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, codebook_size, 1)
        
    def forward(self, cond_feat, logits):
        # logits: [B, T, V] -> [B, V, H, W]
        if logits.dim() == 3:
            B, T, V = logits.shape
            H = W = int(T**0.5)
            logits_spatial = logits.view(B, H, W, V).permute(0, 3, 1, 2)
        else:
            logits_spatial = logits  # already [B, V, H, W]
        
        # Fuse condition + logits
        x = torch.cat([cond_feat, logits_spatial], dim=1)  # [B, C_cond+V, H, W]
        
        # Mask prediction
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_mid(x))
        mask_logits = self.conv_out(x)  # [B, V, H, W]
        
        mask = torch.sigmoid(mask_logits)  # soft mask in [0,1]
        return mask

torch.autograd.set_detect_anomaly(True)

TRAIN_IMAGE_SIZE = (1024, 1024)

def print(*args, **kwargs):
    pass
class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(self, switti, vae, text_encoder, text_encoder_2,
                 device, dtype=torch.float32, verbose=True
                 ):
        self.verbose = verbose
        self.switti = switti.to(dtype)
        self.vae = vae.to(dtype)
        self.vae.quantize = self.vae.quantize.to(dtype)
        self.text_encoder = text_encoder.to(dtype)
        self.text_encoder_2 = text_encoder_2.to(dtype)
        self.dtype = dtype

        self.switti.eval()
        self.vae.eval()
        self.visualize = False

        self.device = device
        self.begin_ends = []
        self.seed = 3
        self.prompt_pos = None
        cur = 0
        for pn in self.switti.patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device="cuda",
                        reso=1024,
                        ):
        switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)
        

        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: str or list[str]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        prompt_embeds = torch.concat(
            [encoding.last_hidden_state for encoding in encodings], dim=-1
        )
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias
        prompt_pos = self.text_encoder.tokenizer.convert_ids_to_tokens(self.text_encoder.tokenizer(prompt)['input_ids'][0])
        if len(prompt_pos) >  4 and self.verbose:
           print('#' * 20)
           print(prompt_pos)
           print('#' * 20)
           self.prompt_pos = prompt_pos

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(
        self,
        prompt: str or list[str],
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)
            
            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str or list[str],
        null_prompt: str = "",
        seed: int or None = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = False,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: None or float = None,
        return_fhats = False,
        kv_caching: bool = False,
    ) -> torch.Tensor or list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        fhats = []
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

        B = context.shape[0] // 2

        cond_vector = switti.text_pooler(cond_vector)

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
                                             2 * B * [TRAIN_IMAGE_SIZE[1]],
                                             ).to(cond_vector.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = cond_vector

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])
        if kv_caching:
            for b in switti.blocks:
                b.attn.kv_caching(switti.use_ar) # Use KV caching if switti is in the AR mode 
                b.cross_attn.kv_caching(True)

        for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map



            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                if kv_caching:
                    for b in switti.blocks:
                        if b.attn.caching and b.attn.cached_k is not None:
                            b.attn.cached_k = b.attn.cached_k[:B]
                            b.attn.cached_v = b.attn.cached_v[:B]
                        if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                            b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                            b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in switti.blocks:
                start = time.time()
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
                end = time.time()
                print(f"Block {si} forward time: {end - start:.4f}s")
                x_BLC = x_BLC['x']
            cur_L += pn * pn

            logits_BlV = switti.get_logits(x_BLC, cond_BD)


            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            _, f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw,
            )
            fhats.append(f_hat.clone())
            if si != switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)
        
            for b in switti.blocks:
                b.attn.kv_caching(False)
                b.cross_attn.kv_caching(False)
        if return_fhats:
            fs = []
            for f in fhats:
                tmp = vae.fhat_to_img(f)
                fs.append(tmp)
            fs = torch.stack(fs, dim=0)
            img = fs[-1]
            return {'fhats': fs, 'img': img}
        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)

        return img

    def fhats_to_img(self, fhats):
        vae = self.vae
        fs  =[]
        for f in fhats:
            tmp = vae.fhat_to_img(f)
            fs.append(tmp)
        fs = torch.stack(fs, dim=0)
        return fs
    def precondition_logits_nudge(
            self,
    logits_pred: torch.Tensor,    # [B, N, V] predicted by model
    gt_tokens: torch.Tensor,      # [B, N] ground-truth token ids
    alpha: float = .2,
    nudge_mask: torch.Tensor = None
) -> torch.Tensor:
        print('precondition_logits_nudge alpha: ', alpha)
        if nudge_mask is not None:
            keepmask = nudge_mask.unsqueeze(-1).expand_as(logits_pred)
            editmask = torch.ones_like(keepmask).float() - keepmask
        else:
            editmask = 0
            keepmask = 1
        with torch.no_grad():
            p_gt = F.one_hot(gt_tokens, num_classes=logits_pred.size(-1)).float()  # [B, N, V]
        p_pred = F.softmax(logits_pred, dim=-1)
        # Soft residual toward GT distribution
        logits_precond = logits_pred  + editmask * 12* (p_gt - p_pred) + keepmask * alpha * (p_gt - p_pred)
        return logits_precond
            
    def apply_control_nudge(self,logits_pred, control_logits, alpha=0.2, mask=None):
        with torch.no_grad():
            p_ctrl = F.one_hot(control_logits, num_classes=logits_pred.size(-1)).float()  # [B, N, V]
        p_pred = F.softmax(logits_pred, dim=-1)


        delta = p_ctrl - p_pred
        print('apply_control_nudge alpha: ', alpha)
        if mask is not None:
            delta = delta * mask.unsqueeze(-1).float()

        return logits_pred + alpha * delta

    @torch.inference_mode()
    def single_step(
        self,
        x_BLC,
        si, 
        switti,
        cond_BD, 
        context, 
        context_attn_bias,
        crop_cond,
        top_k,
        top_p,
        cfg,
        more_smooth,
        smooth_start_si,
        turn_on_cfg_start_si,
        turn_off_cfg_start_si,
        last_scale_temp,
        rng,
        f_hat,
        full_context = False,
        freqs_cis = None,
        save_attn_maps = True,
        replace_cross_map = None,
        replace_self_map = None,
        return_logit=False,
        noise_map=None,
        preconditioner=None,
        nudge_gt=None, 
        nudge_alpha=0.2,
        control_nudge=None,
        control_alpha=0.2,
        nudge_mask=None


    ):
        if isinstance(cfg, (list, tuple)):
            cfg = cfg[si]
        # print('step forward  %i'%si)
        B = x_BLC.shape[0] // 2
        vae_quant = self.vae.quantize.to(x_BLC.dtype)
        selfAttenMaps = []
        v_self = []
        v_cross = []
        crossAttenMaps = []
        selfweights= []
        crossweights = []
        if freqs_cis is None:
            freqs_cis = switti.freqs_cis[:,  : switti.levels[si +1]]

        if si >= turn_off_cfg_start_si:
            apply_smooth = False
            x_BLC = x_BLC[:B]
            context = context[:B]
            context_attn_bias = context_attn_bias[:B]
            freqs_cis = freqs_cis[:B]
            cond_BD = cond_BD[:B]
            if crop_cond is not None:
                crop_cond = crop_cond[:B]
            for b in switti.blocks:
                if b.attn.caching and b.attn.cached_k is not None:
                    b.attn.cached_k = b.attn.cached_k[:B]
                    b.attn.cached_v = b.attn.cached_v[:B]
                if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                    b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                    b.cross_attn.cached_v = b.cross_attn.cached_v[:B]


        else:
            apply_smooth = more_smooth
        if x_BLC.shape[0] == 2:
            m1 = x_BLC[:B]
            m2 = x_BLC[B:]
        
        
            print('cfg at scale: ', si)
        else:
            m1 = x_BLC
        freqs_cis = freqs_cis.repeat(1,2,1)
        if nudge_mask is not None:
            nudge_mask= F.interpolate(nudge_mask, size=(self.switti.patch_nums[si],self.switti.patch_nums[si]), mode='bilinear')
            nudge_mask = nudge_mask.view(1, -1)
        for b in switti.blocks:
            b.cross_attn.kv_caching(True)
        ratio = si / switti.num_stages_minus_1
        # x_BLC = next_token_map




        if si >= turn_off_cfg_start_si:
            apply_smooth = False
            x_BLC = x_BLC[:B]
            context = context[:B]
            context_attn_bias = context_attn_bias[:B]
            freqs_cis = freqs_cis[:B]
            cond_BD = cond_BD[:B]
            if crop_cond is not None:
            
                crop_cond = crop_cond[:B]
            for b in switti.blocks:
                if b.attn.caching and b.attn.cached_k is not None:
                    b.attn.cached_k = b.attn.cached_k[:B]
                    b.attn.cached_v = b.attn.cached_v[:B]
                if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                    b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                    b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
        else:
            apply_smooth = more_smooth

        for block in switti.blocks:
            x_BLC = block(
                x=x_BLC,
                cond_BD=cond_BD,
                attn_bias=None,
                context=context,
                context_attn_bias=context_attn_bias,
                freqs_cis=freqs_cis.to('cuda'),
                crop_cond=crop_cond,
            )
 
            crossAttenMaps.append(x_BLC['cross_attn_map']) if 'cross_attn_map' in x_BLC else None
            x_BLC = x_BLC['x']





        #     m1 = block(
        #         x=m1,
        #         cond_BD=cond_BD[:B],
        #         attn_bias=None,
        #         context=context[:B],
        #         context_attn_bias=context_attn_bias[:B],
        #         freqs_cis=freqs_cis[:B],
        #         crop_cond=crop_cond[:B],
        #         replace_cross_map=replace_cross_map[j] if replace_cross_map is not None else None,
        #         replace_self_map=replace_self_map[j] if replace_self_map is not None else None,
        #         randomness_map=noise_map if noise_map is not None else None,
        #     )
        #     selfAttenMaps.append(m1['self_attn_map']) if 'self_attn_map' in m1 else None
        #     crossAttenMaps.append(m1['cross_attn_map']) if 'cross_attn_map' in m1 else None
        #     m1 = m1['x']
        #     v_self = None
        #     v_cross = None
        #     selfweights = None
        #     crossweights = None
            
        # if preconditioner is not None:
        #     m1 = m1[:, preconditioner.shape[1]:]
        # # if x_BLC.shape[0] == 2:
        # #     for j, block in enumerate(switti.blocks):
        # #         m2 = block(
        # #             x=m2,
        # #             cond_BD=cond_BD[B:],
        # #             attn_bias=None,
        # #             context=context[B:],
        # #             context_attn_bias=context_attn_bias[B:],
        # #             freqs_cis=freqs_cis,
        # #             crop_cond=crop_cond[B:] if crop_cond is not None else None,
        # #             replace_cross_map=replace_cross_map[j] if replace_cross_map is not None else None,
        # #             replace_self_map=replace_self_map[j] if replace_self_map is not None else None,
        # #             randomness_map=noise_map if noise_map is not None else None,
        # #         )   
        # #         selfAttenMaps.append(m2['self_attn_map']) if 'self_attn_map' in m2 else None
        # #         m2 = m2['x']
        # #     x_BLC = torch.cat([m1, m2], dim=0)
        # # else:
        # x_BLC = m1
            

        # cur_L += pn * pn
        # print('x_BLC shape', x_BLC.shape)
        x_BLC_init = x_BLC[:,:switti.levels[si]].clone()
        # x_BLC = x_BLC[:,switti.levels[si]:switti.levels[si+1]]
        logits_BlV = switti.get_logits(x_BLC, cond_BD)
        ratio = si / switti.num_stages_minus_1

        # Guidance
        if si < turn_on_cfg_start_si:
            logits_BlV = logits_BlV[:B]
            logits_cond = logits_BlV
            logits_uncond = None
        elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
            t = cfg * ratio
            logits_cond = logits_BlV[:B]
            logits_uncond = logits_BlV[B:]
            if nudge_gt is not None:
                logits_cond = self.precondition_logits_nudge(
                logits_pred=logits_cond,
                gt_tokens=nudge_gt,
                alpha=nudge_alpha,
                nudge_mask=nudge_mask
            )
            if control_nudge is not None:
                logits_cond = self.apply_control_nudge(
                    logits_pred=logits_cond,
                    control_logits=control_nudge,
                    alpha=control_alpha,
                    nudge_mask=nudge_mask
                )
                print('nudged scale: ', si)
            logits_BlV = (1 + t) * logits_cond - t * logits_uncond
        elif last_scale_temp is not None:
            logits_BlV = logits_BlV / last_scale_temp
        if return_logit:
            return {
                'logits_cond': logits_cond,
                'logits_uncond': logits_uncond, 
                'logits': logits_BlV,
                'x_BLC': x_BLC,
            }

        if apply_smooth and si >= smooth_start_si:
            # not used when evaluating FID/IS/Precision/Recall
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
            idx_Bl = gumbel_softmax_with_rng(
                logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng, seed=self.seed,
            ).to(self.dtype     )
            h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            print('gumbel smooth at scale: ', si)
        else:
            # default nucleus sampling
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1, seed=self.seed,
            )[:, :, 0]
            h_BChw = vae_quant.embedding(idx_Bl)
        pn = switti.patch_nums[si]
        print('h_BChw shape', h_BChw.shape)
        print(' pn\n', pn)
        
        h_BChw = h_BChw.transpose(1, 2).reshape(B, switti.Cvae, pn, pn)

        
        h_BChw_res,f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                si, len(switti.patch_nums), f_hat, h_BChw,
            )

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if si != switti.num_stages_minus_1:  # prepare for next stage
            next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
            next_token_map = (
                switti.word_embed(next_token_map)
                + lvl_pos[:, switti.levels[si + 1] : switti.levels[si + 2]]
            )
            # double the batch sizes due to CFG
            next_token_map = next_token_map.repeat(2, 1, 1)
        else:
            next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
            next_token_map = (
                switti.word_embed(next_token_map)
                + lvl_pos[:, switti.levels[si] : switti.levels[si + 1]]
            )
            # double the batch sizes due to CFG
            next_token_map = next_token_map.repeat(2, 1, 1)
                
        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)
        return {
            'next_token_map': next_token_map,
            'h_BChwres': h_BChw_res,
            'h_BChw': h_BChw,
            # 'selfAttenMaps': self.process_attn_maps(selfAttenMaps, pn=pn) ,
            'crossAttenMaps': self.process_attn_maps(crossAttenMaps, cross_attn_map=True, pn=pn) ,
            'v_self': v_self,
            'v_cross': v_cross,
            'f_hat': f_hat,
            'selfweights': selfweights,
            'crossweights': crossweights,
        }

        return tmp

    def process_attn_maps(self, attn_mapin, cross_attn_map=False,pn=4):
        # ['<|startoftext|>', 'image</w>', 'of</w>', 'a</w>', 'dog</w>'
        #  , 'playing</w>', 'on</w>', 'gras</w>', ',</w>', '4</w>', 'k</w>', ',</w>', 
        #  'high</w>', 'resolution</w>', ',</w>', 'photo', 'realistic</w>', '<|endoftext|>']
        dic = {}
        visualize = self.visualize
        self.attn_map = {}
        final_maps = []

        for i in range(len(self.switti.blocks)):
            dic.update({'cross_layer_%i'%i: attn_mapin})
        no_prompt = ['a</w>','of</w>','on</w>','<|startoftext|>','<|endoftext|>',',</w>','jockey</w>','rides</w>']

        if cross_attn_map:
            if self.prompt_pos is None: return None
            
            for i,word in enumerate(self.prompt_pos[1:-2]):
                if word in no_prompt:
                    continue
                attn_map = torch.cat(attn_mapin, dim=0)[:30]
                attn_map = torch.mean(attn_map, dim=1)[3:27]
                attn_map = torch.mean(attn_map, dim=0).unsqueeze(0)
                
                attn_map = attn_map.permute(0, 2, 1)
                attn_map = attn_map[:,i,:]
                attn_map = attn_map.reshape(1,1, pn, pn)
                # attn_map = attn_map / attn_map.max()
                final_maps.append(attn_map.detach())
            final_maps = torch.cat(final_maps, dim=0)
            # final_maps = final_maps / final_maps.max()



        else:
            final_maps = torch.cat(attn_mapin, dim=0).cpu()
        return final_maps


    # @torch.inference_mode()
    def invert_step(
        self,
        image_B3HW,
        prompt: str or list[str],
        null_prompt: str = "",
        seed: int or None = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = True,
        return_pil: bool = True,
        smooth_start_si: int = 2,
        turn_off_cfg_start_si: int = 2,
        turn_on_cfg_start_si: int = 5,
        cut_forward: bool = True,
        last_scale_temp=.1,
        step=3,
        f_hat=None,
        save_attn_maps = False,
        replace_cross_map = None,
        replace_self_map = None,
        return_logit=False,
        noise_map=None,
        preconditioner=None,
        nudge_gt=None,
        nudge_alphas=None,
        nudge_mask=None,
        control_nudge=None,
        control_alphas=None,
    ):
        apply_smooth = more_smooth
        seed=5
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        switti.rng.manual_seed(seed)
        rng = switti.rng
        fh = f_hat
        if nudge_alphas is not None:
            nudge_alpha = nudge_alphas[step] 
        else:
            nudge_alpha = 0.2
        if control_alphas is not None:
            control_alpha = control_alphas[step] 
        else:
            control_alpha = 0.2 
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        cond_vector = switti.text_pooler(cond_vector)
        B = context.shape[0] // 2
        crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
                                             2 * B * [TRAIN_IMAGE_SIZE[1]],
                                             ).to(cond_vector.device)
        crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
        crop_cond = switti.crop_proj(crop_embed)
        sos  = cond_BD = cond_vector
        fhats = []
        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        gt_fhat = vae.img_to_fhat(image_B3HW)
        gt_idxBl= vae.img_to_idxBl(image_B3HW)
        inp = gt_idxBl[step]
        h_BChw = vae_quant.embedding(inp)
        h_BChw = h_BChw.transpose(1, 2).reshape(B, switti.Cvae, switti.patch_nums[step], switti.patch_nums[step])
        f_hat = torch.nn.functional.interpolate(
            gt_fhat[step] if f_hat is None else f_hat.clone(),
            size=(switti.patch_nums[step], switti.patch_nums[step]),
            mode="area",
        )
        # f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
        #     len(switti.patch_nums)-3, len(switti.patch_nums), f_hat, h_BChw,
        # )
        next_token_map = f_hat
        next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
        
        next_token_map = (
            switti.word_embed(next_token_map)
            + lvl_pos[:, switti.levels[step] : switti.levels[step+1]]
        )
        
        uncond = torch.zeros_like(next_token_map)
        freqs_cis = switti.freqs_cis[:, switti.levels[step] : switti.levels[step + 1]]
        next_token_map = next_token_map.repeat(2, 1, 1)
        # if noise_map is not None:
            # noise_map = torch.nn.functional.interpolate(
            #     noise_map,
            #     size=(switti.patch_nums[step], switti.patch_nums[step]),
            #     mode="area",
            # )
            # noise_map = noise_map.view(B, switti.Cvae, -1).transpose(1, 2)
            # noise_map = (switti.word_embed(noise_map) + lvl_pos[:, switti.levels[step] : switti.levels[step + 1]])
            # next_token_map[:] = next_token_map[:] + noise_map
            # cond_BD[:] = cond_BD[:] + noise_map
        # double the batch sizes due to CFG
        x_BLC = next_token_map
        tmp = self.single_step(
        x_BLC,
        step,
        switti,
        cond_BD, 
        context, 
        context_attn_bias,
        crop_cond,
        top_k,
        top_p,
        cfg,
        more_smooth,
        smooth_start_si,
        turn_on_cfg_start_si,
        turn_off_cfg_start_si,
        last_scale_temp,
        rng,
        gt_fhat[step] if fh is None else fh,
        full_context = False,
        freqs_cis = freqs_cis,
        save_attn_maps = save_attn_maps,
        replace_cross_map = replace_cross_map,
        replace_self_map = replace_self_map,
        return_logit=return_logit,
        # noise_map=noise_map,
        preconditioner=preconditioner,
        nudge_gt=nudge_gt,
        nudge_alpha= nudge_alpha,
        nudge_mask=nudge_mask,
        control_nudge=control_nudge,
        control_alpha=control_alpha,
        )
        return  tmp


    def get_gt_token(self, image_B3HW, step):
        vae = self.vae
        gt_fhat = vae.img_to_fhat(image_B3HW)
        gt_idxBl= vae.img_to_idxBl(image_B3HW)
        curfh = gt_fhat[step]
        finfh = gt_fhat[-1]
        gt_id = finfh - curfh
        gt_id = vae.fhat_to_idxBl(gt_id)


        return h_BChw

    @torch.inference_mode()
    def get_mask(self,prompt,eprompt,image_B3HW,smooth_start_si,
                 turn_on_cfg_start_si,turn_off_cfg_start_si,last_scale_temp,steps=[6],quantile=0.8):

        f_hat = self.vae.img_to_fhat(image_B3HW)[steps[0]-1]
        f_hatinit = f_hat.clone()
        maps = {'self': [], 'cross': [], 'crosse':[]}
        for si in steps:
            with torch.no_grad():
                tmp = self.invert_step(
                        image_B3HW, 
                        eprompt,
                        step=si,
                        f_hat=  f_hat,
                        # noise_map=nm[si-4] if si in [4,5,6] else None,
                        # preconditioner=nm[si-3] if si in [3,4] else None, 
                        smooth_start_si=smooth_start_si,
                        turn_on_cfg_start_si=turn_on_cfg_start_si,
                        turn_off_cfg_start_si=turn_off_cfg_start_si,
                        last_scale_temp=last_scale_temp
                                    )
                f_hat = tmp['f_hat'].clone()
                maps['cross'].append(tmp['crossAttenMaps'])
        f_hat = f_hatinit.clone()
        for si in steps:
            with torch.no_grad():
                tmp = self.invert_step(
                        image_B3HW, 
                        prompt,
                        step=si,
                        f_hat=  f_hat,
                        # noise_map=nm[si-4] if si in [4,5,6] else None,
                        # preconditioner=nm[si-3] if si in [3,4] else None, 
                        smooth_start_si=smooth_start_si,
                        turn_on_cfg_start_si=turn_on_cfg_start_si,
                        turn_off_cfg_start_si=turn_off_cfg_start_si,
                        last_scale_temp=last_scale_temp
                                    )
                f_hat = tmp['f_hat'].clone()
                maps['crosse'].append(tmp['crossAttenMaps'])
        diff = []
        length = max(len(maps['cross']), len(maps['crosse']))
        for i in range(length):
            cross = maps['cross'][i]
            crosse = maps['crosse'][i] 
            if cross.shape != crosse.shape:
                if cross.shape[0] < crosse.shape[0]:
                    t = torch.zeros_like(crosse)
                    cross = torch.cat([cross, t[cross.shape[0]:]], dim=0)
                elif crosse.shape[0] < cross.shape[0]:
                    t = torch.zeros_like(cross)
                    crosse = torch.cat([crosse, t[crosse.shape[0]:]], dim=0)


            # tmp1 = F.mse_loss(cross, crosse, reduction='none')
            tmp1 = self._normalize(crosse) * self._normalize(cross)
            tmp1 = tmp1/tmp1.max()
            diff.append(tmp1)

        diffs = []
        for i,d in enumerate(diff):
            tmp =torch.mean(d, dim=0)
            tmp = tmp/tmp.max()
            tmp = tmp.to(torch.float32)
            threshold = torch.quantile(tmp, quantile)  # top 20%
            tmp = tmp.to(self.dtype)

            tmp = torch.where(tmp > threshold, 1, torch.tensor(0.0, device=tmp.device))
            diffs.append(tmp.unsqueeze(0))
        if self.visualize:    
            save_data_batch(torch.cat([diff[-1],diffs[-1]]), f'output/diff_crossAttenMaps', grid=[3,6])
            save_data_batch((maps['cross'][-1]), f'output/orig_crossAttenMaps', grid=[5,6])
            save_data_batch((maps['crosse'][-1]), f'output/recon_crossAttenMaps', grid=[5,6])
        return diffs[-1].cuda()
    
    def _normalize(self,X):
        # per-map min-max to [0,1]
        eps = 1e-5
        mn = X.amin(dim=tuple(range(X.ndim-1)), keepdim=True) if X.ndim==3 else X.min()
        mx = X.amax(dim=tuple(range(X.ndim-1)), keepdim=True) if X.ndim==3 else X.max()
        return (X - mn) / (mx - mn + eps)


    @torch.inference_mode()
    def invert_stepwise(
        self,
        image_B3HW,
        prompt: str or list[str],
        ref_img= None,
        eprompt: str or list[str] = "",
        null_prompt: str = "",
        seed: int or None = None,
        cfg: float = 8.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = True,
        return_pil: bool = True,
        smooth_start_si: int = 2,
        turn_on_cfg_start_si: int = 8,
        turn_off_cfg_start_si: int = 2,
        cut_forward: bool = True,
        last_scale_temp=.1,
        start_scale = 7,
        visualize = True,
        include_residual: bool = False,
        mask = None,
        nudge_alphas=[6,6,6,6,6,6,5,4,3,0,0,0,0,0],

    ):
        self.visualize = visualize
        apply_smooth = more_smooth
        self.seed=seed
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        switti.rng.manual_seed(seed)
        rng = switti.rng
        # canny = kornia.filters.Canny(low_threshold=0.02, high_threshold=0.4)
        # edges, _ = canny(ref_img)
        # edges = edges.to(self.dtype)
        # image_B3HW = image_B3HW.to(self.dtype)
        # ref_img = ref_img.to(self.dtype)
        # fhat_edges = vae.img_to_fhat(edges.repeat(1,3, 1, 1))[-1]
        # edges_idx = vae.img_to_idxBl(edges.repeat(1,3, 1, 1))
        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        B = context.shape[0] // 2
        cond_vector = switti.text_pooler(cond_vector)
        crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
                                             2 * B * [TRAIN_IMAGE_SIZE[1]],
                                             ).to(cond_vector.device)
        crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
        crop_cond = switti.crop_proj(crop_embed)
        sos  = cond_BD = cond_vector
        fhats = []
        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        gt_fhat = vae.img_to_fhat(image_B3HW)
        f_rest = vae.img_to_frest(image_B3HW)
        # gt_fhatr = vae.img_to_fhat(ref_img)
        gt_idxBl= vae.img_to_idxBl(image_B3HW)
        # gtfh = vae.idxBl_to_fhat(gt_idxBl, same_shape=True)
        # gt_edges = vae.img_to_fhat(edges.repeat(1,3, 1, 1))
        inp = gt_idxBl[start_scale]
        h_BChw = vae_quant.embedding(inp)
        # f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
        #     len(switti.patch_nums)-3, len(switti.patch_nums), f_hat, h_BChw,
        # )
        f_hat = gt_fhat[1].clone()
        fhats.append(gt_fhat[-1].clone())
        # f_hat = self.vae.img_to_f(prior_from_edges(edges.to('cuda')).to('cuda'))

        # f_hat = gt_edges[0].clone()
        gt= vae.img_to_f(image_B3HW)

        fhats.append(f_hat.clone())
        maps = {'self': [], 'cross': [], 'crossCond': []}
        if mask is not None:
            m1 = F.interpolate(mask, size=(self.switti.patch_nums[9],self.switti.patch_nums[9]), mode='bilinear')
            save_data_batch(torch.cat([m1,m1]), 'output/diff_crossAttenMaps', grid=[1,2])

        else:
            mask = self.get_mask(prompt, eprompt, image_B3HW, smooth_start_si, turn_on_cfg_start_si, turn_off_cfg_start_si, last_scale_temp, steps=[10,11,12,13], quantile=0.8)

        print('initial prompt', prompt)
        print('initial eprompt', eprompt)
        # f_hat = vae.img_to_frest(image_B3HW)
        for si in [2,3,4,5,6,7,8,9,10,11,12,13]:
            with torch.no_grad():
                tmp = self.invert_step(
                        image_B3HW, 
                        eprompt,
                        step=si,
                        f_hat=  f_hat,
                        # noise_map=nm[si-4] if si in [4,5,6] else None,
                        # preconditioner=nm[si-3] if si in [3,4] else None, 
                        nudge_gt=gt_idxBl[si],
                        nudge_mask=mask,
                        # nudge_alphas=[6,4,3,4,6,6,6,6,6,6,6,6,6,6],
nudge_alphas=nudge_alphas,
                        # control_nudge=edges_idx[si],
                        #     control_alphas=[4,4,1,1,1,1,0,0,0,0],
                            # save_attn_maps=True,
                        # cfg= [4,4,4,4,4,4,4,6,6,6,6,6,6,6,6],
                        smooth_start_si=smooth_start_si,
                        turn_on_cfg_start_si=turn_on_cfg_start_si,
                        turn_off_cfg_start_si=turn_off_cfg_start_si,
                        last_scale_temp=last_scale_temp
                                    )
            f_hat = tmp['f_hat'].clone()
            # f_hat = vae.img_to_fhat_single(image_B3HW)
            fhats.append(f_hat.clone())
            # selfAttenMaps = tmp['selfAttenMaps']
            # crossAttenMaps = tmp['crossAttenMaps']
            # if self.visualize:
            #     save_data_batch(crossAttenMaps, f'output/crossAttenMaps_{si}', grid=[5,6])
            # # maps['self'].append(selfAttenMaps)
            # maps['cross'].append(crossAttenMaps)
        rec_img = vae.fhat_to_img(f_hat)
        # edges, _ = canny(rec_img.to(torch.float32))
        # edges = edges.to(self.dtype)
        # edges = edges.repeat(1,3, 1, 1)
        # fhat_edges1 = vae.img_to_fhat(edges)
        # fhats.append(fhat_edges1[-1].clone())
        # fhats.append(fhat_edges.clone())


        
        # f_hat = gt_fhat[2].clone()
        # for si in [3,4,5,6,7,8,9]:
        #     with torch.no_grad():
        #         tmp = self.invert_step(
        #                 image_B3HW, 
        #                 eprompt,
        #                 step=si,
        #                 f_hat=  f_hat,
        #                 # replace_cross_map= maps['cross'][si-3],
        #                 replace_self_map= maps['self'][si-3],
        #                 # noise_map=nm[si-4] if si in [4,5] else None,
        #                             )
        #     f_hat = tmp['f_hat'].clone()
        #     fhats.append(f_hat.clone())
        #     h_BChw = tmp['h_BChwres']
            # save_data_batch(crossAttenMaps, f'output/visuals/p2p_only_prompt/crossAttenMaps_{si}', grid=[5,6])

        fimg = vae.fhat_to_img(f_hat)
        final = vae.quantize.iterative_soft_project(
            gt['f_BChw'],
            f_hat,
            mask=mask
        )
        if include_residual:
            fhats.append(final)


        fhats = self.fhats_to_img(fhats)
        return {
            'fhats': fhats, 
            'h_BChw': h_BChw,
        }


    def logits_to_tokens(self, logits_BlV,
        top_k: int = 400,
        top_p: float = 0.95,
        si: int = 0,
        cfg: float = 6.,
        apply_smooth: bool = True,
        smooth_start_si: int = 2,
        turn_on_cfg_start_si: int =2,
        turn_off_cfg_start_si: int = 8,
        last_scale_temp= .1,
        rng = torch.Generator(device='cuda'),
        B = 1,
        deterministic: bool = True,
        ):
        ratio = si / self.switti.num_stages_minus_1
        vae_quant = self.vae.quantize
        # if deterministic:
        #     print('deterministic sampling')
        #     rng.manual_seed(0)

        # Guidance
        if si < turn_on_cfg_start_si:
            logits_BlV = logits_BlV[:B]
        elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
            t = cfg * ratio
            print('cfg ratio', cfg, ratio)
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        elif last_scale_temp is not None:
            logits_BlV = logits_BlV / last_scale_temp
        if apply_smooth and si >= smooth_start_si:
            # not used when evaluating FID/IS/Precision/Recall
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
            idx_Bl = gumbel_softmax_with_rng(
                logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
            )
            # print('idx_Bl', idx_Bl)
            print('gumbel')

            # idx_Bl = gumbel_softmax_with_rng(
            h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
        else:
            # default nucleus sampling
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
            )[:, :, 0]
            h_BChw = SampleSTE.apply(
                logits_BlV,
                rng,
                top_k,
                top_p,
                vae_quant.embedding
            )           
            # h_BChw = vae_quant.embedding(idx_Bl)

        return {
            'idx_Bl': idx_Bl,               
            'h_BChw': h_BChw,
        }

