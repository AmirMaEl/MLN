import time
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
from adain import AdaIN
from tqdm import tqdm
import numpy as np
import torchvision.transforms as ts
import torch.nn as nn
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans as SkKMeans
import logging

logger = logging.getLogger(__name__)

try:
    from cuml.cluster import KMeans as CuKMeans  # type: ignore
    import cudf  # type: ignore
    import cupy  # type: ignore
    GPU_KMEANS_AVAILABLE = Tru11
except Exception as exc:  # pragma: no cover - optional dependency
    CuKMeans = None
    cudf = None
    cupy = None
    GPU_KMEANS_AVAILABLE = False
    print(
        "Falling back to scikit-learn KMeans because RAPIDS imports failed: %s",
        exc,
    )

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

TRAIN_IMAGE_SIZE = (512, 512)

def print(*args, **kwargs):
    pass
#     print(*args, **kwargs)
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
        # ensure prompt_pos exists even if _encode_prompt never sets it
        self.prompt_pos = None

        self.device = device
        self.begin_ends = []
        self.seed = 42
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
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: None or float = None,
        return_fhats = False,
        kv_caching: bool = True,
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
        start_time = time.time()
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
        f_hat = sos.new_zeros(B, switti .Cvae, switti.patch_nums[-1], switti.patch_nums[-1])
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
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
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
        finaltime = time.time() - start_time
        print('final generation time: ', finaltime)

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
    # torch.Size([1, 9, 4096])
# (Pdb) p nudge_mask.shape
# torch.Size([1, 9])
# (Pdb) p gt_tokens.shape
# torch.Size([1, 9])
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
        nudge_mask=None,
        cfg_in_mask=None
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
        rng.manual_seed(self.seed)

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
            print('cfg t at scale %i: '%si, t)
            logits_BlV = (1 + t) * logits_cond - t * logits_uncond
        # elif si >= turn_off_cfg_start_si:
        #     if nudge_gt is not None:
        #         logits_BlV= self.precondition_logits_nudge(
        #         logits_pred=logits_BlV,
        #         gt_tokens=nudge_gt,
        #         alpha=nudge_alpha,
        #         nudge_mask=nudge_mask
        #     )
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
        no_prompt = ['a</w>','with</w>','cup</w>','of</w>','on</w>',
                     'photo</w>','cat</w>','standing</w>','rocks</w>','near</w>','the</w>','ocean</w>', 
                         '<|startoftext|>','<|endoftext|>',',</w>']

        if cross_attn_map:
            if self.prompt_pos is None: return None
            
            for i,word in enumerate(self.prompt_pos[1:]):
                if word in no_prompt:
                    continue
                attn_map = torch.cat(attn_mapin, dim=0)[:30]
                # all_attn_per_head = torch.mean(attn_map, dim=1)
                # all_attn_per_head = all_attn_per_head.permute(0, 2, 1)
                # all_attn_per_head = all_attn_per_head[:,i,:]
                # all_attn_per_head = all_attn_per_head.unsqueeze(1)/ all_attn_per_head.max()
                # all_attn_per_head = all_attn_per_head.reshape(-1,1, pn, pn)

                # save_data_batch(all_attn_per_head, './all_attn_per_head',grid=[5,6])
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


    @torch.inference_mode()
    def invert_step(
        self,
        image_B3HW,
        
                        econd_vector,
                        econtext,
                        econtext_attn_bias,      
        
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
        cfg_in_mask=None
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
        context = econtext
        cond_vector = econd_vector
        context_attn_bias = econtext_attn_bias
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
        cfg_in_mask=cfg_in_mask,
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
    def get_mask(self,
                 cond_vector,
                 econd_vector,
                 context,
                    econtext,
                 context_attn_bias,
                    econtext_attn_bias,
                 image_B3HW,smooth_start_si,
                 turn_on_cfg_start_si,turn_off_cfg_start_si,last_scale_temp,steps=[6],quantile=0.7):

        f_hat = self.vae.img_to_fhat(image_B3HW)[steps[0]-1]
        f_hatinit = f_hat.clone()
        maps = {'self': [], 'cross': [], 'crosse':[]}
        for si in steps:
            with torch.no_grad():
                tmp = self.invert_step(
                        image_B3HW, 
                        econd_vector,
                        econtext,
                        econtext_attn_bias,
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
                    cond_vector,
                        context,

                            context_attn_bias,
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


            # tmp1 = F.mse_loss(crosse, cross, reduction='none')
            tmp1 = torch.abs(crosse - cross)
            # tmp1 = self._normalize(crosse) * self._normalize(cross)

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
            save_data_batch(torch.cat([diff[-1],diffs[-1]]), f'output/diff_crossAttenMaps', grid=[5,6])
            save_data_batch((maps['cross'][1]), f'output/orig_crossAttenMaps', grid=[5,6])
            save_data_batch((maps['crosse'][1]), f'output/recon_crossAttenMaps', grid=[5,6])
        # self.get_lime(diff[-1], num_clusters=8)

        return diffs[-1].cuda()
    

    
    def _normalize(self,X):
        # per-map min-max to [0,1]
        eps = 1e-5
        mn = X.amin(dim=tuple(range(X.ndim-1)), keepdim=True) if X.ndim==3 else X.min()
        mx = X.amax(dim=tuple(range(X.ndim-1)), keepdim=True) if X.ndim==3 else X.max()
        return (X - mn) / (mx - mn + eps)



    def get_lime(self, maps, num_clusters=8):
        if maps.shape[0] >1:
            maps = torch.mean(maps, dim=0)
        attn = maps.reshape(-1, 1).cpu().numpy()

        out_res = 32

        arr = normalize(attn, axis=1, norm='l2')

        if GPU_KMEANS_AVAILABLE and cudf is not None and CuKMeans is not None:
            cudf_arr = cudf.DataFrame(arr)
            kmeans_model = CuKMeans(n_clusters=num_clusters, n_init=1)
            kmeans_model.fit(cudf_arr)
            labels = kmeans_model.labels_.values.astype(np.uint8).get()
        else:
            kmeans_model = SkKMeans(n_clusters=num_clusters, n_init=10, random_state=0)
            kmeans_model.fit(arr)
            labels = kmeans_model.labels_.astype(np.uint8)
        labels_spatial = labels.reshape(32,32)
        labels_spatial = cv2.resize(labels_spatial, dsize=(out_res, out_res), interpolation=cv2.INTER_NEAREST)
        labels_spatial = torch.from_numpy(labels_spatial).float()
        labels_spatial = labels_spatial/num_clusters
        save_data_batch(torch.cat([maps.unsqueeze(0).cpu(), labels_spatial.unsqueeze(0).unsqueeze(0)], dim=0), f'output/lime_segmentation', grid=[1,2])

        # plot_array("Segmentation Map", labels_spatial, cmap='viridis')  # requires matplotlib

        return labels_spatial
    

    def invert_stepwise(
        self,
        image_B3HW,
        prompt: str or list[str] = "",
        ref_img = None,
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
        last_scale_temp: float = .1,
        gt_fix_scales: int = 2,
        visualize: bool = True,
        mask = None,
        include_residual: bool = False,
        nudge_alphas = [6, 6, 6, 6, 6, 4, 2, 0, 0, 0],
        quant: float = 0.7,
    ):
        self.visualize = visualize
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        image_B3HW = image_B3HW.to(self.dtype)

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, eprompt)
        econtext, econd_vector, econtext_attn_bias = self.encode_prompt(eprompt, prompt)
        B = context.shape[0] // 2
        cond_vector = switti.text_pooler(cond_vector)
        econd_vector = switti.text_pooler(econd_vector)
        crop_coords = get_crop_condition(
            2 * B * [TRAIN_IMAGE_SIZE[0]], 2 * B * [TRAIN_IMAGE_SIZE[1]]
        ).to(cond_vector.device)
        crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
        crop_cond = switti.crop_proj(crop_embed)

        gt = vae.img_to_f(image_B3HW)
        gt_fhat = gt['fhats']
        gt_idxBl = gt['idx_list']

        # GT-seed: accumulate f_hat using GT tokens for scales 0..(gt_fix_scales-1)
        # so the transformer at higher scales has the right coarse context.
        f_hat = cond_vector.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])
        with torch.no_grad():
            for si in range(gt_fix_scales):
                idx = gt_idxBl[si]
                pn = switti.patch_nums[si]
                h_BChw = vae_quant.embedding(idx)
                h_BChw = h_BChw.transpose(1, 2).reshape(B, switti.Cvae, pn, pn)
                _, f_hat, _ = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw
                )

        fhats = [f_hat.clone()]

        # Compute attention-based edit mask from cross-attention difference.
        # turn_off=7 is hardcoded (matching original) so mask steps always run
        # without CFG — gives deterministic sharp attention maps for clean masking.
        if mask is None:
            mask = self.get_mask(
                cond_vector, econd_vector,
                context, econtext,
                context_attn_bias, econtext_attn_bias,
                image_B3HW, smooth_start_si,
                turn_on_cfg_start_si, 7,
                last_scale_temp, steps=[7, 8, 9], quantile=quant,
            )

        self.seed = seed
        start_time = time.time()

        # Run transformer from scale gt_fix_scales to the end.
        # nudge_alphas control per-scale nudge strength toward GT tokens:
        # high at lower scales preserves structure, lower/zero at fine scales
        # lets CFG freely generate target-prompt features.
        for si in range(gt_fix_scales, len(switti.patch_nums)):
            with torch.no_grad():
                tmp = self.invert_step(
                    image_B3HW,
                    econd_vector, econtext, econtext_attn_bias,
                    step=si,
                    f_hat=f_hat,
                    nudge_gt=gt_idxBl[si],
                    nudge_alphas=nudge_alphas,
                    nudge_mask=mask,
                    cfg=cfg,
                    smooth_start_si=smooth_start_si,
                    turn_on_cfg_start_si=turn_on_cfg_start_si,
                    turn_off_cfg_start_si=turn_off_cfg_start_si,
                    last_scale_temp=last_scale_temp,
                    seed=seed,
                )
            f_hat = tmp['f_hat'].clone()
            fhats.append(f_hat.clone())

        if include_residual:
            final = vae.quantize.iterative_soft_project(
                gt['f_BChw'], f_hat, mask=mask, max_iter=8, tau=.2
            )
            fhats.append(final)

        print('editing time: %.1fs' % (time.time() - start_time))

        fhats = self.fhats_to_img(fhats)
        fs = [f.to(torch.float32) for f in fhats]
        fhats = torch.stack(fs, dim=0)
        return {'fhats': fhats}

    def get_noise_maps_control(self,ref_img, img, prompt, steps=[8], steps_per_loop=50000):
        nmaps = []
        gt_fhatsr = self.vae.img_to_fhat(ref_img)
        gt_fhats = self.vae.img_to_fhat(img)
        gt_idxBl = self.vae.img_to_idxBl(img)
        import kornia
        canny = kornia.filters.Canny(low_threshold=0.002, high_threshold=0.04)
        edges, _ = canny(ref_img)
        fhat_edges = self.vae.img_to_fhat(edges.repeat(1,3, 1, 1))[-1]
        edges_idx = self.vae.img_to_idxBl(edges.repeat(1,3, 1, 1))
        f_hat = gt_fhats[3].clone()
        for i in [3,4,5,6,7,8]:
            with torch.no_grad():
                tmp = self.invert_step(
                    img,
                    prompt,
                    step=i,
                    f_hat=f_hat,
                    control_nudge=edges_idx[j],
                    control_alphas=[0,0,4,4,4,4,2,1,1,0.1],
                    smooth_start_si=2,
                    turn_on_cfg_start_si=2,
                    turn_off_cfg_start_si=11,
                    last_scale_temp=0.1,

                )
            f_hat = tmp['f_hat'].clone()

        noise_map = torch.zeros(1, self.switti.patch_nums[9]**2, 4096).to('cuda')
        noise_map.requires_grad = True
        optim = torch.optim.AdamW([noise_map], lr=1e-1)
        pbar = tqdm(range(steps_per_loop), desc='opt step')
        optim.zero_grad()
        tmp = self.invert_step(
                        img,
                        prompt=prompt,
                        step=i,
                        f_hat=f_hat.clone().detach(),
                        return_logit=True,
                    control_nudge=edges_idx[j],
                    control_alphas=[0,0,4,4,4,4,2,1,1,noise_map],
                            smooth_start_si=2,
    turn_on_cfg_start_si=2,
    turn_off_cfg_start_si=11,
    last_scale_temp=0.1,




                            )
        breakpoint()
        pred = tmp['logits']


        # optim.zero_grad()
        # loss.backward()
        # optim.step()
        # pbar.set_postfix({'loss': loss.item(), 'max':noise_map.max().item(), 'min': noise_map.min().item()})
        # if loss.item() < .1:              
        #     break
        # old_noise_map = noise_map.clone().detach()
        # nmaps.append(old_noise_map)
    # return nmaps


    def train_masknet(self, steps=[7,8], steps_per_loop=500):
        self.mask_nets = [
            MaskNet(cond_channels=1920, codebook_size=4096),
        ]
        for mask_net in self.mask_nets:
            mask_net.train()
            optimizer = torch.optim.Adam(mask_net.parameters(), lr=1e-4)
            raise NotImplementedError("train_masknet requires a custom DataLoader — supply one here")
            dl = None  # replace with your DataLoader
            pbar = tqdm(dl, desc='opt step')
            prompt = ['' for _ in range(8)]
            for b in pbar:
                img = b['image'].to('cuda')
                inp = self.vae.img_to_idxBl(img)
                fhs = self.vae.idxBl_to_fhat(inp, same_shape=True)
                cond = b['cond_canny'].repeat(1,3,1,1).to('cuda')
                cond = self.vae.img_to_idxBl(cond)
                tmp = self.invert_step(
                    img,
                    prompt,
                    step=8,
                    # f_hat=fhs[7],
                    return_logit=True,
                )
                f_hat = tmp['f_hat'].clone()
                logits = tmp['logits']

                breakpoint()
                mask = mask_net(logits, cond[8])
                loss = F.binary_cross_entropy(mask, img)  # Example loss
                loss.backward()
                optimizer.step()

    def get_global_noise_maps(self, img, prompt, steps=[7,8], steps_per_loop=500):
        nmaps = []
        # self.vae.to_cpu()
        # gt_fhats = self.vae.img_to_fhat(img)
        def sample(idxs,cond):
            fhats = []
            f_hat = self.vae.idxBl_to_fhat(idxs, same_shape=True)[0]
            fhats.append(f_hat.clone())
            for si in [1,2,3,4,5,6,7,8,9]:
                tmp = self.invert_step(
                    img,
                    prompt,
                    step=si,
                    f_hat=f_hat,
                    # noise_map=cond if si in [1,2,4,3] else None,
                )
                f_hat = tmp['f_hat'].clone()
                # pred = self.logits_to_tokens(pred)
                # pred = pred['idx_Bl']
                fhats.append(f_hat.clone())
            f_hat = self.fhats_to_img(fhats).squeeze()
            save_data_batch(f_hat, f'output/visual_fhat', grid=[5,6])
            return f_hat
            
        def get_canny(ig):
            ig = ts.ToPILImage()(ig.squeeze())
            ig = np.array(ig)
            edges = cv2.Canny(ig,100,200)
            edges = ts.ToTensor()(edges).unsqueeze(0).repeat(1,3,1,1)
            edges = (edges*2)-1
            return edges.to('cuda'  )

                    
        gt_idxBl = self.vae.img_to_idxBl(img)
        gt_edges = self.vae.img_to_idxBl(get_canny(img))
        # gt_Bledges = torch.cat(gt_edges, dim=1)[:]

        
        (prompt_embeds,
        pooled_prompt_embeds,
        prompt_attn_bias,
            ) = self.encode_prompt(prompt, encode_null=False)
        breakpoint()    

        gt_BL = torch.cat(gt_idxBl, dim=1)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_switti_input(gt_idxBl)[:,:65]
        torch.cuda.empty_cache()
        cond = torch.zeros(1,1920).to('cuda')
        cond = cond.requires_grad_(True)
        optim = torch.optim.AdamW([cond], lr=1e-1)
        pbar = tqdm(range(steps_per_loop), desc='opt step')
        for j in pbar:
            logits_BLV = self.switti.forward_control(
                    x_BLCv_wo_first_l,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_attn_bias=prompt_attn_bias,
                    batch_width= [512],
                    batch_height= [512],
                    L=66,
                    cond=cond
           )
            pred = torch.argmax(logits_BLV, dim=-1)
            p1 = pred[:,:1]
            p2 = pred[:,1:5]
            p3 = pred[:,5:14]
            p4 = pred[:, 14:30]   
            p5 = pred[:, 30:66]
            gt_idxBl[0]= p1
            gt_idxBl[1]= p2
            gt_idxBl[2]= p3
            gt_idxBl[3]= p4
            gt_BL = gt_BL[:, :66]
            loss = F.cross_entropy(
                logits_BLV.view(-1, 4096),
                gt_BL.view(-1)
            )
            tmp = self.vae.idxBl_to_img(gt_idxBl, same_shape=True)
            fs = self.vae.idxBl_to_fhat(gt_idxBl, same_shape=True)
            tmp = torch.cat(tmp, dim=0)
            if loss.item() < 1:
                breakpoint()
                save_data_batch(tmp, f'output/visual1',grid=[5,6])
                sample(gt_idxBl, cond.detach().clone())
                breakpoint()
            optim.zero_grad()
            loss.backward()
            pbar.set_postfix({'loss': loss.item()})
            optim.step()
    
    
    def get_noise_maps(self,ref_img, img, prompt, steps=[8], steps_per_loop=50000):
        def logits_to_img(logits,f_hat,grtr):
            """
            logits: (B, T, vocab_size) VAR output for a scale
            codebook: (vocab_size, embed_dim) VQ-VAE codebook
            decoder: nn.Module mapping (B, embed_dim, H, W) to (B, 3, H, W) in [0,1]
            H, W: spatial resolution of tokens at this scale
            """
            codebook = self.vae.quantize.embedding.weight  # (V, embed_dim)
            B, T, V = logits.shape
            # Soft decoding (weighted codebook embedding)
            # probs = F.softmax(logits, dim=-1)                   # (B, T, V)
            rng = torch.Generator(device='cuda')

            hard_pred = sample_with_top_k_top_p_(
                logits, top_k=400, top_p=0.95, num_samples=1,rng=rng,
            )[:, :, 0]

            probs = F.gumbel_softmax(logits, tau=0.1, hard=False)  # or straight-through
            # probs = F.softmax(logits, dim=-1)                   # (B, T, V)

            H = W = round(probs.shape[1]**0.5)
            si = self.switti.patch_nums.index(H)+1
            SN = len(self.switti.patch_nums)
            f_hat_init = f_hat.clone().detach()

            emb = torch.matmul(probs, codebook)                 # (B, T, embed_dim)
            emb_2d = emb .view(B, H, W, -1).permute(0, 3, 1, 2)   # (B, C, H, W)
            _,f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
            si,SN, f_hat_init, emb_2d,
        )   
            with torch.no_grad():
                h_BChw = self.vae.quantize.embedding(hard_pred)
                h_BChw = h_BChw.transpose(1, 2).reshape(B, self.switti.Cvae, H, W)


                _,f_hat_hard ,_ = self.vae.quantize.get_next_autoregressive_input(
                    si,SN, f_hat_init, h_BChw
                )


            img,imginit,imghard,grtr = self.fhats_to_img([f_hat, f_hat_init, f_hat_hard,grtr])

            tmp = torch.cat([imginit, img, imghard, grtr], dim=0)
            save_data_batch(tmp, f'output/visualimg',grid=[2,3])
            breakpoint()
            loss = F.mse_loss(img, grtr)
            loss.backward()
            return loss

        def logits_to_edges(logits,f_hat,grtr):
            """
            logits: (B, T, vocab_size) VAR output for a scale
            codebook: (vocab_size, embed_dim) VQ-VAE codebook
            decoder: nn.Module mapping (B, embed_dim, H, W) to (B, 3, H, W) in [0,1]
            H, W: spatial resolution of tokens at this scale
            """
            codebook = self.vae.quantize.embedding.weight  # (V, embed_dim)
            B, T, V = logits.shape
            # Soft decoding (weighted codebook embedding)
            # probs = F.softmax(logits, dim=-1)                   # (B, T, V)
            rng = torch.Generator(device='cuda')

            hard_pred = sample_with_top_k_top_p_(
                logits, top_k=400, top_p=0.95, num_samples=1,rng=rng,
            )[:, :, 0]

            probs = F.gumbel_softmax(logits, tau=0.1, hard=False)  # or straight-through
            # probs = F.softmax(logits, dim=-1)                   # (B, T, V)

            H = W = round(probs.shape[1]**0.5)
            si = self.switti.patch_nums.index(H)+1
            SN = len(self.switti.patch_nums)
            f_hat_init = f_hat.clone().detach()

            emb = torch.matmul(probs, codebook)                 # (B, T, embed_dim)
            emb_2d = emb .view(B, H, W, -1).permute(0, 3, 1, 2)   # (B, C, H, W)
            _,f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
            si,SN, f_hat, emb_2d,
        )   
            with torch.no_grad():
                h_BChw = self.vae.quantize.embedding(hard_pred)
                h_BChw = h_BChw.transpose(1, 2).reshape(B, self.switti.Cvae, H, W)


                _,f_hat_hard ,_ = self.vae.quantize.get_next_autoregressive_input(
                    si,SN, f_hat_init, h_BChw
                )


            img,imginit,imghard,grtr = self.fhats_to_img([f_hat, f_hat_init, f_hat_hard,grtr])

            # Differentiable Canny
            import kornia
            canny = kornia.filters.Canny(low_threshold=0.002, high_threshold=0.04)
            edges, _ = canny(img)
            edgesgt, _ = canny(grtr)
            # edgesgt = (edgesgt*2)-1
            # edges=(edges*2)-1
            tmp = torch.cat([imginit, img, edges.repeat(1, 3,1,1),imghard,grtr, edgesgt.repeat(1, 3,1,1)], dim=0)
            save_data_batch(tmp, f'output/visualedges',grid=[2,3])
            breakpoint()
            loss = F.mse_loss(edges, edgesgt)*100
            loss.backward()
            return loss


        nmaps = []
        gt_fhatsr = self.vae.img_to_fhat(ref_img)
        gt_fhats = self.vae.img_to_fhat(img)
        gt_idxBl = self.vae.img_to_idxBl(img)
        
        old_noise_map = None
        for i in (steps):
            f_hat = gt_fhats[steps[i-steps[0]]-1]
            gt = gt_idxBl[i]
            # noise_map = torch.randn(1, 1920).to('cuda')
            # noise_map = torch.zeros(1,30,self.switti.patch_nums[i]**2, self.switti.patch_nums[i]**2).to('cuda')
            # noise_map = torch.zeros_like(f_hat).to('cuda')
            noise_map = torch.zeros(1, self.switti.patch_nums[i]**2, 1920).to('cuda')
            # if old_noise_map is None else F.interpolate(
            #     old_noise_map.unsqueeze(0),
            #     size=(self.switti.patch_nums[i]**2, 1920),
            #     mode='area',
            # ).squeeze(0)
            # noise_map = f_hat.clone().detach()
            noise_map.requires_grad = True
            optim = torch.optim.AdamW([noise_map], lr=1e-1)
            pbar = tqdm(range(steps_per_loop), desc='opt step')
            for j in pbar:
                optim.zero_grad()
                tmp = self.invert_step(
                            img,
                            prompt=prompt,
                            step=i,
                            f_hat=f_hat.clone().detach(),
                            return_logit=True,
                            # noise_map=noise_map,]
                            preconditioner=noise_map,
                            # cfg=1
                            )
                pred = tmp['logits']
                # loss =logits_to_img(pred, f_hat.clone().detach(), grtr=gt_fhats[steps[i-steps[0]]])
                # loss = logits_to_edges(pred, f_hat.clone().detach(),grtr=gt_fhatsr[steps[i-steps[0]]])
                # optim.step()
                # pbar.set_postfix({'loss': loss.item()})         

                # pred =self.logits_to_tokens(pred)
                # pred = pred['idx_Bl']

        #         _,f_hat, next_token_map = self.vae.quantize.get_next_autoregressive_input(
        #             steps[0]-1, len(self.switti.patch_nums), f_hat, pred,
        # )
                # loss = F.l1_loss(f_hat,gt)

                loss = F.cross_entropy(pred.view(-1,4096), gt.view(-1))#+ noise_map.abs().max()
                # # loss = self.ste_loss(pred, f_hat, f_hat)

                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_postfix({'loss': loss.item(), 'max':noise_map.max().item(), 'min': noise_map.min().item()})
                if loss.item() < .1:              
                    break
            old_noise_map = noise_map.clone().detach()
            nmaps.append(old_noise_map)
        return nmaps





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

