import time
import torch
import torch.nn.functional as F

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
from mln_utils import save_data_batch

TRAIN_IMAGE_SIZE = (512, 512)


class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(self, switti, vae, text_encoder, text_encoder_2,
                 device, dtype=torch.float32, verbose=True):
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
        self.prompt_pos = None

        self.device = device
        self.begin_ends = []
        self.seed = 42
        cur = 0
        for pn in self.switti.patch_nums:
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,
                        torch_dtype=torch.bfloat16, device="cuda", reso=1024):
        switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)
        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype)

    def _encode_prompt(self, prompt):
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
        prompt_pos = self.text_encoder.tokenizer.convert_ids_to_tokens(
            self.text_encoder.tokenizer(prompt)['input_ids'][0]
        )
        if len(prompt_pos) > 4 and self.verbose:
            print('#' * 20)
            print(prompt_pos)
            print('#' * 20)
            self.prompt_pos = prompt_pos
        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(self, prompt, null_prompt="", encode_null=True):
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

    def fhats_to_img(self, fhats):
        fs = []
        for f in fhats:
            fs.append(self.vae.fhat_to_img(f))
        return torch.stack(fs, dim=0)

    def precondition_logits_nudge(self, logits_pred, gt_tokens, alpha=0.2, nudge_mask=None):
        if nudge_mask is not None:
            keepmask = nudge_mask.unsqueeze(-1).expand_as(logits_pred)
            editmask = torch.ones_like(keepmask).float() - keepmask
        else:
            editmask = 0
            keepmask = 1
        with torch.no_grad():
            p_gt = F.one_hot(gt_tokens, num_classes=logits_pred.size(-1)).float()
        p_pred = F.softmax(logits_pred, dim=-1)
        return logits_pred + editmask * 12 * (p_gt - p_pred) + keepmask * alpha * (p_gt - p_pred)

    def apply_control_nudge(self, logits_pred, control_logits, alpha=0.2, mask=None):
        with torch.no_grad():
            p_ctrl = F.one_hot(control_logits, num_classes=logits_pred.size(-1)).float()
        p_pred = F.softmax(logits_pred, dim=-1)
        delta = p_ctrl - p_pred
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
        full_context=False,
        freqs_cis=None,
        save_attn_maps=True,
        replace_cross_map=None,
        replace_self_map=None,
        return_logit=False,
        noise_map=None,
        preconditioner=None,
        nudge_gt=None,
        nudge_alpha=0.2,
        control_nudge=None,
        control_alpha=0.2,
        nudge_mask=None,
        cfg_in_mask=None,
    ):
        if isinstance(cfg, (list, tuple)):
            cfg = cfg[si]
        B = x_BLC.shape[0] // 2
        vae_quant = self.vae.quantize.to(x_BLC.dtype)
        crossAttenMaps = []
        v_self = []
        v_cross = []
        selfweights = []
        crossweights = []
        if freqs_cis is None:
            freqs_cis = switti.freqs_cis[:, : switti.levels[si + 1]]
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

        freqs_cis = freqs_cis.repeat(1, 2, 1)
        if nudge_mask is not None:
            nudge_mask = F.interpolate(
                nudge_mask,
                size=(self.switti.patch_nums[si], self.switti.patch_nums[si]),
                mode='bilinear',
            )
            nudge_mask = nudge_mask.view(1, -1)
        for b in switti.blocks:
            b.cross_attn.kv_caching(True)
        ratio = si / switti.num_stages_minus_1

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
            if 'cross_attn_map' in x_BLC:
                crossAttenMaps.append(x_BLC['cross_attn_map'])
            x_BLC = x_BLC['x']

        logits_BlV = switti.get_logits(x_BLC, cond_BD)
        ratio = si / switti.num_stages_minus_1

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
                    nudge_mask=nudge_mask,
                )
            if control_nudge is not None:
                logits_cond = self.apply_control_nudge(
                    logits_pred=logits_cond,
                    control_logits=control_nudge,
                    alpha=control_alpha,
                    mask=nudge_mask,
                )
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
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            idx_Bl = gumbel_softmax_with_rng(
                logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng, seed=self.seed,
            ).to(self.dtype)
            h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
        else:
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1, seed=self.seed,
            )[:, :, 0]
            h_BChw = vae_quant.embedding(idx_Bl)
        pn = switti.patch_nums[si]
        h_BChw = h_BChw.transpose(1, 2).reshape(B, switti.Cvae, pn, pn)

        h_BChw_res, f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
            si, len(switti.patch_nums), f_hat, h_BChw,
        )

        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
        if si != switti.num_stages_minus_1:
            next_token_map = (
                switti.word_embed(next_token_map)
                + lvl_pos[:, switti.levels[si + 1] : switti.levels[si + 2]]
            )
        else:
            next_token_map = (
                switti.word_embed(next_token_map)
                + lvl_pos[:, switti.levels[si] : switti.levels[si + 1]]
            )
        next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)
        return {
            'next_token_map': next_token_map,
            'h_BChwres': h_BChw_res,
            'h_BChw': h_BChw,
            'crossAttenMaps': self.process_attn_maps(crossAttenMaps, cross_attn_map=True, pn=pn),
            'v_self': v_self,
            'v_cross': v_cross,
            'f_hat': f_hat,
            'selfweights': selfweights,
            'crossweights': crossweights,
        }

    def process_attn_maps(self, attn_mapin, cross_attn_map=False, pn=4):
        self.attn_map = {}
        final_maps = []
        no_prompt = [
            'a</w>', 'with</w>', 'cup</w>', 'of</w>', 'on</w>',
            'photo</w>', 'cat</w>', 'standing</w>', 'rocks</w>', 'near</w>',
            'the</w>', 'ocean</w>', '<|startoftext|>', '<|endoftext|>', ',</w>',
        ]

        if cross_attn_map:
            if self.prompt_pos is None:
                return None
            for i, word in enumerate(self.prompt_pos[1:]):
                if word in no_prompt:
                    continue
                attn_map = torch.cat(attn_mapin, dim=0)[:30]
                attn_map = torch.mean(attn_map, dim=1)[3:27]
                attn_map = torch.mean(attn_map, dim=0).unsqueeze(0)
                attn_map = attn_map.permute(0, 2, 1)
                attn_map = attn_map[:, i, :]
                attn_map = attn_map.reshape(1, 1, pn, pn)
                final_maps.append(attn_map.detach())
            final_maps = torch.cat(final_maps, dim=0)
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
        null_prompt="",
        seed=None,
        cfg=6.,
        top_k=400,
        top_p=0.95,
        more_smooth=True,
        return_pil=True,
        smooth_start_si=2,
        turn_off_cfg_start_si=2,
        turn_on_cfg_start_si=5,
        cut_forward=True,
        last_scale_temp=.1,
        step=3,
        f_hat=None,
        save_attn_maps=False,
        replace_cross_map=None,
        replace_self_map=None,
        return_logit=False,
        noise_map=None,
        preconditioner=None,
        nudge_gt=None,
        nudge_alphas=None,
        nudge_mask=None,
        control_nudge=None,
        control_alphas=None,
        cfg_in_mask=None,
    ):
        seed = 5
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        switti.rng.manual_seed(seed)
        rng = switti.rng
        fh = f_hat
        nudge_alpha = nudge_alphas[step] if nudge_alphas is not None else 0.2
        control_alpha = control_alphas[step] if control_alphas is not None else 0.2
        context = econtext
        cond_BD = econd_vector
        context_attn_bias = econtext_attn_bias
        B = context.shape[0] // 2
        crop_coords = get_crop_condition(
            2 * B * [TRAIN_IMAGE_SIZE[0]], 2 * B * [TRAIN_IMAGE_SIZE[1]]
        ).to(cond_BD.device)
        crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
        crop_cond = switti.crop_proj(crop_embed)
        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        gt_fhat = vae.img_to_fhat(image_B3HW)
        gt_idxBl = vae.img_to_idxBl(image_B3HW)
        inp = gt_idxBl[step]
        h_BChw = vae_quant.embedding(inp)
        h_BChw = h_BChw.transpose(1, 2).reshape(
            B, switti.Cvae, switti.patch_nums[step], switti.patch_nums[step]
        )
        f_hat = F.interpolate(
            gt_fhat[step] if f_hat is None else f_hat.clone(),
            size=(switti.patch_nums[step], switti.patch_nums[step]),
            mode="area",
        )
        next_token_map = f_hat.view(B, switti.Cvae, -1).transpose(1, 2)
        next_token_map = (
            switti.word_embed(next_token_map)
            + lvl_pos[:, switti.levels[step] : switti.levels[step + 1]]
        )
        freqs_cis = switti.freqs_cis[:, switti.levels[step] : switti.levels[step + 1]]
        x_BLC = next_token_map.repeat(2, 1, 1)
        return self.single_step(
            x_BLC, step, switti, cond_BD,
            context, context_attn_bias, crop_cond,
            top_k, top_p, cfg, more_smooth, smooth_start_si,
            turn_on_cfg_start_si, turn_off_cfg_start_si, last_scale_temp,
            rng, gt_fhat[step] if fh is None else fh,
            full_context=False,
            freqs_cis=freqs_cis,
            save_attn_maps=save_attn_maps,
            replace_cross_map=replace_cross_map,
            replace_self_map=replace_self_map,
            return_logit=return_logit,
            preconditioner=preconditioner,
            nudge_gt=nudge_gt,
            nudge_alpha=nudge_alpha,
            nudge_mask=nudge_mask,
            control_nudge=control_nudge,
            control_alpha=control_alpha,
            cfg_in_mask=cfg_in_mask,
        )

    @torch.inference_mode()
    def get_mask(
        self,
        cond_vector, econd_vector,
        context, econtext,
        context_attn_bias, econtext_attn_bias,
        image_B3HW, smooth_start_si,
        turn_on_cfg_start_si, turn_off_cfg_start_si,
        last_scale_temp, steps=[6], quantile=0.7,
    ):
        f_hat = self.vae.img_to_fhat(image_B3HW)[steps[0] - 1]
        f_hatinit = f_hat.clone()
        maps = {'cross': [], 'crosse': []}
        for si in steps:
            with torch.no_grad():
                tmp = self.invert_step(
                    image_B3HW,
                    econd_vector, econtext, econtext_attn_bias,
                    step=si, f_hat=f_hat,
                    smooth_start_si=smooth_start_si,
                    turn_on_cfg_start_si=turn_on_cfg_start_si,
                    turn_off_cfg_start_si=turn_off_cfg_start_si,
                    last_scale_temp=last_scale_temp,
                )
                f_hat = tmp['f_hat'].clone()
                maps['cross'].append(tmp['crossAttenMaps'])
        f_hat = f_hatinit.clone()
        for si in steps:
            with torch.no_grad():
                tmp = self.invert_step(
                    image_B3HW,
                    cond_vector, context, context_attn_bias,
                    step=si, f_hat=f_hat,
                    smooth_start_si=smooth_start_si,
                    turn_on_cfg_start_si=turn_on_cfg_start_si,
                    turn_off_cfg_start_si=turn_off_cfg_start_si,
                    last_scale_temp=last_scale_temp,
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
                    pad = torch.zeros_like(crosse)
                    cross = torch.cat([cross, pad[cross.shape[0]:]], dim=0)
                else:
                    pad = torch.zeros_like(cross)
                    crosse = torch.cat([crosse, pad[crosse.shape[0]:]], dim=0)
            tmp1 = torch.abs(crosse - cross)
            tmp1 = tmp1 / tmp1.max()
            diff.append(tmp1)
        diffs = []
        for d in diff:
            tmp = torch.mean(d, dim=0)
            tmp = tmp / tmp.max()
            tmp = tmp.to(torch.float32)
            threshold = torch.quantile(tmp, quantile)
            tmp = tmp.to(self.dtype)
            tmp = torch.where(tmp > threshold, 1, torch.tensor(0.0, device=tmp.device))
            diffs.append(tmp.unsqueeze(0))
        if self.visualize:
            save_data_batch(torch.cat([diff[-1], diffs[-1]]), 'output/diff_crossAttenMaps', grid=[5, 6])
            save_data_batch(maps['cross'][1], 'output/orig_crossAttenMaps', grid=[5, 6])
            save_data_batch(maps['crosse'][1], 'output/recon_crossAttenMaps', grid=[5, 6])
        return diffs[-1].cuda()

    def invert_stepwise(
        self,
        image_B3HW,
        prompt="",
        ref_img=None,
        eprompt="",
        null_prompt="",
        seed=None,
        cfg=8.,
        top_k=400,
        top_p=0.95,
        more_smooth=True,
        return_pil=True,
        smooth_start_si=2,
        turn_on_cfg_start_si=8,
        turn_off_cfg_start_si=2,
        cut_forward=True,
        last_scale_temp=.1,
        gt_fix_scales=2,
        visualize=True,
        mask=None,
        include_residual=False,
        nudge_alphas=[6, 6, 6, 6, 6, 4, 2, 0, 0, 0],
        quant=0.7,
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
        fhats = torch.stack([f.to(torch.float32) for f in fhats], dim=0)
        return {'fhats': fhats}
