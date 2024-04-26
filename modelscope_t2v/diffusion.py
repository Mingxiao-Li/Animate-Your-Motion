# Part of the implementation is borrowed and modified from latent-diffusion,
# publicly avaialbe at https://github.com/CompVis/latent-diffusion.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import torch
from tqdm import tqdm
from einops import rearrange
import copy
__all__ = ['GaussianDiffusion', 'beta_schedule']


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    tensor = tensor.to(x.device)
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)


def beta_schedule(schedule,
                  num_timesteps=1000,
                  init_beta=None,
                  last_beta=None):
    if schedule == 'linear_sd':
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')


class GaussianDiffusion(object):
    r""" Diffusion Model for DDIM.
    "Denoising diffusion implicit models." by Song, Jiaming, Chenlin Meng, and Stefano Ermon.
    See https://arxiv.org/abs/2010.02502
    """

    def __init__(self,
                 betas,
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon=1e-12,
                 rescale_timesteps=False):
        # check input
        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)
        assert min(betas) > 0 and max(betas) <= 1
        assert mean_type in ['x0', 'x_{t-1}', 'eps']
        assert var_type in [
            'learned', 'learned_range', 'fixed_large', 'fixed_small'
        ]
        assert loss_type in [
            'mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1',
            'charbonnier'
        ]
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.epsilon = epsilon
        self.rescale_timesteps = rescale_timesteps

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:],
             alphas.new_zeros([1])])

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0
                                                        - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0
                                                      - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod
                                                      - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (
            1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(
            self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (
                1.0 - self.alphas_cumprod)

    def p_mean_variance(self,
                        xt,
                        t,
                        model,
                        model_kwargs={},
                        clamp=None,
                        percentile=None,
                        guide_scale=None,
                        obj_oriented_attn_ratio=0.0,
                        control_feat=None,
                        img_free_guidance=0.0,
                        ti_free_guidance=0.0,
                        ):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs,
                        obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=control_feat)
            if img_free_guidance != 0.0:
                f_out = model(xt, self._scale_timesteps(t), **model_kwargs,
                        obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=torch.zeros_like(control_feat).to(control_feat.device))
                dim = y_out.size(1) if self.var_type.startswith(
                'fixed') else y_out.size(1) // 2
                a = f_out[:, :dim]
                b = img_free_guidance * (out[:, :dim] - f_out[:,:dim])
                c = out[:, dim:]
                out = torch.cat([a+b, c], dim=1)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0],
                          obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=copy.deepcopy(control_feat))
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1],
                          obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=copy.deepcopy(control_feat))
            dim = y_out.size(1) if self.var_type.startswith(
                'fixed') else y_out.size(1) // 2
            if img_free_guidance != 0.0:
                if control_feat is not None:
                    zero_control = [feat*0 for feat in copy.deepcopy(control_feat)]
                else:
                    zero_control = control_feat
                zero_xt = copy.deepcopy(xt)
                zero_xt[:,4:,:,:] = zero_xt[:,4:,:,:]*0
                f_out = model(zero_xt, self._scale_timesteps(t), **model_kwargs[0],
                          obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=zero_control)
                f1 = f_out[:, :dim]
                f = f1 + img_free_guidance * (y_out[:,:dim]-f_out[:, :dim])
                #f = img_free_guidance * f_out[:, :dim]
            
            if ti_free_guidance != 0.0:
                if control_feat is not None:
                    zero_control = [feat*0 for feat in copy.deepcopy(control_feat)]
                else:
                    zero_control = control_feat
                zero_xt = copy.deepcopy(xt)
                zero_xt[:,4:,:,:] = zero_xt[:,4:,:,:]*0
                i_out = model(xt, self._scale_timesteps(t), **model_kwargs[1],
                          obj_oriented_attn_ratio=obj_oriented_attn_ratio, control_feat=zero_control)
                i1 = i_out[:, :dim]
                i = i1 + ti_free_guidance * (y_out[:, :dim]-i_out[:,:dim])
                #i = ti_free_guidance * i_out[:,:dim]
  
            a = u_out[:, :dim]
            b = guide_scale * (y_out[:, :dim] - u_out[:, :dim])
            c = y_out[:, dim:]
            
            guidance = a + b
            if img_free_guidance != 0.0:
                guidance += f 
            if ti_free_guidance != 0.0:
                guidance += i1 + i 
            out = torch.cat([guidance, c], dim=1)
        
        # compute variance
        if xt.shape != out.shape:
            xt = xt[:, :dim]
        if self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        # compute mean and x0
        if self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(
                self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(
                x0.flatten(1).abs(), percentile,
                dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(
            self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var

    @torch.no_grad()
    def ddim_sample(self,
                    xt,
                    t,
                    model,
                    model_kwargs={},
                    clamp=None,
                    percentile=None,
                    condition_fn=None,
                    guide_scale=None,
                    ddim_timesteps=20,
                    eta=0.0,
                    obj_oriented_attn_ratio=0.0,
                    control_feat=None,
                    img_free_guidance=0.0,
                    ti_free_guidance=0.0
                    ):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp,
                                           percentile, guide_scale, obj_oriented_attn_ratio, control_feat,img_free_guidance,ti_free_guidance)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(
                self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(
                xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(
                self.sqrt_recipm1_alphas_cumprod, t, xt) * eps

        # derive variables
        if xt.shape != x0.shape:
            xt = xt[:, :x0.shape[1]]
            
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(
            self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        a = (1 - alphas_prev) / (1 - alphas)
        b = (1 - alphas / alphas_prev)
        sigmas = eta * torch.sqrt(a * b)

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas**2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0

    @torch.no_grad()
    def ddim_sample_loop(self,
                         noise,
                         model,
                         controlnet_encoder,
                         model_kwargs={},
                         clamp=None,
                         percentile=None,
                         condition_fn=None,
                         guide_scale=None,
                         ddim_timesteps=20,
                         eta=0.0,
                         first_frame_cond=None,
                         obj_oriented_attn_ratio=0.0,
                         img_free_guidance = 0.0,
                         ti_free_guidance = 0.0,
                         cat_init=False
                         ):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps,
                                  self.num_timesteps // ddim_timesteps)).clamp(
                                      0, self.num_timesteps - 1).flip(0)

        control_feat = None
        for step in tqdm(steps, total=ddim_timesteps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            if controlnet_encoder is not None:
                assert first_frame_cond is not None, "we need first frame for control net"
                first_latent = rearrange(first_frame_cond, "b c f h w -> b f c h w")[:, 0, :, :]
                control_feat = controlnet_encoder(xt, first_latent, t, model_kwargs[0]['y'])
            if cat_init:
                xt = torch.cat((xt, first_frame_cond), dim=1)
            xt, _ = self.ddim_sample(xt, t, model, model_kwargs, clamp,
                                     percentile, condition_fn, guide_scale,
                                     ddim_timesteps, eta, obj_oriented_attn_ratio, control_feat, img_free_guidance, ti_free_guidance)
        return xt

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t