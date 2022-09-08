import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import linear_beta_schedule, cosine_beta_schedule, default, \
    extract, ModelPrediction, unnormalize_to_zero_to_one, normalize_to_neg_one_to_one


class ProgressiveDistillationGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        loss_type = 'l2',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        clip_denoised=False,
        jumpsched=2,     # if integer, then specifies jump sizes for all phases, if list, specifies jump sizes for each phase
    ):
        super().__init__()
        assert not (type(self) == ProgressiveDistillationGaussianDiffusion and model.channels != model.out_dim)

        self.model = model
        self.ema = None #EMA(self.model, beta=0.99, power=3/4, update_every=1, update_after_step=1000)
        self.teacher = None
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size
        self.clip_denoised = clip_denoised

        self.jumpsched = jumpsched if not isinstance(jumpsched, int) \
            else [jumpsched**k for k in range(0, int(math.ceil(math.log(timesteps)/math.log(jumpsched)))+1)]

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # jump size
        self.register_buffer('distillphase', torch.tensor([0]).long())

    def get_jump_size(self):
        phase = self.distillphase[0].cpu().item()
        ret = self.jumpsched[phase]
        return ret

    def get_prev_jump_size(self):
        phase = self.distillphase[0].cpu().item()
        ret = self.jumpsched[phase - 1] if phase >= 0 else None
        return ret

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, useteacher=False):
        model = self.model if not useteacher else self.teacher
        model_output = model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = None):
        """
        :param x:   input at timestep t: x_t
        :param t:   timestep t: integer tensor
        :param x_self_cond:
        :param clip_denoised:
        :return:
            model_mean: mean of the posterior distribution (gaussian)
            posterior_variance:
            posterior_log_variance:
            x_start: predicted x_0
        """
        clip_denoised = self.clip_denoised if clip_denoised is None else clip_denoised
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, x_T=None, clip_denoised=None, return_trajectories=False):
        clip_denoised = self.clip_denoised if clip_denoised is None else clip_denoised
        batch, device, total_timesteps = shape[0], self.betas.device, self.num_timesteps
        jumpsize = self.get_jump_size()

        imgacc = []
        x0acc = []

        times = torch.linspace(-1, total_timesteps-1, steps=total_timesteps//jumpsize+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device) if x_T is None else x_T

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            alpha_next = self.alphas_cumprod_prev[time_next+1]
            img = x_start * alpha_next.sqrt() + (1 - alpha_next).sqrt() * pred_noise

            #imgacc.append(unnormalize_to_zero_to_one(img))
            imgacc.append(img)
            x0acc.append(unnormalize_to_zero_to_one(x_start))

        img = unnormalize_to_zero_to_one(img)

        if return_trajectories:
            return img, times, imgacc, x0acc
        else:
            return img

    @torch.no_grad()
    def sample(self, batch_size = 16, x_T=None, return_trajectories=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), x_T=x_T, return_trajectories=return_trajectories)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None, clip_denoised=None):
        clip_denoised = self.clip_denoised if clip_denoised is None else clip_denoised
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        jumpsize = self.get_jump_size()
        # noise sample

        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x_t, t)

        if self.teacher is None:        # then pretrain teacher on original timesteps
            if self.objective == 'pred_noise':
                target = noise
                lossweights = 1.        # simplified objective in DDPM paper
            elif self.objective == 'pred_x0':
                target = x_start
                # !!! When predicting x0 instead, the simplified objective from DDPM
                # would correspond to weighing the loss by \alpha / (1 - \alpha)
                # This gives lower weights to the beginning of the process.
                alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
                lossweights = (alpha_t / (1 - alpha_t)).clamp_min(1e-2)
                # lossweights = 1.
            else:
                raise ValueError(f'unknown objective {self.objective}')

        else:       # compute target x_start using the teacher
            # 1. Run DDIM sampler for two steps to get x_t-2
            prev_jump_size = self.get_prev_jump_size()
            with torch.no_grad():
                _t = t
                x_tmX = x_t

                while torch.any(_t > (t - jumpsize)):
                    pred_noise, x_start, *_ = self.model_predictions(x_tmX, _t, useteacher=True)

                    if clip_denoised:
                        x_start.clamp_(-1., 1.)
                        pred_noise = self.predict_noise_from_start(x_tmX, t, x_start)

                    _t = _t - prev_jump_size
                    alpha_next = extract(self.alphas_cumprod_prev, _t + 1, x_t.shape)   # +1 because we get alphas from alphas_cumprod_prev
                    x_tmX = x_start * alpha_next.sqrt() + (1 - alpha_next).sqrt() * pred_noise

                # 2. compute x*_0 from x_t-2 and x_t as the new target
                # x_tm2 = alpha_t-2.sqrt() * x*_0 + (1-alpha_t-2).sqrt() * pred_noise(x*_0)
                # x_tm2 = alpha_tm2.sqrt() * x_star_0 + (1-alpha_tm2).sqrt() * (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_star_0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                # x_tm2 = alpha_tm2 * x_star_0 + (1-alpha_tm2) * (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) - (1-alpha_tm2) * x_star_0 / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                # x_tm2 = (alpha_tm2 - (1-alpha_tm2) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)) * x_star_0 + (1-alpha_tm2) * (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
                # x_star_0 = (x_tm2 - (1-alpha_tm2) * x_t * (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))) \
                #            / (alpha_tm2 - (1-alpha_tm2) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
                sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
                sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                x_star_0 = (x_tmX - (1 - alpha_next).sqrt() * x_t * ( sqrt_recip_alphas_cumprod_t / sqrt_recipm1_alphas_cumprod_t)) \
                                / (alpha_next.sqrt() - (1-alpha_next).sqrt() / sqrt_recipm1_alphas_cumprod_t)

                # 3. check that x*_0 results in x_tm2 with the noise prediction equations
                # _pred_noise = self.predict_noise_from_start(x_t, t, x_star_0)
                # _x_tmX = x_star_0 * alpha_next.sqrt() + (1 - alpha_next).sqrt() * _pred_noise
                # # if not torch.allclose(_x_tm2, x_tm2):
                # assert torch.allclose(_x_tmX, x_tmX, atol=1e-6)

                target = x_star_0

            alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
            lossweights = (alpha_t / (1 - alpha_t)).clamp_min(1)
            # lossweights = (alpha_t / (1 - alpha_t)) + 1
            # lossweights = 1.

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * lossweights

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        jumpsize = self.get_jump_size()

        t = (torch.randint(0, self.num_timesteps//jumpsize, (b,), device=device).long() + 1) * jumpsize - 1

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

    def increase_jump_size(self):
        if self.ema is not None:        # initial denoising function is done using EMA because large losses at convergence
            self.teacher = copy.deepcopy(self.ema.ema_model)
            self.ema = None
        else:
            self.teacher = copy.deepcopy(self.model)
        self.distillphase[0] += 1

    def on_after_optim_step(self):
        if self.ema is not None:
            self.ema.update()

