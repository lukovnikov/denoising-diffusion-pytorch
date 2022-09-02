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

# constants
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import exists, WeightStandardizedConv2d, Block, default, \
    ResnetBlock, LearnedSinusoidalPosEmb, SinusoidalPosEmb, PreNorm, LinearAttention, Residual, Downsample, Attention, \
    Upsample, GaussianDiffusion, unnormalize_to_zero_to_one, normalize_to_neg_one_to_one, linear_beta_schedule, \
    cosine_beta_schedule, extract

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# model
# gaussian diffusion trainer class

class GaussianDiffusionSurfGAN(nn.Module):
    def __init__(
        self,
        generators,
        discriminator,
        image_size,
            *,
        timesteps = 1000,
        jumps=8,
        loss_type = 'l2',
        beta_schedule = 'cosine',
    ):
        super().__init__()

        self.discriminator = discriminator
        self.generators = torch.nn.ModuleList(generators)
        self.channels = self.generators[0].channels
        self.self_condition = self.generators[0].self_condition

        self.image_size = image_size

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

        # sampling related parameters

        self.sampling_timesteps = self.num_jumps = jumps

        assert self.sampling_timesteps <= timesteps

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

        # register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.register_buffer('current_time', torch.tensor([-1], dtype=torch.long))
        self.current_generator = None

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

    def model_predictions(self, x, t, time=None):
        # select the right generator
        model, generator_i, _, _ = self.get_generator(time)
        model_output = model(x, t)

        pred_noise = self.predict_noise_from_start(x, t, model_output)
        x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = False, x_T=None, return_trajectories=False):
        batch, device, total_timesteps, sampling_timesteps \
            = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps

        imgacc = []
        x0acc = []

        times = torch.linspace(-1., total_timesteps-1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device) if x_T is None else x_T

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, time=time)

            if clip_denoised:
                x_start.clamp_(-1., 1.)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)

            if time_next > -1:
                alpha_next = self.alphas_cumprod[time_next]
                img = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()
            else:
                img = x_start

            if return_trajectories:
                imgacc.append(unnormalize_to_zero_to_one(img))
                x0acc.append(unnormalize_to_zero_to_one(x_start))

        img = unnormalize_to_zero_to_one(img)

        if return_trajectories:
            return img, times, imgacc, x0acc
        else:
            return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

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

    # HOW TO TRAIN THIS:
    # .forward_discriminator() returns the loss of the discriminator
    # .forward_generator() returns the loss for the current generator for a given image batch
    # .set_current_time() configures the current generator according to the given time step
    # - training sequence: start with time=0 and repeat the iteration:
    #       Phase I: train discriminator by calling .forward_discriminator() and backpropagating with a batch
    #       Phase II: train generator:
    #           1. call .set_current_time(t)  --> this makes sure self.current_generator is set
    #               if the returned value is None, the current generator is the same as it was
    #                                 ... is not None, then the generator changed and we have to create a new optimizer!
    #           2. call .forward_generator()    --> to train generator
    #           3. repeat steps 1 and 2 for a certain number of batches
    #       Set t <- t + 1 and repeat Phases I and II

    def forward_discriminator(self, img, time=None, clip_denoised=True):
        img = normalize_to_neg_one_to_one(img)
        img, img2 = torch.chunk(img, 2, 0)

        time = time if time is not None else self.current_time.item()
        # pick the right generator and train using discriminator
        _, _, jump_start_time, jump_size = self.get_generator(time)

        t = torch.full((len(img),), time, device=img.device).long()
        jump_start_t = torch.full((img.size(0),), jump_start_time, device=img.device).long()
        x_jump_start_time = self.q_sample(img, jump_start_t)

        with torch.no_grad():
            # FAKE
            # predict x0 using generator
            x0_current_G = self.current_generator(x_jump_start_time, jump_start_t)
            if clip_denoised:
                x0_current_G.clamp_(-1., 1.)

            # sample q(x_t | x_T, x_0) for the current time
            alpha_next = self.alphas_cumprod_prev[time]
            pred_noise = self.predict_noise_from_start(x_jump_start_time, jump_start_t, x0_current_G)
            generated = x0_current_G * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()
            # REAL
            noise = torch.randn_like(img2)
            real = self.q_sample(x_start=img2, t=t, noise=noise) if time > 0 else img2

        # run through discriminator
        discr_inp = torch.cat([generated, real], 0)
        discr_scores = self.discriminator(discr_inp, t)
        mult = torch.ones_like(discr_scores)
        mult[len(img):] = -1
        discr_scores *= mult

        loss = F.softplus(discr_scores)
        acc = (discr_scores < 0).float()
        return {"loss": loss.mean(), "acc": acc.mean()}

    def get_generator(self, time):
        jump_id = int(math.floor(self.num_jumps * time / self.num_timesteps))
        jump_start_time = int(self.num_timesteps * (jump_id + 1) / self.num_jumps) - 1
        jump_size = jump_start_time - time + 1
        return self.generators[jump_id], jump_id, jump_start_time, jump_size

    def forward_generator(self, img, time=None, clip_denoised=True):
        time = time if time is not None else self.current_time.item()
        # pick the right generator and train using discriminator
        _, _, jump_start_time, jump_size = self.get_generator(time)

        t = torch.full((img.size(0),), time, device=img.device).long()
        jump_start_t = torch.full((img.size(0),), jump_start_time, device=img.device).long()
        x_jump_start_time = self.q_sample(img, jump_start_t)

        if jump_size > 0:   # train using discriminator
            # predict x0 using generator
            x0_current_G = self.current_generator(x_jump_start_time, jump_start_t)
            if clip_denoised:
                x0_current_G.clamp_(-1., 1.)

            # sample q(x_t | x_T, x_0) for the current time
            alpha_next = self.alphas_cumprod_prev[time]
            pred_noise = self.predict_noise_from_start(x_jump_start_time, jump_start_t, x0_current_G)
            generated = x0_current_G * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

            # run through discriminator
            discr_score = self.discriminator(generated, t)

            loss = F.softplus(-discr_score)
            acc = (discr_score > 0).float()
        else:       # train generator to do one step of denoising
            assert False, "don't do this yet"
            x0_G = self.current_generator(x_jump_start_time, jump_start_t)
            loss = self.loss_fn(x0_G, img, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return {"loss": loss.mean(), "acc": acc.mean()}

    def set_current_time(self, time):
        oldtime = self.current_time.item()
        oldjump = int(math.floor(self.num_jumps * oldtime / self.num_timesteps))
        newjump = int(math.floor(self.num_jumps * time / self.num_timesteps))

        if oldtime != time and self.current_generator is not None:
            # store current generator into generators
            self.generators[oldjump].load_state_dict(copy.deepcopy(self.current_generator.state_dict()))

        if oldjump != newjump:
            # copy the new generator to current generator
            self.current_generator = copy.deepcopy(self.generators[newjump])

        self.current_time[0] = time
        return self.current_generator if oldjump != newjump else None
