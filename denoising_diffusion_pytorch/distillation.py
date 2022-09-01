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

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16,
        ignore_time=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        self.ignore_time = ignore_time
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time=None, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = None
        if time is not None and not self.ignore_time:
            t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

class RollingDistillationGaussianDiffusion(nn.Module):
    def __init__(
        self,
        teacher,
        students,
        image_size,
        timesteps = 1000,
        jumps=8,
        loss_type = 'l2',
        beta_schedule = 'cosine',
    ):
        super().__init__()

        self.teacher = teacher
        self.students = torch.nn.ModuleList(students)
        self.channels = self.teacher.channels
        self.self_condition = self.teacher.self_condition

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
        self.current_student = None

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
        # select the right student
        model, student_i = self.get_student(time)
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
    # .forward_teacher() returns the loss of the teacher for a given image batch
    # .forward_student() returns the loss for the current student for a given image batch
    # .set_current_time() configures the current student according to the given time step
    # - training sequence: start with time=0 and repeat the iteration:
    #       Phase I: train teacher by calling .forward_teacher() and backpropagating with a number of batches
    #       Phase II: train student:
    #           1. call .set_current_time(t)  --> this makes sure self.current_student is set
    #               if the returned value is None, the current student is the same as it was
    #                                 ... is not None, then the student changed and we have to create a new optimizer!
    #           2. call .forward_student()    --> this distills teacher into student
    #           3. repeat steps 1 and 2 for a certain number of batches
    #       Set t <- t + 1 and repeat Phases I and II

    def forward_teacher(self, img, time=None, x_T=None):
        time = time if time is not None else self.current_time.item()
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.full((b,), time, device=device).long()

        img = normalize_to_neg_one_to_one(img)

        noise = torch.randn_like(img) if x_T is None else x_T

        # noise sample
        x = self.q_sample(x_start=img, t=t, noise=noise)

        # predict and take gradient step
        model_out = self.teacher(x, t)

        loss = self.loss_fn(model_out, img, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        return loss.mean()

    def get_student(self, time):
        student_i = int(math.floor(self.num_jumps * time / self.num_timesteps))
        return self.students[student_i], student_i

    def forward_student(self, img, time=None, clip_denoised=True):
        time = time if time is not None else self.current_time.item()
        # pick the right student and distill from teacher
        with torch.no_grad():
            model, student_i = self.get_student(time)
            jump_start_time = int(student_i / self.num_jumps * self.num_timesteps)
            jump_size = time - jump_start_time

            jump_start_t = torch.full((img.size(0),), jump_start_time, device=img.device).long()
            x_jump_start_time = self.q_sample(img, jump_start_t)

            if jump_size > 1:
                # predict x0 using current (incomplete) student
                x0_current_student = model(x_jump_start_time, jump_start_t)
                if clip_denoised:
                    x0_current_student.clamp_(-1., 1.)

                # sample q(x_t | x_T, x_0) for the current time
                alpha_next = self.alphas_cumprod_prev[time]
                pred_noise = self.predict_noise_from_start(x_jump_start_time, jump_start_t, x0_current_student)
                x_t = x0_current_student * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

                # predict next step using teacher
                t = torch.full((img.size(0),), time, device=img.device).long()
                x0_teacher = self.teacher(x_t, t)

                if clip_denoised:
                    x0_teacher.clamp_(-1., 1.)
            else:
                x0_teacher = img

        # train current student to predict the same x_t-1 as teacher
        x0_student = self.current_student(x_jump_start_time, jump_start_t)
        loss = self.loss_fn(x0_student, x0_teacher, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        return loss.mean()

    def set_current_time(self, time):
        oldtime = self.current_time.item()
        oldjump = int(math.floor(self.num_jumps * oldtime / self.num_timesteps))
        newjump = int(math.floor(self.num_jumps * time / self.num_timesteps))

        if oldtime != time and self.current_student is not None:
            # store current student into students
            self.students[oldjump].load_state_dict(copy.deepcopy(self.current_student.state_dict()))

        if oldjump != newjump:
            # copy the new student to current student
            self.current_student = copy.deepcopy(self.students[newjump])

        self.current_time[0] = time
        return self.current_student if oldjump != newjump else None

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
