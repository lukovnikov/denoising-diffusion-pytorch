import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import math
import os, sys
from tqdm import tqdm
import random

from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import SinusoidalPosEmb, unnormalize_to_zero_to_one
from denoising_diffusion_pytorch.distillation import RollingDistillationGaussianDiffusion


class DebugGaussianDiffusion(RollingDistillationGaussianDiffusion):

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = None):
        clip_denoised = (self.objective != "pred_x0") if clip_denoised is None else clip_denoised
        batch, device, total_timesteps, sampling_timesteps, objective \
            = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.objective

        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device = device)
        imgacc = []
        x0acc = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, time=time)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            img = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()
            imgacc.append(unnormalize_to_zero_to_one(img))
            x0acc.append(unnormalize_to_zero_to_one(x_start))

        img = unnormalize_to_zero_to_one(img)
        # if x_T is None:
        #     return img, imgacc, x0acc
        # else:
        return img, times, imgacc, x0acc


class OneDModel(torch.nn.Module):
    def __init__(self, dim=16, layers=3, notime=False, **kw):
        super().__init__(**kw)
        self.channels = self.out_dim = 1
        self.self_condition = False
        self.notime = notime
        time_dim = dim * 4

        if not self.notime:
            sinu_pos_emb = SinusoidalPosEmb(dim)

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, dim)
            )

        self.main = torch.nn.ModuleList([
            torch.nn.Linear(1, dim),
            torch.nn.GELU(),
        ])

        for _ in range(layers):
            self.main.append(torch.nn.Linear(dim, dim))
            self.main.append(torch.nn.GELU())

        self.main.append(torch.nn.Linear(dim, 1))

    def forward(self, x_t, t=None, self_cond=None):
        """ x: (batsize, 1, 1, 1)"""
        x_t = x_t.squeeze(-1).squeeze(-1)

        h = self.main[0](x_t)

        if not self.notime:
            temb = self.time_mlp(t)
            h = h + temb

        for layer in self.main[1:]:
            h = layer(h)

        h = h[:, :, None, None]
        return h


class OneDDataset(torch.utils.data.Dataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.peaks = [0, 0.2, 0.8, 1]
        self.var = 0.001

    def __getitem__(self, i):
        # randomly choose one of the peaks
        ret = random.choice(self.peaks)
        ret = ret + random.gauss(0, self.var)
        return torch.tensor([ret])[:, None, None]

    def __len__(self):
        return 4000

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]





def run():
    m = OneDModel(64)
    x = torch.randn((5, 1, 1, 1))
    t = torch.rand((5,))
    print(x, t)

    # y = m(x, t)
    # print(y)

    diffusion = DebugGaussianDiffusion(model=m, image_size=1, timesteps=100, loss_type="l2")

    ds = OneDDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    print(len(ds))
    samples = [x[0, 0, 0].item() for x in ds]
    print(samples[0:100])
    # _ = plt.hist(samples, density=True, bins=100)
    # plt.show()

    step = 0
    epochs = 100

    device = torch.device("cuda:0")
    diffusion.to(device)
    done = False

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3, betas=(0.9, 0.99))

    with tqdm(initial=step, total=epochs * len(dl)) as pbar:
        while not done:
            losses = []
            for batch in dl:
                batch = batch.to(device)

                total_loss = 0.

                loss = diffusion(batch)
                loss.backward()
                losses.append(loss.cpu().item())

                pbar.set_description(f'loss: {np.mean(losses):.4f}')

                optimizer.step()
                optimizer.zero_grad()

                step += 1
                pbar.update(1)

                if step >= epochs * len(dl):
                    done = True
            if step % (len(dl) * 50) == 0:
                print("")

    print("done training")

    diffusion.is_ddim_sampling = True
    sampled_images = diffusion.sample(batch_size=1000)
    print(sampled_images.shape)  # (4, 3, 128, 128)
    sampled_images = sampled_images[:, 0, 0, 0]
    print(sampled_images)
    _ = plt.hist(sampled_images.cpu().numpy(), density=True, bins=100)
    plt.show()

if __name__ == '__main__':
    run()