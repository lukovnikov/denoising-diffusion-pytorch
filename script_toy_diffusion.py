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


class DebugGaussianDiffusion(GaussianDiffusion):
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        imgacc = []
        x0acc = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)     # img: x_t-1, x_start: predicted x_0
            imgacc.append(unnormalize_to_zero_to_one(img))
            x0acc.append(unnormalize_to_zero_to_one(x_start))

        img = unnormalize_to_zero_to_one(img)
        return img, imgacc, x0acc

    @torch.no_grad()
    def ddim_sample(self, shape, x_T=None, clip_denoised=True):
        print("ddim")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
                                                                                 0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device) if x_T is None else x_T

        x_start = None

        imgacc = []
        x0acc = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            if time_next > -1:
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = ((1 - alpha_next) - sigma ** 2).sqrt()

                noise = torch.randn_like(img) if time_next > 0 else 0.

                img = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
            else:
                img = x_start

            imgacc.append(unnormalize_to_zero_to_one(img))
            x0acc.append(unnormalize_to_zero_to_one(x_start))

        img = unnormalize_to_zero_to_one(img)

        if x_T is None:
            return img, imgacc, x0acc
        else:
            return img, times, imgacc, x0acc


class OneDModel(torch.nn.Module):
    def __init__(self, dim=16, **kw):
        super().__init__(**kw)
        self.channels = self.out_dim = 1
        self.self_condition = False
        time_dim = dim * 4

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
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, 1)
        ])

    def forward(self, x_t, t, self_cond=None):
        """ x: (batsize, 1, 1, 1)"""
        x_t = x_t.squeeze(-1).squeeze(-1)

        temb = self.time_mlp(t)

        h = self.main[0](x_t)
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
    diffusion.sampling_timesteps = 10
    sampled_images = diffusion.sample(batch_size=1000)
    print(sampled_images.shape)  # (4, 3, 128, 128)
    sampled_images = sampled_images[:, 0, 0, 0]
    print(sampled_images)
    _ = plt.hist(sampled_images.cpu().numpy(), density=True, bins=100)
    plt.show()

if __name__ == '__main__':
    run()