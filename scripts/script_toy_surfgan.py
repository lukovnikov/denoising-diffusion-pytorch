from itertools import repeat

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
from denoising_diffusion_pytorch.surfgan import GaussianDiffusionSurfGAN


class OneDModel(torch.nn.Module):
    def __init__(self, dim=16, layers=3, notime=False, isdiscr=False, **kw):
        super().__init__(**kw)
        self.channels = self.out_dim = 1
        self.self_condition = False
        self.notime = notime
        self.isdiscr = isdiscr
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

        if self.isdiscr:
            return h[:, 0]
        else:
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


def repeater(dl):
    for loader in repeat(dl):
        for batch in loader:
            yield batch


def run():
    batsize=256
    numsteps = 100
    numjumps = 1
    itersperstep=25

    generators = [OneDModel(32, 2, notime=True) for _ in range(numjumps)]
    discriminator = OneDModel(32, 2, notime=True, isdiscr=True)

    x = torch.randn((5, 1, 1, 1))
    t = torch.rand((5,))
    print(x, t)

    # y = m(x, t)
    # print(y)

    m = GaussianDiffusionSurfGAN(
        generators=generators, discriminator=discriminator, image_size=1, timesteps=numsteps, jumps=len(generators))

    ds = OneDDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=batsize)

    print(len(ds))
    samples = [x[0, 0, 0].item() for x in ds]
    print(samples[0:100])
    # _ = plt.hist(samples, density=True, bins=100)
    # plt.show()

    dliter = repeater(dl)
    # for i in range(1000):
    #     batch = next(dliter)
    # print("done iterating")

    totaliters = itersperstep * numsteps
    print(f"Total number of iterations: {totaliters} (~= {totaliters/len(dl)} epochs)")

    device = torch.device("cuda:3")
    m.to(device)

    discr_optimizer = torch.optim.Adam(m.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

    initstep = 0
    totalitercount = 0

    intermediate_gens = []

    with tqdm(initial=initstep, total=m.num_timesteps) as pbar:
        for step in reversed(range(initstep, m.num_timesteps)):
            Dlosses, Glosses, Daccs, Gaccs = [], [], [], []

            newgenerator = m.set_current_time(step)
            if newgenerator is not None:
                gen_optimizer = torch.optim.Adam(newgenerator.parameters(), lr=1e-4, betas=(0.9, 0.99))

            numiters = itersperstep
            if newgenerator is not None:
                numiters *= 5

            for iter in range(numiters):
                # do update of discriminator
                discr_optimizer.zero_grad()
                batch = next(dliter).to(device)
                out = m.forward_discriminator(batch, time=0)
                out["loss"].backward()
                Dlosses.append(out["loss"].detach().cpu().item())
                Daccs.append(out["acc"].detach().cpu().item())
                discr_optimizer.step()
                discr_optimizer.zero_grad()

                # do update of generator
                gen_optimizer.zero_grad()
                batch = next(dliter).to(device)
                out = m.forward_generator(batch, time=0)
                out["loss"].backward()
                Glosses.append(out["loss"].detach().cpu().item())
                Gaccs.append(out["acc"].detach().cpu().item())
                gen_optimizer.step()
                gen_optimizer.zero_grad()

                totalitercount += 1

                pbar.set_description(f'Step: {step}: D: {np.mean(Dlosses):.4f}, {np.mean(Daccs):.4f} '
                                     f'G: {np.mean(Glosses):.4f}, {np.mean(Gaccs):.4f} (iters: {totalitercount})')

            pbar.update(1)

            sampled_images = m.sample(batch_size=1000)
            sampled_images = sampled_images[:, 0, 0, 0].cpu().numpy()
            sampled_hist, _ = np.histogram(sampled_images, density=True, bins=100, range=(-2, 2))
            sampled_hist = sampled_hist / sampled_hist.max()
            intermediate_gens.append(sampled_hist)

            if (step+1) % 10 == 0:
                print("")

    m.set_current_time(-1)
    # Problem: generator doesn't follow discriminator


    print("done training")

    imdata = np.stack(intermediate_gens, 0)
    plt.imshow(imdata)
    plt.show()

    print(imdata[-1])
    """
    sampled_images = m.sample(batch_size=1000)
    print(sampled_images.shape)  # (4, 3, 128, 128)
    sampled_images = sampled_images[:, 0, 0, 0]
    print(sampled_images)
    _ = plt.hist(sampled_images.cpu().numpy(), density=True, bins=100)
    plt.show()
    """

if __name__ == '__main__':
    run()