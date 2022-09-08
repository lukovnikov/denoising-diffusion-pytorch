from copy import deepcopy

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
from denoising_diffusion_pytorch.progressive_distillation import ProgressiveDistillationGaussianDiffusion


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
    timesteps = 512

    diffusion = ProgressiveDistillationGaussianDiffusion(model=m, image_size=1, timesteps=timesteps,  #)
                                                         jumpsched=[1, 8, 64, 256, 512])
                                                         # jumpsched = [1, 2,4,8,16,32, 64, 128, 256, 512])
                                                         # jumpsched = 2)

    ds = OneDDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=256)

    print(len(ds))
    samples = [x[0, 0, 0].item() for x in ds]
    print(samples[0:100])
    # _ = plt.hist(samples, density=True, bins=100)
    # plt.show()

    step = 0
    epochs = [2000]
    for _ in diffusion.jumpsched[1:]:
        epochs.append(400)

    _epochs = deepcopy(epochs)

    device = torch.device("cuda:0")
    diffusion.to(device)
    done = False

    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    models = []

    with tqdm(initial=step, total=sum(epochs) * len(dl)) as pbar:
        while not done:
            losses = []
            for batch in dl:
                batch = batch.to(device)

                loss = diffusion(batch)
                loss.backward()
                losses.append(loss.cpu().item())

                pbar.set_description(f'loss: {np.mean(losses):.4f}')

                optimizer.step()
                optimizer.zero_grad()

                diffusion.on_after_optim_step()

                step += 1
                pbar.update(1)

            epochs[0] -= 1
            if epochs[0] == 0:
                models.append(deepcopy(diffusion))
                epochs.pop(0)
                if len(epochs) == 0:
                    break
                diffusion.increase_jump_size()
                print(f"increased jump size from {diffusion.get_prev_jump_size()} to {diffusion.get_jump_size()}")

            if step % (len(dl) * 50) == 0:
                print("")

    # models.append(diffusion)
    print("done training")

    for model in models:
        sampled_images = model.sample(batch_size=2000)
        print(sampled_images.shape)  # (4, 3, 128, 128)
        sampled_images = sampled_images[:, 0, 0, 0]
        # print(sampled_images)
        _ = plt.hist(sampled_images.cpu().numpy(), density=True, bins=200)
        plt.show()

if __name__ == '__main__':
    run()