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


def repeater(dl):
    for loader in repeat(dl):
        for batch in loader:
            yield batch


def run():
    batsize=256
    numsteps = 500
    numjumps = 4
    itersperstep=20
    teacher = OneDModel(64, 3, notime=True)
    students = [OneDModel(32, 3, notime=True) for _ in range(numjumps)]

    x = torch.randn((5, 1, 1, 1))
    t = torch.rand((5,))
    print(x, t)

    # y = m(x, t)
    # print(y)

    diffusion = RollingDistillationGaussianDiffusion(
        teacher=teacher, students=students, image_size=1, timesteps=numsteps, jumps=len(students))

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


    inititers = 5 * itersperstep
    iterschedule = [itersperstep for _ in range(numsteps)]
    iterschedule[0] = inititers
    iterschedule[-1] = inititers
    totaliters = sum(iterschedule)
    print(f"Total number of iterations: {totaliters} (~= {totaliters/len(dl)} epochs)")

    device = torch.device("cuda:3")
    diffusion.to(device)

    teacher_optimizer = torch.optim.Adam(diffusion.teacher.parameters(), lr=1e-3, betas=(0.9, 0.99))

    initstep = 0
    teacher_batchcount = 0
    student_batchcount = 0

    with tqdm(initial=initstep, total=diffusion.num_timesteps) as pbar:
        for step in reversed(range(initstep, diffusion.num_timesteps)):
            teacher_losses = []
            student_losses = []

            # PHASE I: train teacher for current step
            for i in range(iterschedule[step]):
                batch = next(dliter).to(device)
                teacher_batchcount += 1
                loss = diffusion.forward_teacher(batch, time=step)
                loss.backward()
                teacher_losses.append(loss.cpu().item())
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

                pbar.set_description(f'Step {step}: Teacher loss: {np.mean(teacher_losses):.4f}, Updates: {teacher_batchcount}')

            # # PHASE II: train student for current step
            # newstudent = diffusion.set_current_time(step)
            # if newstudent is not None:
            #     student_optimizer = torch.optim.Adam(newstudent.parameters(), lr=1e-3, betas=(0.9, 0.99))
            #
            # for i in range(iterschedule[step]):
            #     batch = next(dliter).to(device)
            #     student_batchcount += 1
            #     loss = diffusion.forward_student(batch, time=step)
            #     loss.backward()
            #     student_losses.append(loss.cpu().item())
            #     student_optimizer.step()
            #     student_optimizer.zero_grad()
            #
            #     pbar.set_description(f'Teacher loss: {np.mean(teacher_losses):.4f}, '
            #                          f'Student loss: {np.mean(student_losses):.4f}, '
            #                          f'Updates: {teacher_batchcount},{student_batchcount}')

            pbar.update(1)

            if (step+1) % 10 == 0:
                # print(f'\nTeacher loss: {np.mean(teacher_losses):.4f}, '
                #                      f'Student loss: {np.mean(student_losses):.4f}, '
                #                      f'Updates: {teacher_batchcount},{student_batchcount}')
                print(f'\nStep {step}: Teacher loss: {np.mean(teacher_losses):.4f}, '
                                     f'Updates: {teacher_batchcount}')



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