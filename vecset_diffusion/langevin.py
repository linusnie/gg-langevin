import torch
from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np

class SeededGenerator:
    def __init__(self, seeds, device=None, dtype=None):
        self.generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
        self.device = device
        self.dtype = dtype

    def randn(self, *shape):
        out = []
        for g in self.generators:
            out.append(torch.randn(
                *shape,
                generator=g,
                device=self.device, dtype=self.dtype
            ))
        return torch.stack(out)

    def rand(self, *shape):
        out = []
        for g in self.generators:
            out.append(torch.rand(
                *shape,
                generator=g,
                device=self.device, dtype=self.dtype
            ))
        return torch.stack(out)

@dataclass
class GeometryGuidedLangevinState:
    latents: torch.Tensor
    optimizer: torch.optim.Optimizer
    class_labels: torch.Tensor
    random_generator: SeededGenerator
    step: int

@dataclass
class GeometricLoss:
    sdf_weight: float = 1.0
    eikonal_weight: float = 0.1
    siren_weight: float = 0.001
    siren_alpha: float = 50
    num_surface_points: int = 1024
    num_uniform_points: int = 500
    num_siren_points: int = 5000
    eikonal_sigma: float = 0.01
    kl_weight: float = 0.
    distance_power: float = 1.0

    def compute_loss(self, latents, autoencoder, surface_points, random_generator):
        num_shapes = len(latents)

        latents_learn = autoencoder.learn(latents, autoencoder.encoder_layers)

        query_sdfs = autoencoder.decode_final(
            latents_learn,
            surface_points,
        )[..., 0]
        sdf_loss = (query_sdfs.abs() ** self.distance_power).mean(dim=-1)

        if self.eikonal_weight != 0:
            uniform_points = (random_generator.rand(self.num_uniform_points, 3) - 0.5) * 2
            perturbed_points = surface_points + random_generator.randn(*surface_points.shape[1:]) * self.eikonal_sigma

            off_surface_points = torch.concatenate([uniform_points, perturbed_points], dim=1).requires_grad_(True)
            off_surface_sdfs = autoencoder.decode_final(
                latents_learn,
                off_surface_points,
            )[..., 0]
            grad_outputs = torch.ones_like(off_surface_sdfs, device=off_surface_sdfs.device)
            gradients = torch.autograd.grad(
                outputs=off_surface_sdfs,
                inputs=off_surface_points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            eikonal_loss = ((gradients.norm(dim=-1) - 1) ** 2).mean(dim=-1)
        else:
            eikonal_loss = torch.zeros(num_shapes, device='cuda')

        if self.siren_weight != 0:
            siren_points = (random_generator.rand(self.num_siren_points, 3) - 0.5) * 2
            siren_sdfs = autoencoder.decode_final(
                latents_learn,
                siren_points,
            )[..., 0]

            # might as well add uniform points from eikonal loss as well
            all_siren_points = torch.concat([siren_sdfs, off_surface_sdfs[:, :self.num_uniform_points]], dim=1)
            siren_loss = (- self.siren_alpha * all_siren_points.abs()).exp().mean(dim=-1) * 2**3 * self.siren_alpha / 2
        else:
            siren_loss = torch.zeros(num_shapes, device='cuda')

        kl_loss = (latents ** 2).mean(-1).mean(-1)

        loss = self.sdf_weight * sdf_loss + \
            self.eikonal_weight * eikonal_loss + \
            self.siren_weight * siren_loss + \
            self.kl_weight * kl_loss

        return loss, sdf_loss, eikonal_loss, siren_loss, kl_loss


@dataclass
class GeometryGuidedLangevin:
    sigma: float = 0.03
    learning_rate: float = 5e-2
    optimizer: str = 'adam'
    beta1: float = 0.9
    beta2: float = 0.999
    sigma_schedule: Optional[str] = None
    sigma_schedule_max: int = 5000
    sigma_min: float = 0.01
    noise_step: int = 1
    scale_lr: bool = False
    loss: GeometricLoss = field(default_factory=lambda: GeometricLoss())
    anchor_weight: Optional[float] = None

    def init_state(self, init_latents, class_labels, random_seed):
        latents = init_latents.detach().clone().requires_grad_(True)
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([latents], lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD([latents], lr=self.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')
        return GeometryGuidedLangevinState(
            latents=latents,
            optimizer=optimizer,
            class_labels=class_labels,
            step=0,
            random_generator=SeededGenerator(
                random_seed,
                device='cuda',
            ),
        )

    def step(self, surface_points, autoencoder, model, state: GeometryGuidedLangevinState):
        num_shapes = len(surface_points)
        if self.sigma_schedule is None:
            sigma = self.sigma
        elif state.step > self.sigma_schedule_max:
            sigma = self.sigma_min
        elif self.sigma_schedule == 'cos':
            sigma = (self.sigma - self.sigma_min) * (1 + np.cos(np.pi* state.step / self.sigma_schedule_max)) / 2 + self.sigma_min
        elif self.sigma_schedule == 'sin':
            sigma = self.sigma_min + (self.sigma - self.sigma_min) * np.sin(np.pi * state.step / self.sigma_schedule_max)
        elif self.sigma_schedule == 'sin_warmup':
            sigma = self.sigma_min * np.sin(np.pi / 2 * state.step / self.sigma_schedule_max)
        else:
            raise ValueError(f'Unknown sigma schedule: {self.sigma_schedule}')

        if self.scale_lr:
            for param_group in state.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * (sigma / self.sigma_min)  ** 2

        if self.learning_rate != 0:
            loss, sdf_loss, eikonal_loss, siren_loss, kl_loss = self.loss.compute_loss(
                latents=state.latents,
                autoencoder=autoencoder,
                surface_points=surface_points,
                random_generator=state.random_generator,
            )
        else:
            sdf_loss = torch.zeros(num_shapes)
            eikonal_loss = torch.zeros(num_shapes)
            siren_loss = torch.zeros(num_shapes)
            kl_loss = torch.zeros(num_shapes)
            loss = torch.zeros(num_shapes, device='cuda') * state.latents.mean() # ensure backprop works

        if self.anchor_weight is None:
            loss.sum().backward()

        with torch.no_grad():
            if (sigma != 0) and (state.step % self.noise_step == 0):
                noise = state.random_generator.randn(*state.latents.shape[1:])
                latents_noised = state.latents + sigma * noise
                noise_prediction = model(latents_noised, torch.tensor(sigma, device='cuda'), state.class_labels)[0]
                latents_half_denoised = latents_noised - .5 * sigma * noise_prediction

        if sigma != 0:
            if self.anchor_weight is None:
                # in-place update to avoid interfering with optimizer
                with torch.no_grad():
                    state.latents.copy_(latents_half_denoised)
                anchor_loss = torch.zeros(num_shapes)
            else:
                anchor_loss = ((state.latents - latents_half_denoised.detach().clone()) ** 2).mean(-1).mean(-1)
                loss += anchor_loss * self.anchor_weight
                loss.sum().backward()
        else:
            anchor_loss = torch.zeros(num_shapes)

        state.optimizer.step()
        state.optimizer.zero_grad()
        state.step += 1

        metrics = {
            'sdf_loss': sdf_loss.detach().clone(),
            'eikonal_loss': eikonal_loss.detach().clone(),
            'siren_loss': siren_loss.detach().clone(),
            'kl_loss': kl_loss.detach().clone(),
            'anchor_loss': anchor_loss.detach().clone(),
            'loss': loss.detach().clone(),
            'sigma': torch.tensor([sigma for _ in range(num_shapes)]),
            'learning_rate': torch.tensor([state.optimizer.param_groups[0]['lr'] for _ in range(num_shapes)]),
        }

        return state, metrics
