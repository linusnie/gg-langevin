from vecset_diffusion.langevin import GeometricLoss, SeededGenerator

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal

# -----------------------------------------------------------------------------
# Inversion
# -----------------------------------------------------------------------------
from vecset_diffusion.daps import _get_sigma_schedule
@torch.no_grad()
def edm_inversion(x0: torch.Tensor,
                    model: torch.nn.Module,
                    num_steps: int,
                    sigma_max: float,
                    sigma_min: float = 1e-3,
                    rho: float = 7.0,
                    class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Inverse integrator of _ode_solver: integrates from low sigma -> high sigma.
    Uses the same schedule (so that running _ode_solver on the output recovers x0).
    The integrator assumes the epsilon (noise) parameterization:
        dx/dsigma = eps_hat(x, sigma)
    """
    device = x0.device
    t_steps = _get_sigma_schedule(num_steps, sigma_max, sigma_min, rho, device=device)[:-1] # remove last 0
    rev = t_steps.flip(0)  # ascending: sigma_min -> sigma_max
    x_next = x0
    for i, (t_cur, t_next) in enumerate(zip(rev[:-1], rev[1:])):
        x_cur = x_next
        d_cur = model(x_cur, t_cur, class_labels)[0]
        x_next = x_cur + (t_next - t_cur) * d_cur
        # Heun correction
        if i < rev.shape[0] - 2:
            d_prime = model(x_next, t_next, class_labels)[0]
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# -----------------------------------------------------------------------------
# DAPS
# -----------------------------------------------------------------------------
@dataclass
class ClassifierGuidanceState:
    latents: torch.Tensor
    optimizer: torch.optim.Optimizer
    class_labels: torch.Tensor
    step: int
    # DAPS-specific running state
    sigma_schedule: torch.Tensor
    x0_hat: torch.Tensor  # updated at the start of each Langevin window
    random_generator: SeededGenerator

@dataclass
class ClassifierGuidance:
    """
    For applying initialisationg
    """
    # Geometry term (same structure as your GeometryGuidedLangevin)
    loss: GeometricLoss = field(default_factory=lambda: GeometricLoss())

    # Optimizer for the geometry gradient
    #optimizer: str = 'adam'
    learning_rate: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999

    # DPS schedules / hyperparams
    sigma_max: float = 10.0
    sigma_min: float = 2e-2

    num_steps: int = 500
    rho: float = 7.0

    # Initialization
    """
    dps: use strategy form the paper. gradient from measurement is normalsied and multipled by lr. This is equivalent to sgd with constant lr + gradient clipping.
    adam: our strategy by using adam optimizer
    none: for debug purposes omit guidance
    """
    init_strategy: Literal["noise", "inversion", "random"] = "noise"  # if noise use sigma_max * N(0, 1) + init. if inversion use edm_inversion

    guidance_step_strategy: Literal["dps", "adam", "none", "sigma"] = "adam"  # dps uses normalised gradient * constant
    inversion_steps: int = 256

    def init_state(self, init_latents: torch.Tensor, class_labels: torch.Tensor, random_seed: int) -> ClassifierGuidanceState:
        latents = init_latents.detach().clone().requires_grad_(True)
        device = latents.device

        # Build annealing schedule once
        sigma_schedule = _get_sigma_schedule(
            self.num_steps, self.sigma_max, self.sigma_min, self.rho, device=device
        )  # length == annealing_steps + 1, last == 0

        if self.guidance_step_strategy == 'adam':
            opt = torch.optim.Adam([latents], lr=self.learning_rate, betas=(self.beta1, self.beta2))
        elif self.guidance_step_strategy == 'dps':
            opt = torch.optim.SGD([latents], lr=self.learning_rate)
        elif self.guidance_step_strategy == 'none':
            opt = torch.optim.SGD([latents], lr=1e-12) # no operation.
        else:
            raise ValueError(f'Unknown guidance step strategy: {self.guidance_step_strategy}')

        # x0_hat computed at first step=0 (start of the first window)
        x0_hat = torch.zeros_like(latents)

        return ClassifierGuidanceState(
            latents=latents,
            optimizer=opt,
            class_labels=class_labels,
            step=0,
            sigma_schedule=sigma_schedule,
            x0_hat=x0_hat,
            random_generator=SeededGenerator(
                random_seed,
                device=device,
            ),
        )

    def step(self,
             surface_points: torch.Tensor,
             autoencoder: torch.nn.Module,
             model: torch.nn.Module,
             state: ClassifierGuidanceState):

        num_shapes = len(surface_points)
        device = state.latents.device

        if state.step == 0:
            state.optimizer.zero_grad(set_to_none=False)
            with torch.no_grad():
                # Optional: scale random init by the first sigma
                if self.init_strategy == "noise":
                    state.latents.copy_(state.latents + state.random_generator.randn(*state.latents.shape[1:]) * self.sigma_max)  # use z + sigma_max * noise
                elif self.init_strategy == "inversion":
                    state.latents.copy_(edm_inversion(state.latents, model, self.inversion_steps, self.sigma_max, self.sigma_min, self.rho, state.class_labels))
                elif self.init_strategy == "random":
                    state.latents.copy_(state.random_generator.randn(state.latents.shape) * self.sigma_max)

        # recompute x0_hat using current prediction by tweddie formula
        sigma = state.sigma_schedule[state.step]
        eps_hat = model(state.latents, sigma, state.class_labels)[0]
        x0_hat = state.latents - sigma * eps_hat

        # ----- Geometry loss & grads -----
        if self.guidance_step_strategy != "none": # for debug pruporses omit guidance
            loss, sdf_loss, eikonal_loss, siren_loss, _ = self.loss.compute_loss(
                latents=x0_hat,
                autoencoder=autoencoder,
                surface_points=surface_points,
                random_generator=state.random_generator,
            )
            # Backprop to populate .grad for geometry update
            loss.sum().backward()
            if self.guidance_step_strategy == 'dps':
                # gradient normalisation to 1 norm
                g = state.latents.grad
                g = g / (g.norm(p=2) + 1e-12)
                state.latents.grad = g
        else:
            loss = torch.tensor([0.0], device=device); sdf_loss = torch.tensor([0.0], device=device); eikonal_loss = torch.tensor([0.0], device=device); siren_loss = torch.tensor([0.0], device=device)
        # geom_grad = state.latents.grad.detach().clone()

        with torch.no_grad():
            if (sigma != 0): # edm update
                t_cur = state.sigma_schedule[state.step]
                t_next = state.sigma_schedule[state.step + 1]
                x_cur = state.latents.detach().clone()

                d_cur = eps_hat.detach().clone()
                x_next = x_cur + (t_next - t_cur) * d_cur  # Euler
                if state.step < self.num_steps - 1:
                    d_prime = model(x_next, t_next, state.class_labels)[0]
                    x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)  # Heun

                state.latents.copy_(x_next)
            else:
                print("TODO probably an error we should not be here")

        # Geometry optimizer step (use the gradient computed from the geometry loss)

        # state.latents.grad = geom_grad  # set explicit grad from loss
        if self.guidance_step_strategy != "none":
            state.optimizer.step()
            state.optimizer.zero_grad()

        state.step += 1

        # sdf_loss = torch.tensor([0.0], device=device); eikonal_loss = torch.tensor([0.0], device=device); siren_loss = torch.tensor([0.0], device=device); loss = torch.tensor([0.0], device=device);
        metrics = {
            'sdf_loss': sdf_loss.detach().clone(),
            'eikonal_loss': eikonal_loss.detach().clone(),
            'siren_loss': siren_loss.detach().clone(),
            'loss': loss.detach().clone(),
            'sigma': torch.full((num_shapes,), float(sigma), device=device),
        }
        return state, metrics
