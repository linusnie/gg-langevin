from vecset_diffusion.langevin import GeometricLoss, SeededGenerator

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional

# -----------------------------------------------------------------------------
# Utilities reused by DAPS
# -----------------------------------------------------------------------------
def _get_sigma_schedule(num_steps: int,
                        sigma_max: float,
                        sigma_min: float,
                        rho: float = 7.0,
                        device: str = 'cuda') -> torch.Tensor:
    """
    Karras-style noise schedule in latent space. Returns length (num_steps+1)
    with the last entry == 0.
    """
    step_idx = torch.arange(num_steps, dtype=torch.float32, device=device)
    t = (sigma_max ** (1.0 / rho)
         + step_idx / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    t = torch.cat([t, torch.zeros_like(t[:1])])  # ensure t_N == 0
    return t


@torch.no_grad()
def _ode_solver(init_latents: torch.Tensor,
                model: torch.nn.Module,
                num_steps: int,
                sigma_max: float,
                sigma_min: float = 1e-3,
                rho: float = 7.0,
                class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Heun (2nd-order) solver in latent space used by DAPS to obtain x0_hat.
    """
    device = init_latents.device
    t_steps = _get_sigma_schedule(num_steps, sigma_max, sigma_min, rho, device=device)
    x_next = init_latents
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        d_cur = model(x_cur, t_cur, class_labels)[0]
        x_next = x_cur + (t_next - t_cur) * d_cur  # Euler
        if i < num_steps - 1:
            d_prime = model(x_next, t_next, class_labels)[0]
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)  # Heun
    return x_next

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
class DAPSState:
    latents: torch.Tensor
    optimizer: torch.optim.Optimizer
    class_labels: torch.Tensor
    step: int
    # DAPS-specific running state
    anneal_schedule: torch.Tensor
    x0_hat: torch.Tensor  # updated at the start of each Langevin window
    random_generator: SeededGenerator


@dataclass
class DAPS:
    """
    Diffusion-Aided Projected Sampling (latent DAPS) adapted to your training loop.
    It alternates:
      - ODE step to obtain x0_hat at the start of each annealing window
      - Langevin dynamics combining prior gradient (toward x0_hat) and geometry loss
    """
    # Geometry term (same structure as your GeometryGuidedLangevin)
    loss: GeometricLoss = field(default_factory=lambda: GeometricLoss())

    # Optimizer for the geometry gradient
    optimizer: str = 'adam'
    learning_rate: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999

    # DAPS schedules / hyperparams
    annealing_steps: int = 10            # number of annealing levels
    langevin_steps: int = 500            # steps per annealing level
    ode_steps: int = 2                   # ODE steps to compute x0_hat
    annealing_sigma_max: float = 10.0
    annealing_sigma_min: float = 2e-2
    daps_lr: float = 1e-4                # LR used inside the Langevin update (r_t, noise term)
    rho: float = 7.0

    # Initialization
    initialization: str = 'random_normal'

    def init_state(self, init_latents: torch.Tensor, class_labels: torch.Tensor, random_seed, model=None) -> DAPSState:
        # Build annealing schedule once
        anneal_schedule = _get_sigma_schedule(
            self.annealing_steps, self.annealing_sigma_max, self.annealing_sigma_min, self.rho, device='cuda'
        )  # length == annealing_steps + 1, last == 0

        if self.initialization == 'random_normal':
            latents = torch.randn_like(init_latents) * anneal_schedule[0]
        elif self.initialization == 'from_latents':
            latents = (init_latents + torch.randn_like(init_latents) * anneal_schedule[0]).detach().clone()
        elif self.initialization == 'inversion':
            latents = edm_inversion(
                init_latents,
                model,
                256, self.annealing_sigma_max, self.annealing_sigma_min, self.rho,
                class_labels,
            )
        else:
            raise ValueError(f'Unknown initialization: {self.initialization}')

        latents.requires_grad_(True)

        if self.optimizer == 'adam':
            opt = torch.optim.Adam([latents], lr=self.learning_rate, betas=(self.beta1, self.beta2))
        elif self.optimizer == 'sgd':
            opt = torch.optim.SGD([latents], lr=self.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        # x0_hat computed at first step=0 (start of the first window)
        x0_hat = torch.zeros_like(latents)

        return DAPSState(
            latents=latents,
            optimizer=opt,
            class_labels=class_labels,
            step=0,
            anneal_schedule=anneal_schedule,
            x0_hat=x0_hat,
            random_generator=SeededGenerator(
                random_seed,
                device='cuda',
            ),
        )

    def _window_indices(self, step: int) -> Tuple[int, int]:
        """Return (anneal_idx, langevin_idx) for the global step."""
        anneal_idx = step // self.langevin_steps
        langevin_idx = step % self.langevin_steps
        return anneal_idx, langevin_idx

    def step(self,
             surface_points: torch.Tensor,
             autoencoder: torch.nn.Module,
             model: torch.nn.Module,
             state: DAPSState):

        num_shapes = len(surface_points)
        device = state.latents.device

        anneal_idx, langevin_idx = self._window_indices(state.step)

        # (Re)compute x0_hat at the start of each annealing window via ODE solve
        if langevin_idx == 0:
            sigma_t = state.anneal_schedule[anneal_idx]
            with torch.no_grad():
                x0_hat = _ode_solver(
                    state.latents, model,
                    num_steps=self.ode_steps,
                    sigma_max=float(sigma_t),
                    sigma_min=1e-3,
                    rho=self.rho,
                    class_labels=state.class_labels,
                )
            state.x0_hat = x0_hat.detach()
            state.latents.data = x0_hat.data.clone()

        # ----- Geometry loss & grads -----
        loss, sdf_loss, eikonal_loss, siren_loss, kl_loss = self.loss.compute_loss(
            latents=state.latents,
            autoencoder=autoencoder,
            surface_points=surface_points,
            random_generator=state.random_generator,
        )
        # Backprop to populate .grad for geometry update
        loss.sum().backward()
        # geom_grad = state.latents.grad.detach().clone()

        # ----- Prior gradient term (towards x0_hat) -----
        sigma_t = state.anneal_schedule[anneal_idx]
        # r_t = lr / sigma_t^2 in the original impl
        prior_coef = self.daps_lr / (float(sigma_t) ** 2 + 1e-12)
        prior_grad = (state.latents - state.x0_hat).detach()

        with torch.no_grad():
            # Langevin noise
            eps = torch.randn_like(state.latents)
            # Apply prior part + noise directly to latents (in-place)
            state.latents.add_(-prior_coef * prior_grad).add_(torch.sqrt(torch.tensor(2.0 * self.daps_lr, device=device)) * eps)

        # Geometry optimizer step (use the gradient computed from the geometry loss)
        # We inject the cached gradient into .grad and step the chosen optimizer
        with torch.no_grad():
            before = state.latents.detach().clone()
        # state.latents.grad = geom_grad  # set explicit grad from loss
        state.optimizer.step()
        state.optimizer.zero_grad()
        with torch.no_grad():
            geom_step_norm = (state.latents - before).norm().detach()

        # At the end of a window, add annealing noise for the *next* sigma
        if langevin_idx == (self.langevin_steps - 1):
            with torch.no_grad():
                # sample from N(0, sigma_{k+1}^2) and add (last schedule value is 0)
                next_sigma = state.anneal_schedule[min(anneal_idx + 1, len(state.anneal_schedule) - 1)]
                if next_sigma > 0:
                    state.latents.add_(torch.randn_like(state.latents) * float(next_sigma))

        state.step += 1

        metrics = {
            'sdf_loss': sdf_loss.detach().clone(),
            'eikonal_loss': eikonal_loss.detach().clone(),
            'siren_loss': siren_loss.detach().clone(),
            'kl_loss': kl_loss.detach().clone(),
            'loss': loss.detach().clone(),
            'sigma': torch.full((num_shapes,), float(sigma_t), device=device),
            'diffusion_gradient_norm_with_lr': torch.full((num_shapes,), prior_coef, device=device) * prior_grad.norm(dim=list(range(1, prior_grad.ndim))).detach(),
            'geom_gradient_norm_with_lr': torch.full((num_shapes,), geom_step_norm.item(), device=device),
            'langevin_lr': torch.full((num_shapes,), float(self.daps_lr), device=device),
        }
        return state, metrics
