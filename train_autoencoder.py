import vecset_diffusion
from vecset_diffusion.models import NoisePredictor, ModelConfig
from vecset_diffusion.autoencoder import AEConfig
import utils

import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import tyro
from datetime import datetime
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path
import torch
from torch import inf

@dataclass
class TrainingConfig:
    output_dir: Path
    dataset_path: Path
    experiment_name: str = 'experiment'
    with_timestamp: bool = True

    # Path to yaml config with default settings
    yaml_config: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    n_epochs: int = 5000
    batch_size: int = 16
    num_workers: int = 0
    flash_attention: bool = False
    categories: Optional[List[str]] = None # Chairs=03001627

    learning_rate: float = 5e-5
    learning_rate_warmup: bool = False
    warmup_epochs: int = 16
    beta1: float = 0.3
    beta2: float = 0.9
    weight_decay: float = 0.0
    clip_grad: Optional[float] = None
    accum_iter: int = 1
    eikonal_weight: float = 0.001
    kl_weight: float = 0.001
    original_vecset: bool = False

    # Autoencoder to use, does not apply if dataset_path is a .npy file
    surface_normalization: str = 'cube'
    num_surface_points: int = 2048

    # percentage of input tokens to drop in each batch
    token_dropout: float = 0.

    gaussian_noise_scale: float = 0.
    outlier_probability: float = 0.
    mean_outlier_ratio: float = 0.

    # Augmentation flags
    augment_rotation: bool = False
    augment_jitter: bool = False
    augment_scaling: bool = False

    model_config: AEConfig = field(default_factory=lambda: AEConfig())

    checkpoint_step: int = 10

    eval_step: int = 1

    wandb_project: Optional[str] = None

def points_gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    return points_grad

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def add_point_cloud_noise(
        points,
        gaussian_noise_scale,
        outlier_probability,
        mean_outlier_ratio,
        outlier_range=2.0,
    ):
    noisy_points = points.clone()
    batch_size, num_points = points.shape[:2]

    if gaussian_noise_scale > 0:
        noise_stds = torch.distributions.exponential.Exponential(1 / gaussian_noise_scale).sample((batch_size,)).to(points.device, points.dtype)
        noisy_points = noisy_points + torch.randn_like(noisy_points) * noise_stds[:, None, None]

    for b in range(batch_size):
        if np.random.rand() < outlier_probability:
            num_outliers = np.random.binomial(num_points, mean_outlier_ratio)
            if num_outliers > 0:
                outlier_indices = torch.randperm(num_points, device=points.device)[:num_outliers]
                outliers = (torch.rand(num_outliers, 3, device=points.device) * (2 * outlier_range) - outlier_range).to(points.device, points.dtype)
                noisy_points[b, outlier_indices] = outliers

    return noisy_points


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def run(config: TrainingConfig):
    '''Run training'''
    world_size, local_rank, is_main_process, is_distributed = utils.initialize_distributed()

    if config.yaml_config is not None:
        print(f"\033[92mYAML config {config.yaml_config} provided.")
        with open(config.yaml_config, 'r') as yaml_file:
            defaults = tyro.extras.from_yaml(TrainingConfig, yaml_file)
            config = tyro.cli(TrainingConfig, default=defaults)

    timestamp_str = datetime.now().strftime("_%Y-%m-%d_%H:%M:%S") if config.with_timestamp else ''
    config.experiment_name += timestamp_str
    experiment_dir = config.output_dir / config.experiment_name
    checkpoint_dir = experiment_dir / 'checkpoints'
    if is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir(parents=False, exist_ok=False)

    if config.flash_attention and config.eikonal_weight != 0:
        raise ValueError('Eikonal loss not supported with flash attention.')

    print(f'---\033[93m{config.experiment_name}\033[0m---')
    print(f'\033[96mexperiment dir\033[0m: {experiment_dir}')

    if config.checkpoint_path is None:
        print('creating AE')
        model = vecset_diffusion.autoencoder.create_autoencoder(
            depth=24, dim=512, M=512, N=1,
            query_type=config.model_config.query_type,
            attention_class='flash' if config.flash_attention else 'simple',
            bottleneck=vecset_diffusion.autoencoder.KLBottleneck,
            bottleneck_layer=config.model_config.bottleneck_layer,
            bottleneck_args={'dim': 512, 'latent_dim': 8, 'kl_weight': 1.},
        ).to('cuda')
        optimizer_state, epoch_idx = None, 0
        print('done')
    else:
        print(f'Loading model from {str(config.checkpoint_path)}...')
        model_config, model, optimizer_state, epoch_idx = utils.load_autoencoder_checkpoint(
            config.checkpoint_path,
            model_config=config.model_config if config.original_vecset else None, # Use config from checkpoint if available
            flash_attention=config.flash_attention,
        )
        config.model_config = model_config

    if is_distributed:
        print('creating DDP')
        print(local_rank, len([p for p in list(model.parameters()) if p.requires_grad]), force=True)
        model = torch.nn.parallel.DistributedDataParallel(model, [local_rank])
        print('done')

    utils.print_config(config)
    print('\n\033[96mModel\033[0m:\n', model)

    if is_main_process:
        utils.save_config(config, experiment_dir)

    results = []
    global_batch_size = config.batch_size * world_size
    print("Loading dataset...")

    dataset = vecset_diffusion.shapenet.ShapeNet(
        config.dataset_path,
        split='train',
        transform=vecset_diffusion.shapenet.ShapeNetTransform(
            normalization=config.surface_normalization,
            apply_random_rotation=config.augment_rotation,
            apply_random_jitter=config.augment_jitter,
            apply_random_scaling=config.augment_scaling,
        ),
        sampling=True,
        num_samples=1024,
        return_surface=True,
        surface_sampling=True,
        return_only_surface=False,
        pc_size=config.num_surface_points,
        data_type="sdf",
        categories=config.categories,
        normalization=config.surface_normalization,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        drop_last=False,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=config.num_workers,
    )
    dataset_size = len(dataset)

    dataset_val = vecset_diffusion.shapenet.ShapeNet(
        config.dataset_path,
        split='val',
        transform=vecset_diffusion.shapenet.ShapeNetTransform(
            normalization=config.surface_normalization,
        ),
        sampling=True,
        num_samples=1024,
        return_surface=True,
        surface_sampling=True,
        return_only_surface=False,
        pc_size=config.num_surface_points,
        data_type="sdf",
        categories=config.categories,
        normalization=config.surface_normalization,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=config.batch_size,
        drop_last=False,
        num_workers=config.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    if optimizer_state is not None:
        for param_group in optimizer_state['param_groups']:
            param_group['lr'] = config.learning_rate
            param_group['initial_lr'] = config.learning_rate
            param_group['betas'] = (config.beta1, config.beta2)
            param_group['weight_decay'] = config.weight_decay
        optimizer.load_state_dict(optimizer_state)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step * config.batch_size * world_size / dataset_size / config.warmup_epochs) if config.learning_rate_warmup else 1.0
    )

    if (config.wandb_project is not None) and is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            dir=config.output_dir,
            reinit=True,
            config={
                'num_gpus': world_size,
                **asdict(config),
            },
        )
    utils.save_checkpoint(checkpoint_dir, model, optimizer, 'init', 'init', 'init')

    debug_dump = False
    sample_idx = dataset_size * epoch_idx
    global_batch_idx = (dataset_size // global_batch_size) * epoch_idx

    scheduler.last_epoch = sample_idx
    scheduler.step()
    criterion = torch.nn.L1Loss()
    loss_scaler = NativeScalerWithGradNormCount()

    for epoch_idx in tqdm(range(epoch_idx, epoch_idx + config.n_epochs), disable=not is_main_process):
        for batch_idx, (points, labels, surface, _) in tqdm(
                enumerate(data_loader), leave=False, disable=not is_main_process, total=len(data_loader)
            ):

            points = points.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)
            surface = surface.to('cuda', non_blocking=True)

            if config.gaussian_noise_scale > 0 or config.outlier_probability > 0:
                noisy_surface = add_point_cloud_noise(
                    surface,
                    gaussian_noise_scale=config.gaussian_noise_scale,
                    outlier_probability=config.outlier_probability,
                    mean_outlier_ratio=config.mean_outlier_ratio,
                    outlier_range=2.0,
                )

            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=1)
            outputs = model(noisy_surface, points_all)

            kl = outputs['kl'].mean()
            output = outputs['o'].squeeze(-1)

            if config.eikonal_weight != 0.:
                grad = points_gradient(points_all, output)
                loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
            else:
                loss_eikonal = torch.tensor(0.)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
                # TODO: hard coded point numbers
                loss_vol = criterion(output[:, :1024], labels[:, :1024])
                loss_near = criterion(output[:, 1024:2048], labels[:, 1024:2048])
                loss_surface = (output[:, 2048:]).abs().mean()
                loss = loss_vol + 10 * loss_near + config.eikonal_weight * loss_eikonal + 1 * loss_surface
                loss += config.kl_weight * kl

            loss /= config.accum_iter
            update_grad = (batch_idx + 1) % config.accum_iter == 0
            # loss_scaler(loss, optimizer, clip_grad=config.clip_grad,
                        # parameters=model.parameters(), create_graph=False,
                        # update_grad=update_grad)
            loss.backward()
            if update_grad:
                optimizer.step()

            if is_distributed:
                torch.distributed.all_reduce(loss)
                torch.distributed.all_reduce(loss_near)
                torch.distributed.all_reduce(loss_vol)
                torch.distributed.all_reduce(loss_surface)
                torch.distributed.all_reduce(loss_eikonal)
                torch.distributed.all_reduce(kl)
                loss /= world_size
                loss_near /= world_size
                loss_vol /= world_size
                loss_surface /= world_size
                loss_eikonal /= world_size
                kl /= world_size

            if is_main_process:
                batch_results = {
                    'train/loss': loss.item(),
                    'train/loss_near': loss_near.item(),
                    'train/loss_vol': loss_vol.item(),
                    'train/loss_surface': loss_surface.item(),
                    'train/loss_eikonal': loss_eikonal.item(),
                    'train/kl': kl.item(),
                    'batch_idx': batch_idx,
                    'global_batch_idx': global_batch_idx,
                    'sample_idx': sample_idx,
                    'epoch_idx': epoch_idx + batch_idx * global_batch_size / dataset_size,
                    **vecset_diffusion.training.get_gradient_norms(model),
                    **vecset_diffusion.training.map_params(model, torch.std, 'param_std'),
                    **{f'lr_{i}': param_group['lr'] for i, param_group in enumerate(optimizer.state_dict()['param_groups'])},
                }
                results.append(batch_results)
                if config.wandb_project is not None:
                    wandb.log(batch_results)

            if update_grad:
                optimizer.zero_grad()
            scheduler.step()
            global_batch_idx += 1
            sample_idx += len(points) * world_size

        # end of epoch
        if is_main_process:
            # eval loop
            if epoch_idx % config.eval_step == 0:
                kl, loss_eikonal, loss_vol, loss_near, loss_surface = 0., 0., 0., 0., 0.
                with torch.no_grad():
                    for batch_idx, (points, labels, surface, _) in tqdm(
                            enumerate(data_loader_val), leave=False, disable=not is_main_process
                        ):
                        points = points.to('cuda', non_blocking=True)
                        labels = labels.to('cuda', non_blocking=True)
                        surface = surface.to('cuda', non_blocking=True)

                        # points = points.requires_grad_(True)
                        points_all = torch.cat([points, surface], dim=1)
                        outputs = model(surface, points_all)

                        kl += outputs['kl'].mean() / len(data_loader_val)
                        output = outputs['o'].squeeze(-1)

                        # loss_eikonal += (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean() / len(data_loader_val)
                        loss_vol += criterion(output[:, :1024], labels[:, :1024]) / len(data_loader_val)
                        loss_near += criterion(output[:, 1024:2048], labels[:, 1024:2048]) / len(data_loader_val)
                        loss_surface += (output[:, 2048:]).abs().mean() / len(data_loader_val)

                val_results = {
                    'val/loss_near': loss_near.item(),
                    'val/loss_surface': loss_surface.item(),
                    'val/loss_vol': loss_vol.item(),
                    'val/kl': kl.item(),
                    'epoch_idx': epoch_idx,
                }
                if config.wandb_project is not None:
                    wandb.log(val_results)

            if debug_dump:
                pd.DataFrame(results).to_csv(experiment_dir / 'metrics.csv')
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_idx, global_batch_idx, sample_idx)
                np.savez(
                    experiment_dir / 'debug_batch.npz',
                )
                raise ValueError("NaN gradients. Saving debug dump.")
            if (epoch_idx % config.checkpoint_step == 0) or (epoch_idx == config.n_epochs - 1):
                pd.DataFrame(results).to_csv(experiment_dir / 'metrics.csv')
                utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_idx, global_batch_idx, sample_idx)

    return experiment_dir, pd.DataFrame(results)

if __name__ == '__main__':
	config = tyro.cli(TrainingConfig)
	run(config)
