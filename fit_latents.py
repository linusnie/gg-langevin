import vecset_diffusion
from vecset_diffusion.models import NoisePredictor, ModelConfig
from vecset_diffusion import langevin
from vecset_diffusion.langevin import GeometryGuidedLangevin
from vecset_diffusion.daps import DAPS
import utils

import contextlib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import tyro
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Annotated
from pathlib import Path
import torch
import blender_plots as bplt
import blender_utils as bu
import trimesh
from PIL import Image

Methods = Union[
    GeometryGuidedLangevin,
    DAPS,
]

@dataclass
class TrainingConfig:
    output_dir: Path
    pointcloud_path: Path
    model_path: Path
    autoencoder_path: Path

    experiment_name: str = 'experiment'
    with_timestamp: bool = True

    # Path to yaml config with default settings
    yaml_config: Optional[Path] = None

    n_steps: int = 1000
    batch_size: int = 1024

    surface_normalization: str = 'sphere'

    checkpoint_step: int = 10

    # save a blender file with the reconstructed mesh
    save_mesh: bool = True

    # save an animation of the sampling trajectory
    save_animation: bool = True

    wandb_project: Optional[str] = None

    method: Methods = field(default_factory=lambda: langevin.GeometryGuidedLangevin())

    gt_scale: bool = False

    metrics_gt_scale: bool = True

    random_seed: int = 42

    initialization: str = 'random_normal'

    mc_resolution: int = 256

    # ammount of noise to add to the point cloud before passing it to the encoder for initialization
    init_point_cloud_noise: float = 0.

    # factor by which to subsample the input point cloud
    point_cloud_subsample: int = 1

    # ammount of noise to add to the input point cloud (for noisy experiments)
    point_cloud_noise: float = 0.

    shape_indices: Optional[List[int]] = None

@contextlib.contextmanager
def temp_seed(seed):
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)

def sample_indices(surface_points, batch_size, generator):
    """Sample without replacement if len(surface_points) > batch_size, otherwise with replacement."""
    if len(surface_points) < batch_size:
        return torch.randint(
            0,
            len(surface_points),
            size=(batch_size,),
            generator=generator,
            device='cuda',
        )
    else:
        return torch.randperm(
            len(surface_points),
            generator=generator,
            device='cuda',
        )[:batch_size]

def run(config: TrainingConfig):
    '''Run training'''
    if config.yaml_config is not None:
        print(f"\033[92mYAML config {config.yaml_config} provided.\033[0m")
        with open(config.yaml_config, 'r') as yaml_file:
            defaults = tyro.extras.from_yaml(TrainingConfig, yaml_file)
            config = tyro.cli(TrainingConfig, default=defaults)

    timestamp_str = datetime.now().strftime("_%Y-%m-%d_%H:%M:%S") if config.with_timestamp else ''
    config.experiment_name += timestamp_str
    experiment_dir = config.output_dir / config.experiment_name

    experiment_dir.mkdir(parents=True, exist_ok=False)
    if config.save_mesh:
        (experiment_dir / 'images').mkdir(parents=False, exist_ok=False)

    print(f'---\033[93m{config.experiment_name}\033[0m---')
    print(f'\033[96mexperiment dir\033[0m: {experiment_dir}')

    model_config, model, _, _ = utils.load_model_checkpoint(config.model_path)
    _, autoencoder, _, _ = utils.load_autoencoder_checkpoint(
        config.autoencoder_path,
        model_config=None,
        attention_class='simple',
    )

    model.eval()
    autoencoder.eval()

    utils.print_config(model_config)
    utils.save_config(config, experiment_dir)

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    if config.pointcloud_path.suffix == '.json':
        with open(config.pointcloud_path, 'r') as json_file:
            datalist = json.load(json_file)
        with open(experiment_dir / 'datalist.json', 'w') as json_file:
            json.dump(datalist, json_file)
        shape_ids, surface_points, scale_factors, class_labels = [], [], [], []
        if config.shape_indices is not None:
            shape_indices = config.shape_indices
            datalist = [datalist[shape_index] for shape_index in shape_indices]
        else:
            shape_indices = list(range(len(datalist)))

        for i in range(len(datalist)):
            model_id = datalist[i]["model_id"]
            surface_points_ = torch.tensor(np.load(config.pointcloud_path.parent / f'scans/{model_id}.npy'), device='cuda')
            surface_points_ = surface_points_[::config.point_cloud_subsample]
            if config.gt_scale:
                gt_points = np.load(config.pointcloud_path.parent / f'gt_points/{model_id}.npz')
                scale_factor = np.linalg.norm(gt_points['points'], axis=-1).max()
            else:
                scale_factor = surface_points_.norm(dim=-1).max()
            surface_points_ = surface_points_ / scale_factor
            surface_points.append(surface_points_)
            shape_ids.append(model_id)
            scale_factors.append(scale_factor.item())
            class_labels.append(
                vecset_diffusion.shapenet.category_ids[datalist[i]['class_id']]
            )
        class_labels = torch.tensor(class_labels, device='cuda')
        shape_output_dirs = [experiment_dir / shape_id for shape_id in shape_ids]
        for shape_output_dir in shape_output_dirs:
            shape_output_dir.mkdir(parents=False, exist_ok=False)
    else:
        surface_points = torch.tensor(np.load(config.pointcloud_path), device='cuda')
        scale_factor = surface_points.norm(dim=-1).max()
        surface_points = surface_points / scale_factor
        surface_points = [surface_points]
        shape_ids = ['shape']
        shape_output_dirs = [experiment_dir]
        scale_factors = [scale_factor.item()]
        class_labels = torch.tensor([18], device='cuda')
    num_shapes = len(surface_points)

    if config.point_cloud_noise != 0:
        surface_points = [
            surface_points_ + torch.randn_like(surface_points_) * config.point_cloud_noise
            for surface_points_ in surface_points
        ]

    print('\n\033[96mModel\033[0m:\n', model)

    # initialize with random normal
    if config.wandb_project is not None:
        wandb_loggers = [wandb.init(
            project=config.wandb_project,
            name=f'{config.experiment_name}_{shape_id}',
            dir=config.output_dir,
            config=asdict(config),
            reinit='create_new'
        ) for shape_id in shape_ids]

    results_per_shape = [[] for _ in range(num_shapes)]
    latents_per_step = []

    if config.initialization == 'random_normal':
        init_latents = torch.randn(num_shapes, 512, 8, device='cuda')
    elif config.initialization == 'encoder':
        init_latents = []
        for shape_index, surface_points_ in zip(shape_indices, surface_points):
            with temp_seed(config.random_seed + shape_index):
                init_latents.append(autoencoder.encode((
                    surface_points_ + torch.randn_like(surface_points_) * config.init_point_cloud_noise)[None]
                )['x'])
        init_latents = torch.concatenate(init_latents)
    else:
        raise ValueError(f'Unknown initialization: {config.initialization}')

    state = config.method.init_state(
        init_latents=init_latents,
        class_labels=class_labels,
        random_seed=[i + config.random_seed for i in shape_indices],
    )
    np.save(experiment_dir / 'latents_init.npy', state.latents.detach().clone().cpu())
    for epoch in tqdm(range(config.n_steps)):
        idx = [
            sample_indices(
                surface_points_,
                config.batch_size,
                state.random_generator.generators[shape_idx],
            )
            for shape_idx, surface_points_ in enumerate(surface_points)
        ]

        query_points = torch.stack([surface_points_[idx_] for surface_points_, idx_ in zip(surface_points, idx)])

        state, metrics = config.method.step(
            query_points,
            autoencoder,
            model,
            state,
        )
        for i in range(num_shapes):
            batch_results = {
                'epoch': epoch,
                **{metric_name: metric[i].item() for metric_name, metric in metrics.items()},
                'std': state.latents[i].std().item(),
            }
            results_per_shape[i].append(batch_results)
            if config.wandb_project is not None:
                wandb_loggers[i].log(batch_results)
        if epoch % config.checkpoint_step == 0:
            latents_per_step.append(state.latents.detach().clone().cpu())
    latents_per_step = torch.stack(latents_per_step)

    np.save(experiment_dir / 'latents.npy', latents_per_step.numpy())
    np.save(experiment_dir / 'latents_final.npy', state.latents.detach().clone().cpu())
    results = pd.concat([
        pd.DataFrame(results).assign(model_id=shape_id)
        for results, shape_id in zip(results_per_shape, shape_ids)
    ])
    results.to_csv(experiment_dir / 'train_metrics.csv')

    print('Done! Generating visualizations')
    images, eval_metrics_per_shape = [], []
    for shape_index in tqdm(range(num_shapes)):
        output_dir = shape_output_dirs[shape_index]
        shape_id = shape_ids[shape_index]

        bu.read_homefile()
        if config.save_mesh:
            occupancies = vecset_diffusion.geometry.batch_query(
                autoencoder.decode,
                state.latents[shape_index][None].to('cuda'),
                density=config.mc_resolution,
                device='cuda',
                batch_size=100000,
                max_val=1,
            )
            verts, faces = vecset_diffusion.geometry.get_mesh(occupancies)
            mesh = trimesh.Trimesh(verts, faces)
            mesh.export(output_dir / 'shape_final.obj')

            image_path = experiment_dir / f'images/final_{shape_id}.png'
            bu.render_mesh(
                output_path=image_path,
                mesh=mesh,
                pointcloud=surface_points[shape_index].detach().cpu().numpy(),
            )
            images.append(np.array(Image.open(image_path)))

            if config.pointcloud_path.suffix == '.json':
                # compute evaluation metrics
                gt_data = np.load(config.pointcloud_path.parent / f'gt_points/{shape_ids[shape_index]}.npz')
                if len(mesh.vertices) > 0:
                    sample_points, sample_face_indices = trimesh.sample.sample_surface(mesh, 30_000)
                    sample_normals = mesh.face_normals[sample_face_indices]
                else:
                    sample_points, sample_normals = np.zeros((0, 3)), np.zeros((0, 3))

                eval_metrics = vecset_diffusion.chamfer.compute_chamfer(
                    sample_points * (scale_factors[shape_index] if config.metrics_gt_scale else 1.),
                    sample_normals,
                    gt_data['points'] / (scale_factors[shape_index] if not config.metrics_gt_scale else 1.),
                    gt_data['normals'],
                )
                eval_metrics_per_shape.append(eval_metrics)
                pd.DataFrame([eval_metrics]).to_csv(output_dir / 'eval_metrics.csv')
                pd.DataFrame([eval_metrics]).to_csv(output_dir / 'eval_metrics_final.csv')

        if config.save_animation:
            bu.create_animation(
                surface_points=surface_points[shape_index],
                latents_per_step=latents_per_step[::5, shape_index],
                autoencoder=autoencoder,
                output_dir=output_dir / f'animations',
            )
    if config.save_mesh:
        bu.make_image_grid(
            output_path=experiment_dir / 'images/grid.png',
            images=images,
            image_names=shape_ids,
            eval_metrics=eval_metrics_per_shape,
            fig_title=config.experiment_name
        )
    return experiment_dir, results_per_shape

if __name__ == '__main__':
    config = tyro.cli(TrainingConfig)
    run(config)
