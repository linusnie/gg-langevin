import vecset_diffusion
from vecset_diffusion.models import NoisePredictor, ModelConfig
import train_autoencoder
import utils

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

@dataclass
class TrainingConfig:
	output_dir: Path
	dataset_path: Path
	autoencoder: Path
	experiment_name: str = 'experiment'
	with_timestamp: bool = True

	# Path to yaml config with default settings
	yaml_config: Optional[Path] = None
	checkpoint_path: Optional[Path] = None

	# if True, the provided model config will override the checkpoint config (may crash if incompatible!)
	checkpoint_config_override: bool = False

	n_epochs: int = 1000
	batch_size: int = 1024
	num_workers: int = 0
	categories: Optional[List[str]] = None

	learning_rate: float = 0.0001
	learning_rate_warmup: bool = False
	warmup_epochs: int = 5
	beta1: float = 0.9
	beta2: float = 0.999
	weight_decay: float = 0.0
	optimizer: str = 'adam'
	clip_grad: Optional[float] = None

	encoder_mode: str = 'encode'
	surface_normalization: str = 'cube'
	latent_mean: Optional[Path] = None
	latent_std: float = 1.0

	# percentage of input tokens to drop in each batch
	token_dropout: float = 0.

	log_sigma_mean: float = -1.2
	log_sigma_std: float = 1.2

	model_config: ModelConfig = field(default_factory=lambda: ModelConfig())

	checkpoint_step: int = 10

	wandb_project: Optional[str] = None

def should_skip_batch(model, max_norm=1.0):
	"""Returns True if the gradient norm exceeds max_norm."""
	if max_norm is None:
		return False
	total_norm = 0.0
	for param in model.parameters():
		param_norm = param.grad.data.norm()
		total_norm += param_norm.item() ** 2
	total_norm = total_norm ** (1.0 / 2)
	return total_norm > max_norm

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

	print(f'---\033[93m{config.experiment_name}\033[0m---')
	print(f'\033[96mexperiment dir\033[0m: {experiment_dir}')

	if config.checkpoint_path is None:
		model = NoisePredictor(config.model_config).to('cuda')
		optimizer_state, epoch_idx = None, 0
	else:
		print(f'Loading model from {str(config.checkpoint_path)}...')
		tmp_config, model, optimizer_state, epoch_idx = utils.load_model_checkpoint(
			config.checkpoint_path,
			config=config if config.checkpoint_config_override else None
		)
		config.model_config = tmp_config.model_config

	if is_distributed:
		print('initializing distributed')
		model = torch.nn.parallel.DistributedDataParallel(model, [local_rank])
		print('initialized')

	utils.print_config(config)
	print('\n\033[96mModel\033[0m:\n', model)

	if is_main_process:
		utils.save_config(config, experiment_dir)

	results = []
	global_batch_size = config.batch_size * world_size
	print("Loading dataset...")
	if config.dataset_path.is_file():
		latents = np.load(config.dataset_path)
		latent_mean = latents.mean(0)
		latent_std = (latents - latent_mean).std()
		normalized_latents = (latents - latent_mean) / latent_std
		print(f'\033[96mDataset size\033[0m: {len(latents)}')
		print(f'\033[96mLatent mean\033[0m: {np.abs(latent_mean).mean()}')
		print(f'\033[96mLatent stddev\033[0m: {latent_std}')
		data_loader = partial(utils.iter_latents,
			latents=normalized_latents,
			num_batches=len(latents) // global_batch_size,
			batch_size=config.batch_size,
			leave=False, disable=not is_main_process,
		)
		dataset_size = len(normalized_latents)
	else:
		dataset = vecset_diffusion.shapenet.ShapeNet(
			config.dataset_path,
			split='train',
			transform=None,
			sampling=True,
			num_samples=1024,
			return_surface=True,
			surface_sampling=True,
			return_only_surface=True,
			pc_size=8192,
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
		ae_config = utils.load_config(
			Path(config.autoencoder).parent.parent,
			train_autoencoder.TrainingConfig,
		)
		autoencoder = vecset_diffusion.autoencoder.create_autoencoder_from_config(ae_config.model_config)
		autoencoder_checkpoint = torch.load(
			config.autoencoder,
			map_location='cpu',
			weights_only=False,
		)['model']

		autoencoder_checkpoint = {
			k[7:] if k.startswith('module.') else k: v
			for k, v in autoencoder_checkpoint.items()
		}
		autoencoder.load_state_dict({k: v for k, v in autoencoder_checkpoint.items() if k != 'latent_inds'})
		autoencoder = autoencoder.to('cuda')
		autoencoder.eval()
		data_loader = partial(utils.iter_dataset, autoencoder, data_loader, leave=False, disable=not is_main_process)
		dataset_size = len(dataset)

	if config.optimizer == 'adam':
		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=config.learning_rate,
			betas=(config.beta1, config.beta2),
			weight_decay=config.weight_decay,
		)
	elif config.optimizer == 'sgd':
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=config.learning_rate,
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
			config={
				'num_gpus': world_size,
				**asdict(config),
			},
		)
	utils.save_checkpoint(checkpoint_dir, model, optimizer, 'init', 'init', 'init')

	start_epoch = epoch_idx
	debug_dump = False
	sample_idx = dataset_size * epoch_idx
	global_batch_idx = (dataset_size // global_batch_size) * epoch_idx

	scheduler.last_epoch = sample_idx
	scheduler.step()

	for epoch_idx in tqdm(range(epoch_idx, epoch_idx + config.n_epochs), disable=not is_main_process):
		for batch_idx, (batch_latents, batch_categories) in enumerate(data_loader()):
			token_indices = np.random.choice(
				range(batch_latents.shape[1]),
				size=int(batch_latents.shape[1] * (1 - config.token_dropout)),
				replace=False,
			)
			batch_latents = batch_latents[:, token_indices]
			sigmas = (torch.randn(len(batch_latents)) * config.log_sigma_std + config.log_sigma_mean).exp().to('cuda')
			noise = torch.randn_like(batch_latents)

			denoised_latents, model_output = model(
				x=batch_latents + noise * sigmas[..., None, None],
				sigma=sigmas,
				class_labels=batch_categories,
				indices=token_indices,
				sequence_length=batch_latents.shape[1],
			)

			weights = 1 + sigmas ** 2
			losses = (weights[..., None, None] * (denoised_latents - noise) ** 2).mean(dim=-1).mean(dim=-1)
			loss = losses.mean()

			loss.backward()
			optimizer.step()

			if is_distributed:
				torch.distributed.all_reduce(loss)
				loss /= world_size # assumes batch size is equal for all ranks

			if is_main_process:
				batch_results = {
					'loss': loss.item(),
					'batch_idx': batch_idx,
					'global_batch_idx': global_batch_idx,
					'sample_idx': sample_idx,
					'epoch_idx': epoch_idx + batch_idx * global_batch_size / dataset_size,
					'model_output_norm': model_output.pow(2).mean(-1).mean(-1).sqrt().mean().item(),
					**{f'lr_{i}': param_group['lr'] for i, param_group in enumerate(optimizer.state_dict()['param_groups'])},
				}
				results.append(batch_results)
				if config.wandb_project is not None:
					wandb.log(batch_results)

			optimizer.zero_grad()
			scheduler.step()
			global_batch_idx += 1
			sample_idx += len(batch_latents) * world_size

		# end of epoch
		if is_main_process:
			if debug_dump:
				pd.DataFrame(results).to_csv(experiment_dir / 'metrics.csv')
				utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_idx, global_batch_idx, sample_idx)
				np.savez(
					experiment_dir / 'debug_batch.npz',
					batch_latents=batch_latents.detach().cpu().numpy(),
					losses=losses.detach().cpu().numpy(),
					sigmas=sigmas.detach().cpu().numpy(),
					noise=noise.detach().cpu().numpy(),
				)
				raise ValueError("NaN gradients. Saving debug dump.")
			if (epoch_idx % config.checkpoint_step == 0) or (epoch_idx == start_epoch + config.n_epochs - 1):
				pd.DataFrame(results).to_csv(experiment_dir / 'metrics.csv')
				utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_idx, global_batch_idx, sample_idx)

	return experiment_dir, pd.DataFrame(results)

if __name__ == '__main__':
	config = tyro.cli(TrainingConfig)
	run(config)
