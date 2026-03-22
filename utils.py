import torch

import os
import numpy as np
from tqdm import tqdm
import pickle
import tyro
from copy import deepcopy

import vecset_diffusion
import train_diffusion
import train_autoencoder

def print_config(config):
    for key, value in config.__dict__.items():
        print(f'\033[96m{key}\033[0m: {value}')

def save_config(config, output_dir, name='config'):
    with open(output_dir / f'{name}.yaml', 'w') as yaml_file:
        yaml_file.write(tyro.extras.to_yaml(config))

    # save as pickle as well since yaml loading can break between versions
    with open(output_dir / f'{name}.pickle', 'wb') as pickle_file:
        pickle.dump(config, pickle_file)

def load_config(experiment_dir, config_class, name='config'):
    try:
        with open(experiment_dir / f'{name}.yaml', 'r') as yaml_file:
            return tyro.extras.from_yaml(config_class, yaml_file)
    except Exception as e:
        print(f'WARNING: failed to load config from yaml config from {experiment_dir} due to "{e}", probably a result of version mismatch. Loading pickle file instead.')
        with open(experiment_dir / f'{name}.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)

def iter_dataset(autoencoder, data_loader, **kwargs):
    for _, _, surface, categories in tqdm(data_loader, total=len(data_loader), **kwargs):
        surface = surface.to('cuda')
        categories = categories.to('cuda')
        autoencoder.num_inputs = surface.shape[1]
        with torch.no_grad():
            batch_latents = autoencoder.encode(surface)['x'] # VecSetX
        yield batch_latents, categories


def iter_latents(latents, num_batches, batch_size, **kwargs):
    for _ in tqdm(range(num_batches), **kwargs):
        batch_indices = np.random.choice(
            range(len(latents)),
            size=batch_size,
            replace=False
        )
        batch_latents = torch.tensor(latents[batch_indices]).to('cuda')
        yield batch_latents, (torch.ones(batch_size) * 18).to('cuda', torch.int) # TODO: support multiclass

def save_checkpoint(checkpoint_dir, model, optimizer, epoch_idx, global_batch_idx, sample_idx):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_idx': epoch_idx,
            'global_batch_idx': global_batch_idx,
            'sample_idx': sample_idx,
        },
        checkpoint_dir / f'checkpoint_{epoch_idx}.pt',
    )

def load_model_checkpoint(checkpoint_path, config=None):
    strict = config is None
    if config is None:
        config = load_config(checkpoint_path.parent.parent, train_diffusion.TrainingConfig)
    model_config = config.model_config
    model = vecset_diffusion.models.NoisePredictor(model_config).to('cuda')

    checkpoint = torch.load(checkpoint_path)
    missing, unexpected = model.load_state_dict({
        k[7:] if k.startswith('module.') else k: v
        for k, v in checkpoint['model'].items()
    }, strict=strict)
    print('missing checkpoint keys:', missing)
    print('unexpected checkpoint keys:', unexpected)

    # reset optimizer state if model has changed
    if (len(missing) == 0) and (len(unexpected) == 0):
        optimizer_state = checkpoint['optimizer']
    else:
        optimizer_state = None

    epoch_idx = int(checkpoint_path.parts[-1].split('_')[-1][:-3])
    return config, model, optimizer_state, epoch_idx

def map_3dshapekeys_to_vecsetkeys(orig_state_dict):
    state_dict = deepcopy(orig_state_dict)
    bottleneck_keys = [
        "logvar_fc.bias", "logvar_fc.weight", "mean_fc.bias", "mean_fc.weight", "proj.weight", "proj.bias"
    ]
    for key in bottleneck_keys:
        state_dict[f"bottleneck.{key}"] = state_dict[key]
        del state_dict[key]

    return state_dict

def load_autoencoder_checkpoint(checkpoint_path, model_config=None, attention_class=None):
    if model_config is None:
        model_config = load_config(checkpoint_path.parent.parent, train_autoencoder.TrainingConfig).model_config

    print(model_config)
    autoencoder = vecset_diffusion.autoencoder.create_autoencoder(
        depth=24, dim=512, M=512, N=1,
        query_type=model_config.query_type,
        attention_class=attention_class,
        final_layer_norm=not model_config.original_vecset,
        context_norm=model_config.original_vecset,
        single_cross_attend_head=model_config.original_vecset,
        bottleneck=vecset_diffusion.autoencoder.KLBottleneck,
        bottleneck_layer=model_config.bottleneck_layer,
        bottleneck_args={'dim': 512, 'latent_dim': 8, 'kl_weight': 1.},
    )

    checkpoint = torch.load(checkpoint_path)
    model_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in checkpoint['model'].items()
    }
    if model_config.original_vecset:
        model_dict = map_3dshapekeys_to_vecsetkeys(model_dict)
        epoch_idx = 0
    else:
        epoch_idx = int(checkpoint_path.parts[-1].split('_')[-1][:-3])
    autoencoder.load_state_dict(model_dict)
    autoencoder = autoencoder.to('cuda')
    return model_config, autoencoder, checkpoint['optimizer'], epoch_idx

def initialize_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    is_main_process = rank == 0
    is_distributed = world_size > 1

    torch.cuda.set_device(local_rank)
    if is_distributed:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # disable printing outside of main process
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, force=False, **kwargs):
        if is_main_process or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

    return world_size, local_rank, is_main_process, is_distributed

