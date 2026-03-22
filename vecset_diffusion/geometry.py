import numpy as np
import torch
import mcubes
from tqdm import tqdm

def batch_query(decoder, latents, density, batch_size=100000, device='cuda', progress_bar=True, max_val=1.):
    x = np.linspace(-max_val, max_val, density+1)
    y = np.linspace(-max_val, max_val, density+1)
    z = np.linspace(-max_val, max_val, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    with torch.no_grad():
        occupancies_list = []
        iterator = range(0, grid.shape[1], batch_size)
        if progress_bar:
            iterator = tqdm(iterator, leave=False, desc='decoding shape')
        for i in iterator:
            batch_grid = grid[:, i:i+batch_size]
            batch_occupancies = decoder(latents.to(device), batch_grid).detach()
            occupancies_list.append(batch_occupancies)

        occupancies_tensor = torch.cat(occupancies_list, dim=1)
    return occupancies_tensor

def get_mesh(occupancies, max_val=1.):
    density = int(occupancies.shape[1] ** (1 / 3))
    volume = occupancies.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
    verts, faces = mcubes.marching_cubes(volume, 0)
    verts = verts * (2 * max_val) / density - max_val
    return verts, faces