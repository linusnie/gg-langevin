"""
  ShapeNet dataset, we modify from vecset to work directly with .npz files
"""
import os
import torch
from torch.utils import data

import numpy as np

# vecSetX was trained using sphere normalisation
# Data was preprocessed using box normalisation
# We need to change scale properly
class SphereNormalization(object):
    def __init__(self):
        pass
    def __call__(self, surface):
        distances = np.linalg.norm(surface, axis=1)
        scale = 1 / np.max(distances)
        surface *= scale
        return surface

class CubeNormalization(object):
    def __init__(self):
        pass
    def __call__(self, surface):
        scale = 1 / np.max(surface) * 0.999999
        surface *= scale
        return surface

class ShapeNetTransform:
    def __init__(self, normalization, apply_random_scaling=False, apply_random_jitter=False, apply_random_rotation=False):
        """
            IF only normalisaition is passed apply only normalisation, otherwise apply augmentations too
            normalization: 'cube' or 'sphere'
        """
        assert normalization in ['cube', 'sphere'], f'Invalid normalization: {normalization}'
        self.normalization = normalization
        self.apply_random_scaling = apply_random_scaling
        self.apply_random_jitter = apply_random_jitter
        self.apply_random_rotation = apply_random_rotation

    def __call__(self, surface, point=None, sdfs=None):
        if self.apply_random_scaling and (sdfs is None):
            # axis scaling only applied when running without sdfs
            surface, point = self.random_axis_scaling(surface, point)
            # TODO: could try axis scaling after normalization for OOD

        if self.apply_random_rotation:
            # rotation applied before normalization to maintain relative geometry
            surface, point = self.random_rotation(surface, point)

        if self.normalization == 'cube':
            scale = self.get_cube_normalization_scale(surface)
        else:
            scale = self.get_sphere_normalisation_scale(surface)

        surface = (surface * scale).to(torch.float32)
        if self.apply_random_jitter:
            surface = self.random_jitter(surface)
        outs = [surface]

        if point is not None:
            point = (point * scale).to(torch.float32)
            outs.append(point)
        if sdfs is not None:
            sdfs = (sdfs * scale).to(torch.float32)
            outs.append(sdfs)

        return outs[0] if len(outs) == 1 else tuple(outs)

    @staticmethod
    def random_scaling(*args, n_axises=1):
        scaling = np.random.rand(1, n_axises) * 0.5 + 0.75
        return [arg * scaling for arg in args]

    @staticmethod
    def random_axis_scaling(*args):
        return ShapeNetTransform.random_scaling(*args, n_axises=3)

    @staticmethod
    def get_cube_normalization_scale(surface):
        scale = (1 / torch.abs(surface).max()) * 0.999999
        return scale

    @staticmethod
    def get_sphere_normalisation_scale(surface):
        distances = surface.norm(dim=1)
        scale = 1 / distances.max()
        return scale

    @staticmethod
    def random_jitter(surface):
        jitter = 0.005 * torch.randn(*surface.shape)
        surface += jitter
        # surface = torchp.clip(surface, -1, 1)
        return surface

    @staticmethod
    def random_rotation(*args):
        # Generate random rotation matrix
        x1, x2 = np.random.random(2) * 2 * np.pi
        x3 = np.random.random()

        rz = np.array([
            [np.cos(x1), np.sin(x1), 0.],
            [-np.sin(x1), np.cos(x1), 0.],
            [0., 0., 1.]
        ])

        v = np.array([
            np.cos(x2) * np.sqrt(x3),
            np.sin(x2) * np.sqrt(x3),
            np.sqrt(1 - x3)
        ])

        rotation_matrix = -(np.eye(3) - 2 * np.outer(v, v)) @ rz

        # Apply rotation to all arguments (surface, points, etc.)
        rotated_args = []
        for arg in args:
            if arg is not None:
                # Handle both torch tensors and numpy arrays
                if isinstance(arg, torch.Tensor):
                    arg = arg.cpu().numpy()
                    rotated_arg = (rotation_matrix @ arg.T).T
                    rotated_args.append(torch.from_numpy(rotated_arg).float())
                else:
                    rotated_arg = (rotation_matrix @ arg.T).T
                    rotated_args.append(rotated_arg.astype(np.float32))
            else:
                rotated_args.append(None)

        return rotated_args if len(rotated_args) > 1 else rotated_args[0]

class ShapeNet(data.Dataset):
    SHAPENET_SPLIT_FOLDER = '/storage/user/pozd/shapenet/shapenet_split'
    def __init__(
            self,
            dataset_folder,
            split,
            data_type = "sdf",
            categories = None,
            transform = None,
            sampling = True,
            num_samples = 4096,
            return_surface = True,
            surface_sampling = True,
            pc_size = 2048,
            return_only_surface: bool = False,
            normalization: str = 'sphere',
        ):
        """ Class for the ShapeNet dataset
            Args:
              dataset_folder: str path to the dataset folder with train/val/test subfolders
              split: str, train/val/test
              categories: list of str, categories to load
              transform: callable, transform to apply to the data
              sampling: bool, whether to sample points
              num_samples: int, number of samples to take from vol and near points
              return_surface: bool, whether to return the surface
              surface_sampling: bool, whether to sample the from vol and near points using num_samples
              pc_size: int, size of the surface point (sampling from surface points)
              data_type: str, type of data to load, either 'sdf' or 'occupancy'
              return_only_surface: bool, whether to return only the surface points (used for diffusion training)
        """
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.data_folder = os.path.join(dataset_folder, split)
        print('Shapenet Loading data from {}'.format(self.data_folder))

        if categories is None:
            categories = os.listdir(self.data_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.data_folder, c)) and c.startswith('0')]
        categories.sort()

        print(categories)

        self.models = []
        for c_idx, c in enumerate(categories):
            model_ids = sorted(os.listdir(os.path.join(self.data_folder, c)))
            self.models += [
                {'category': c, 'model': m.replace('.npz', '')}
                for m in model_ids
            ]

        assert data_type in ['sdf', 'occupancy'], f'Invalid data type: {data_type}'
        self.data_type = "sdf" if data_type == "sdf" else 'label' # occpaunices should be saved like label

        self.return_only_surface = return_only_surface
        if normalization == 'sphere':
            self.normalisation_transform = SphereNormalization()
        elif normalization == 'cube':
            self.normalisation_transform = CubeNormalization()
        elif normalization == 'none':
            self.normalisation_transform = torch.nn.Identity()
        else:
            raise ValueError(f'Normalization method "{normalization}" not recognized.')

        if self.return_only_surface:
            assert self.return_surface, 'return_only_surface is True, but return_surface is False. Set return_surface to True'

        print(f'Shapenet dataset length {len(self.models)}')

    def __getitem__(self, idx):
        category = self.models[idx]['category']
        model_id = self.models[idx]['model']

        data_path = os.path.join(self.data_folder, category, model_id+'.npz')
        try:
            with np.load(data_path) as data:
                if not self.return_only_surface:
                    vol_points = data['vol_points']
                    vol_sdf = data[f'vol_{self.data_type}']
                    near_points = data['near_points']
                    near_sdf = data[f'near_{self.data_type}']
                else:
                    vol_points, vol_sdf, near_points, near_sdf = None, None, None, None
                surface = data['surface_points'].astype(np.float32)

                # if self.data_type == "sdf":
                #     assert near_sdf.min() >= -0.5 and near_sdf.max() <= 0.5 and vol_sdf.min() >= -1 and vol_sdf.max() <= 1, f'Near SDF out of range: {near_sdf.min()} {near_sdf.max()} Vol SDF out of range: {vol_sdf.min()} {vol_sdf.max()}'
                if self.data_type == "label":
                    assert np.isin(near_sdf, [0, 1]).all() and np.isin(vol_sdf, [0, 1]).all(), f'Near SDF out of range: {near_sdf.min()} {near_sdf.max()} Vol SDF out of range: {vol_sdf.min()} {vol_sdf.max()}'

        except Exception as e:
            print(e)
            print(data_path)
            exit(0)

        if self.return_surface:
            # we normalise full surface only for diffusion training set. For VAE we also need to normalise the points and sdfs
            if self.return_only_surface:
                surface = self.normalisation_transform(surface)
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.return_only_surface:
            # NOTE that we can only apply transofrms that keep SDF (rotations, shifts, uniform scaling, combining two classes)
            if self.transform:
                surface, _ = self.transform(surface, None)
            return torch.empty(0), torch.empty(0), surface, category_ids[category]

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_sdf = vol_sdf[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_sdf = near_sdf[ind]


        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        near_points = torch.from_numpy(near_points)
        near_sdf = torch.from_numpy(near_sdf).float()

        points = torch.cat([vol_points, near_points], dim=0)
        sdfs = torch.cat([vol_sdf, near_sdf], dim=0)

        # NOTE that we can only apply transofrms that keep SDF (rotations, shifts, uniform scaling, combining two classes)
        if self.transform:
            surface, points = self.transform(surface, points)
        if self.return_surface:
            return points, sdfs, surface, category_ids[category]
        else:
            return points, sdfs, category_ids[category]

    def __len__(self):
        return len(self.models)

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9,
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}