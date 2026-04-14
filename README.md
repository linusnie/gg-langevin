# Generative Shape Reconstruction with Geometry-Guided Langevin Dynamics

> [Linus Härenstam-Nielsen](https://linusnie.github.io/), [Dmitrii Pozdeev](https://diddone.github.io/), [Thomas Dagès](https://tommoo.github.io/), [Nikita Araslanov](https://arnike.github.io/) and [Daniel Cremers](https://vision.in.tum.de/members/cremers)
>
> Technical University of Munich, Munich Center for Machine Learning
>
> 📄 [Paper](https://arxiv.org/abs/2603.27016)

<p align="center"><img src="images/overview.svg" alt="teaser" style="clip-path: inset(0 0 3% 2%); max-width: 100%;"></p>

This repository contains the official implementation of GG-Langevin from the paper **Generative Shape Reconstruction with Geometry-Guided Langevin Dynamics**.
A probabilistic approach to Generative 3D Reconstruction, leveraging a pre-trained shape diffusion model as prior.

**Abstract:**
Reconstructing complete 3D shapes from incomplete or noisy observations is a fundamentally ill-posed problem that requires balancing measurement consistency with shape plausibility. Existing methods for shape reconstruction can achieve strong geometric fidelity in ideal conditions but fail under realistic conditions with incomplete measurements or noise. At the same time, recent generative models for 3D shapes can synthesize highly realistic and detailed shapes but fail to be consistent with observed measurements. In this work, we introduce GG-Langevin: Geometry-Guided Langevin dynamics (GGL for short), a novel probabilistic approach that unifies these complementary perspectives. By traversing the trajectories of Langevin dynamics induced by a diffusion model, while preserving measurement consistency at every step, we generatively reconstruct shapes that fit both the measurements and the data-informed prior. We demonstrate through extensive experiments that GG-Langevin achieves higher geometric accuracy and greater robustness to missing data than existing methods for surface reconstruction. Our project code, trained models, and benchmark are available in this repository.

## Installation

```bash
conda create -n ggl python=3.10
conda activate ggl

# Install PyTorch with CUDA (match your CUDA version, verified with 12.6)
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

# Install torch_cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

### Optional: Flash Attention

Flash Attention is not installed by default. The code falls back to standard attention when unavailable. To install for faster training/inference on GPUs with compute capability ≥ 8.0:

```bash
conda install cuda-nvcc=12.6 -c nvidia -y
pip install flash-attn --no-build-isolation
```

## Download data and models
The datasets and models (diffusion models and autoencoder) can be found [here](https://drive.google.com/drive/folders/1XkmkJe4O4ALGuHX0js4fmPck9b7w3S-q?usp=sharing).
Download and extract with:
```bash
python -m gdown --folder https://drive.google.com/drive/folders/1wYQFFDR4LZHWtfuUxiF040y4gunVHPlu?usp=sharing
unzip ggl/dataset.zip -d ggl/dataset
rm ggl/dataset.zip
```

## Running GG-Langevin
GG-Langevin takes a pointcloud .npy file (or a datalist specifying multiple point clouds) as input, and outputs the estimated shape.
To run GG-Langevin on the sparse Chairs category, use the following command:

```bash
python fit_latents.py \
	--output-dir results \
	--pointcloud-path ggl/dataset/sparse_chairs/datalist.json \
	--model-path ggl/models/unconditioned/checkpoints/checkpoint_5000.pt \
	--autoencoder-path ggl/models/autoencoder/checkpoints/checkpoint_4500.pt \
	--yaml-config configs/sparse/config_ggl.yaml
```

See also experiments.ipynb for an example.

