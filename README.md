# EndoFlow: Flow-Guided Gaussian Deformation Learning for Surgical Scene Reconstruction

Official code implementation for EndoFlow, a flow-guided deformable 3D Gaussian Splatting framework for dynamic surgical scene reconstruction.

> Flow-Guided Gaussian Deformation Learning for Surgical Scene Reconstruction  
> Department of Computer Engineering, Faculty of Engineering, Chulalongkorn University, Bangkok 10330, Thailand

## Overview

This project presents Flow-Guided Gaussian Deformation Learning, a framework designed to improve non-rigid dynamic surgical scene reconstruction using deformable 3D Gaussian Splatting.

The method addresses the under-constrained nature of photometric-only optimization in monocular endoscopic settings, where tissue motion, camera movement, and visibility changes make accurate deformation modeling difficult. EndoFlow introduces flow guidance into Gaussian deformation learning to better supervise the temporal evolution of scene geometry and motion.

The core idea is to reverse the conventional motion-transfer direction: instead of transferring Gaussian motion to pixels, the framework aggregates motion information at the Gaussian level and directly supervises Gaussian deformation using optical flow. This is combined with a depth-conditioned Flow-Guided Gaussian Transport (FGGT) module that lifts 2D motion into 3D and enables bidirectional mean fusion within the differentiable rendering pipeline.

### Key contributions

- Camera-motion decoupling to isolate tissue deformation from camera-induced motion.
- Reverse per-Gaussian motion aggregation for flow-driven deformation learning.
- Direct Gaussian-level optical-flow supervision via the projected Gaussian flow loss.
- Flow-Guided Gaussian Transport (FGGT) to lift 2D flow into 3D using model-predicted target depths.
- Bidirectional Gaussian mean fusion within forward rendering for more stable optimization.

## Pipeline

[flow_guided_pipeline.pdf](https://github.com/user-attachments/files/32809410/flow_guided_pipeline.pdf)

EndoFlow consists of three main stages:

1. Point cloud initialization from the input surgical sequence.
2. Flow-guided deformation learning, including optical-flow estimation, camera-motion decoupling, reverse flow aggregation, and Gaussian-level supervision.
3. Depth-aware rendering and optimization using the differentiable Gaussian rendering pipeline.

## Environment setup

Tested on NVIDIA RTX A5000 GPU.

```bash
git clone https://github.com/hhplinh/EndoFlow.git
cd EndoFlow

conda create -n EndoFlow python=3.7
conda activate EndoFlow

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install -r requirements.txt
pip install -e submodules/diff-plane-rasterization
pip install -e submodules/simple-knn
```

The repository also includes the RAFT optical flow submodule under `submodules/RAFT`. Please refer to its README for pretrained model download and inference instructions if needed.

## Datasets

We evaluate EndoFlow on dynamic surgical benchmarks from:

- [EndoNeRF](https://github.com/med-air/EndoNeRF)
- [StereoMIS](https://zenodo.org/records/7727692)

### EndoNeRF dataset

Two clips are used for evaluation. Please download the data via the request form:

- [EndoNeRF dataset access](https://forms.gle/1VAqDJTEqZduD6157)

Organize the data under:

```text
data/
└── endonerf_full_datasets/
    ├── cutting_tissues_twice/
    │   ├── depth/
    │   ├── images/
    │   ├── masks/
    │   └── pose_bounds.npy
    └── pushing_soft_tissues/
```

### StereoMIS dataset

Three clips are used from the StereoMIS dataset. Please follow the preprocessing instructions in the [robust-pose-estimator repository](https://github.com/aimi-lab/robust-pose-estimator), then organize the processed sequences under:

```text
data/
└── StereoMIS/
    ├── stereo_seq_1/
    ├── stereo_seq_2/
    └── stereo_seq_3/
```

## Training

Training hyperparameters are typically configured in:

```text
arguments/endonerf/default.py
```

To train EndoFlow on an EndoNeRF sequence, run:

```bash
python train.py \
  -s data/endonerf_full_datasets/pulling_soft_tissues \
  --expname endonerf/pulling_flow_guided \
  --configs arguments/endonerf/default.py
```

You may need to adjust the dataset path, experiment name, and configuration file depending on the target sequence or benchmark.

## Testing

Rendering and evaluation are performed separately.

### Rendering

To render the test set:

```bash
python render.py \
  --model_path output/endonerf/pulling_flow_guided \
  --skip_train \
  --reconstruct_test \
  --configs arguments/endonerf/default.py
```

Other common reconstruction modes include:

- `--reconstruct_train`
- `--reconstruct_test`
- `--reconstruct_video`

### Evaluation

To evaluate reconstruction quality:

```bash
python metrics.py \
  --model_path output/endonerf/pulling_flow_guided \
  -p test
```

The `-p` option selects the evaluation split:

- `-p train`
- `-p test`
- `-p video`

## Method summary

EndoFlow is designed around the observation that optical flow offers informative motion cues for deformable surgical tissues, but these cues are not directly usable as a full 3D deformation field. The framework addresses this by:

1. Estimating optical flow between neighboring frames.
2. Decoupling camera motion from non-rigid tissue motion.
3. Aggregating motion in reverse from image space to the Gaussian primitives contributing to each observation.
4. Directly supervising Gaussian motion with a projected Gaussian flow loss.
5. Transporting Gaussian positions using the FGGT module and model-predicted target depth.
6. Fusing transported and forward Gaussian means within the differentiable rendering pipeline.

This depth-conditioned formulation provides a lightweight alternative to full 3D scene-flow modeling while maintaining explicit motion supervision for deformable Gaussian primitives.

## Acknowledgements

This repository builds on code and ideas from:

- [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS/tree/master)
- [EndoGaussian](https://github.com/yifliu3/EndoGaussian/tree/master)
- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

We thank the authors for making their frameworks publicly available.

## Citation

If you find this code useful for your research, please cite the relevant EndoFlow publication. A placeholder BibTeX entry is shown below:

```bibtex
```
