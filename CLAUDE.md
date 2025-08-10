# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MotionBERT is a unified framework for learning human motion representations from 2D/3D pose sequences. The architecture is based on the DSTformer (Dual-Scale Transformer) model that can handle various downstream tasks including 3D pose estimation, action recognition, and mesh recovery.

## Core Architecture

- **DSTformer**: Main transformer-based model (`lib/model/DSTformer.py`) that processes motion sequences
- **Task-specific heads**: Different output layers for various tasks (action classification, mesh regression, pose estimation)
- **Dataset classes**: Specialized loaders for different data types (2D motion, 3D motion, mesh, action, wild videos)

## Common Commands

### Environment Setup
```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Training Commands
```bash
# Pretraining
python train.py --config configs/pretrain/MB_pretrain.yaml -c checkpoint/pretrain/MB_pretrain

# 3D pose estimation
python train.py --config configs/pose3d/MB_ft_h36m.yaml -c checkpoint/pose3d/MB_ft_h36m

# Action recognition
python train_action.py --config configs/action/MB_ft_NTU60_xsub.yaml -c checkpoint/action/MB_ft_NTU60_xsub

# Mesh recovery
python train_mesh.py --config configs/mesh/MB_ft_pw3d.yaml -c checkpoint/mesh/MB_ft_pw3d
```

### Inference
```bash
# In-the-wild pose estimation
python infer_wild.py --vid_path <video_path> --out_path <output_path>

# In-the-wild mesh recovery
python infer_wild_mesh.py --vid_path <video_path> --out_path <output_path>
```

## Key Configuration Files

All training configurations use YAML files in the `configs/` directory:
- `configs/pretrain/`: Pretraining configurations
- `configs/pose3d/`: 3D pose estimation configs
- `configs/action/`: Action recognition configs  
- `configs/mesh/`: Mesh recovery configs

Common parameters:
- `maxlen`: Maximum sequence length (typically 243 frames)
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `epochs`: Number of training epochs
- `dim_feat`/`dim_rep`: Feature dimensions (typically 512)

## Data Structure

The codebase expects data organized as:
```
data/
├── motion3d/MB3D_f243s81/
│   ├── AMASS/
│   └── H36M-SH/
├── motion2d/
│   ├── InstaVariety/
│   └── posetrack18_annotations/
└── mesh/
```

## Model Components

- **lib/model/DSTformer.py**: Core transformer architecture
- **lib/model/model_action.py**: Action recognition head
- **lib/model/model_mesh.py**: Mesh recovery head
- **lib/data/**: Dataset classes for different tasks and data types
- **lib/utils/**: Utility functions for data processing, visualization, and training

## Input/Output Format

- Input: 2D/3D pose sequences with shape `[batch_size, frames, joints(17), channels]`
- Uses H36M 17-joint format by default
- Output representations: `[batch_size, frames, joints, 512]` feature vectors
- Maximum supported sequence length: 243 frames