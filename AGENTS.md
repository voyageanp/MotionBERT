# Repository Guidelines

## Project Structure & Module Organization
- lib/: Core Python code
  - lib/model/: Architectures (e.g., `DSTformer.py`, heads)
  - lib/data/: Dataloaders and augmentations
  - lib/utils/: Training, visualization, SMPL utilities
- configs/: YAML configs for tasks (`pose3d/`, `mesh/`, `action/`, `pretrain/`)
- tools/: Data preprocessing scripts (e.g., `convert_h36m.py`)
- docs/: Task guides (pose3d, action, mesh, pretrain, inference)
- train*.py, infer_wild*.py: Entry scripts per task
- params/: Auxiliary assets (e.g., noise, regressor params)
- checkpoint/, data/, logs/: Local outputs (git‑ignored)

## Build, Test, and Development Commands
- Create env: `conda create -n motionbert python=3.7 anaconda && conda activate motionbert`
- Install PyTorch (per CUDA): see PyTorch site, then `pip install -r requirements.txt`
- Pretrain: `python train.py --config configs/pretrain/MB_pretrain.yaml -c checkpoint/pretrain/MB_pretrain`
- Pose3D: `python train.py --config configs/pose3d/MB_train_h36m.yaml --checkpoint checkpoint/pose3d/MB_train_h36m`
- Action: `python train_action.py --config configs/action/MB_train_NTU60_xsub.yaml --checkpoint checkpoint/action/MB_train_NTU60_xsub`
- Mesh: `python train_mesh.py --config configs/mesh/MB_train_pw3d.yaml --checkpoint checkpoint/mesh/MB_train_pw3d`
- Evaluate: add `--evaluate <path/to/best_epoch.bin>` to the corresponding command

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8 where practical
- snake_case for files/functions; CamelCase for classes
- Configs: YAML with clear, task‑scoped names (e.g., `MB_ft_*.yaml`)
- Keep modules focused; place shared helpers in `lib/utils/`

## Testing Guidelines
- No formal unit test suite in repo; validate changes via task scripts on a small subset
- Smoke test example: run a short Pose3D train/eval with H36M samples and confirm loss/MPJPE trend
- If adding tests, use `pytest` under `tests/` and avoid heavy data; mock I/O where possible

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., "Fix data slicing stride")
- PRs must include: purpose, affected areas, config(s) and dataset(s) used, repro commands, and notes on compatibility
- Link related issues; add before/after metrics or logs when changing training/inference
- Do not commit data/checkpoints/logs; paths under `data/` and `checkpoint/` are git‑ignored

## Security & Configuration Tips
- Keep credentials out of configs; store datasets under `data/`
- Match CUDA/PyTorch versions to your environment; document any driver constraints in PRs
