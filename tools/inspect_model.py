import argparse
import os
from typing import Tuple

import torch

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Inspect MotionBERT model architecture")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--task", choices=["pose3d", "action", "mesh"], default="pose3d", help="Build the task-specific model wrapper")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for a forward pass")
    parser.add_argument("--frames", type=int, default=None, help="Override frames (defaults to config.clip_len)")
    parser.add_argument("--joints", type=int, default=None, help="Override joints (defaults to config.num_joints)")
    parser.add_argument("--channels", type=int, default=3, help="Input channels (2 or 3)")
    parser.add_argument("--cuda", action="store_true", help="Move model and dummy input to CUDA if available")
    parser.add_argument("--graph", action="store_true", help="Export graph to TensorBoard logdir")
    parser.add_argument("--logdir", type=str, default="logs/model_graph", help="TensorBoard log directory root")
    args = parser.parse_args()

    cfg = get_config(args.config)

    # Build model from config
    # Build model (backbone or task-specific wrapper)
    task = args.task
    if task == "pose3d":
        model = load_backbone(cfg)
    elif task == "action":
        from lib.model.model_action import ActionNet
        backbone = load_backbone(cfg)
        num_classes = getattr(cfg, "action_classes", 60)
        dropout_ratio = getattr(cfg, "dropout_ratio", 0.5)
        version = getattr(cfg, "model_version", "class")
        hidden_dim = getattr(cfg, "hidden_dim", 2048)
        num_joints = getattr(cfg, "num_joints", 17)
        model = ActionNet(backbone, dim_rep=getattr(cfg, "dim_rep", 512), num_classes=num_classes,
                          dropout_ratio=dropout_ratio, version=version, hidden_dim=hidden_dim, num_joints=num_joints)
    elif task == "mesh":
        from lib.model.model_mesh import MeshRegressor
        backbone = load_backbone(cfg)
        model = MeshRegressor(cfg, backbone, dim_rep=getattr(cfg, "dim_rep", 512),
                              num_joints=getattr(cfg, "num_joints", 17), hidden_dim=getattr(cfg, "hidden_dim", 1024),
                              dropout_ratio=getattr(cfg, "dropout", 0.5))
    model.eval()

    trainable, total = count_parameters(model)
    print("Model class:", model.__class__.__name__)
    print(f"Params: trainable={trainable:,} total={total:,}")
    print("\nModule tree:\n")
    print(model)

    # Prepare dummy input
    B = args.batch_size
    F = args.frames if args.frames is not None else getattr(cfg, "clip_len", getattr(cfg, "maxlen", 243))
    J = args.joints if args.joints is not None else getattr(cfg, "num_joints", 17)
    C = args.channels
    if task == "action":
        x = torch.randn(B, 1, F, J, C)
    else:
        x = torch.randn(B, F, J, C)

    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()

    # Dry-run forward to reveal output shape
    with torch.no_grad():
        y = model(x)
    print("\nDummy input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    # Optional: export graph to TensorBoard
    if args.graph:
        try:
            from tensorboardX import SummaryWriter
            tag = os.path.splitext(os.path.basename(args.config))[0]
            logdir = os.path.join(args.logdir, tag)
            os.makedirs(logdir, exist_ok=True)
            writer = SummaryWriter(logdir)
            writer.add_graph(model, (x,))
            writer.close()
            print(f"\nGraph written to: {logdir}")
            print("Run: tensorboard --logdir", args.logdir)
        except Exception as e:
            print("Failed to export graph:", e)


if __name__ == "__main__":
    main()
