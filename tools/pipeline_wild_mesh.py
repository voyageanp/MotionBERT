import argparse
import os
import shlex
import subprocess
import sys


def run(cmd: str, cwd: str = None) -> int:
    print(f"[cmd] {cmd}")
    proc = subprocess.Popen(shlex.split(cmd), cwd=cwd)
    return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline: MP4 -> AlphaPose -> JSON -> infer_wild_mesh")
    parser.add_argument("--video", required=True, help="Input video path (mp4)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--json", default=None, help="Existing AlphaPose JSON (skip detection)")
    parser.add_argument("--alphapose-cmd", default=None, help="Custom command template to run AlphaPose (use {video} and {outdir})")
    parser.add_argument("--alphapose-json", default="alphapose-results.json", help="Expected JSON filename produced by AlphaPose in outdir")
    parser.add_argument("--target-track", type=int, default=None, help="Track id to keep (if detection contains multiple)")
    parser.add_argument("--config", default="configs/mesh/MB_ft_pw3d.yaml", help="Mesh config YAML")
    parser.add_argument("--checkpoint", required=True, help="Mesh checkpoint bin file")
    parser.add_argument("--pixel", action="store_true", help="Align with pixel coordinates")
    parser.add_argument("--focus", type=int, default=None, help="Pass-through focus id to infer_wild_mesh")
    parser.add_argument("--clip-len", type=int, default=None, help="Override clip length for infer_wild_mesh")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) AlphaPose detection (optional if --json provided)
    det_json = args.json
    if det_json is None:
        if args.alphapose_cmd is None:
            print("AlphaPose command not provided. Use --alphapose-cmd or provide --json.")
            sys.exit(1)
        cmd = args.alphapose_cmd.format(video=args.video, outdir=args.outdir)
        code = run(cmd)
        if code != 0:
            sys.exit(code)
        det_json = os.path.join(args.outdir, args.alphapose_json)
        if not os.path.exists(det_json):
            print(f"AlphaPose JSON not found at {det_json}")
            sys.exit(1)

    # 2) Convert JSON to MotionBERT format
    mb_json = os.path.join(args.outdir, "alphapose_motionbert.json")
    conv_cmd = f"python tools/convert_alphapose_json.py --in-json {shlex.quote(det_json)} --out-json {shlex.quote(mb_json)} --expect-halpe26"
    if args.target_track is not None:
        conv_cmd += f" --target-track {args.target_track}"
    code = run(conv_cmd)
    if code != 0:
        sys.exit(code)

    # 3) Run infer_wild_mesh
    mesh_cmd = f"python infer_wild_mesh.py --config {shlex.quote(args.config)} -e {shlex.quote(args.checkpoint)} -j {shlex.quote(mb_json)} -v {shlex.quote(args.video)} -o {shlex.quote(args.outdir)}"
    if args.pixel:
        mesh_cmd += " --pixel"
    if args.focus is not None:
        mesh_cmd += f" --focus {args.focus}"
    if args.clip_len is not None:
        mesh_cmd += f" --clip_len {args.clip_len}"
    code = run(mesh_cmd)
    if code != 0:
        sys.exit(code)

    print("Done. See:", os.path.join(args.outdir, "mesh.mp4"))


if __name__ == "__main__":
    main()

