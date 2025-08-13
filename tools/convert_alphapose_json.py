import argparse
import json
import os
import re
from typing import Any, Dict, List


def extract_frame_order(v: Dict[str, Any]) -> int:
    # Try common keys to order frames deterministically
    for key in ("frame_id", "frame", "image_id", "image_path", "imgname", "filename"):
        if key in v:
            val = v[key]
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                # extract last integer in string as frame index
                nums = re.findall(r"(\d+)", val)
                if nums:
                    return int(nums[-1])
    return 0


def main():
    parser = argparse.ArgumentParser(description="Convert AlphaPose JSON to MotionBERT wild JSON format.")
    parser.add_argument("--in-json", required=True, help="AlphaPose output JSON (list of dicts)")
    parser.add_argument("--out-json", required=True, help="Output JSON path for MotionBERT")
    parser.add_argument("--track-key", default=None, help="Key name for track id (e.g., idx, track_id, person_id)")
    parser.add_argument("--target-track", dest="target_track", type=int, default=None, help="If set, filter to this track id")
    parser.add_argument("--expect-halpe26", action="store_true", help="Validate keypoints length is 26*3")
    args = parser.parse_args()

    with open(args.in_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of detections")

    out: List[Dict[str, Any]] = []
    for det in data:
        kpts = det.get("keypoints") or det.get("key_points") or det.get("pose_keypoints_2d")
        if kpts is None:
            continue
        if isinstance(kpts, list):
            flat = kpts
        else:
            # some formats store as dict {x:[], y:[], score:[]}
            xs = det.get("x") or det.get("X")
            ys = det.get("y") or det.get("Y")
            cs = det.get("score") or det.get("conf") or det.get("c")
            if xs is None or ys is None:
                continue
            if cs is None:
                cs = [1.0] * len(xs)
            flat = []
            for x, y, c in zip(xs, ys, cs):
                flat.extend([x, y, c])

        if args.expect_halpe26 and len(flat) != 26 * 3:
            raise ValueError(f"Expected 26*3=78 values, got {len(flat)}")

        # Determine track id
        track_key = args.track_key
        track_id = None
        if track_key is not None:
            track_id = det.get(track_key)
        else:
            for cand in ("idx", "track_id", "person_id", "id"):
                if cand in det:
                    track_id = det[cand]
                    break
        if track_id is None:
            track_id = 0

        if args.target_track is not None and track_id != args.target_track:
            continue

        out.append({
            "idx": int(track_id),
            "keypoints": flat,
            "_frame": extract_frame_order(det)
        })

    # Sort by frame order to ensure temporal alignment
    out.sort(key=lambda d: d.get("_frame", 0))
    for d in out:
        d.pop("_frame", None)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} detections to {args.out_json}")


if __name__ == "__main__":
    main()
