#!/usr/bin/env python3
"""
動画からAlphaPoseで2D姿勢推定を行い、JSON結果を出力するスクリプト
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import cv2

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# AlphaPose libraries
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.writer import DataWriter

def setup_args():
    parser = argparse.ArgumentParser(description='Video to AlphaPose 2D Pose Estimation')
    
    # 入出力設定
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--outdir', default='output', help='Output directory')
    parser.add_argument('--outname', default='alphapose-results', help='Output filename (without extension)')
    
    # モデル設定
    parser.add_argument('--detector', default='yolo', choices=['yolo', 'yolox', 'efficientdet'],
                        help='Human detector')
    parser.add_argument('--pose_model', default='pretrained_models/halpe26_fast_res50_256x192.pth',
                        help='2D pose estimation model path')
    parser.add_argument('--config', default='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='AlphaPose config file')
    
    # バッチサイズ設定
    parser.add_argument('--detbatch', type=int, default=5, help='Detection batch size')
    parser.add_argument('--posebatch', type=int, default=30, help='Pose estimation batch size')
    
    # その他の設定
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save_img', action='store_true', help='Save visualization images')
    parser.add_argument('--vis', action='store_true', help='Show visualization')
    parser.add_argument('--tracking', action='store_true', help='Enable person tracking')
    parser.add_argument('--min_box_area', type=int, default=0, help='Minimum bounding box area')
    
    return parser.parse_args()

def check_video(video_path):
    """動画ファイルの情報を確認"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    print(f"Video info:")
    print(f"  Path: {video_path}")
    print(f"  Frames: {frame_count}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.2f}s")
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }

def video_to_alphapose(args):
    """動画からAlphaPose推論を実行"""
    
    # 動画情報確認
    video_info = check_video(args.video)
    
    # 出力ディレクトリ作成
    os.makedirs(args.outdir, exist_ok=True)
    
    # 設定読み込み
    print(f"Loading config: {args.config}")
    cfg = update_config(args.config)
    
    # デバイス設定
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    
    # DetectionLoader初期化（動画モード）
    print("Initializing detection loader...")
    det_loader = DetectionLoader(
        args.video,  # 動画パスを直接指定
        get_detector(args), 
        cfg, 
        args, 
        mode='video',  # 動画モード
        batchSize=args.detbatch, 
        queueSize=1024
    )
    det_loader.start()
    
    # 2Dポーズ推定モデル読み込み
    print("Loading 2D pose estimation model...")
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    
    if not os.path.exists(args.pose_model):
        raise FileNotFoundError(f"Pose model not found: {args.pose_model}")
    
    print(f'Loading pose model from {args.pose_model}...')
    pose_model.load_state_dict(torch.load(args.pose_model, map_location=args.device))
    pose_model.to(args.device)
    pose_model.eval()
    
    # DataWriter初期化（結果保存用）
    print("Initializing result writer...")
    writer = DataWriter(cfg, args, save_video=False, queueSize=1024).start()
    
    # 推論実行
    print("Starting pose estimation...")
    frame_count = 0
    start_time = time.time()
    
    try:
        with torch.no_grad():
            while True:
                # フレーム読み込み
                detection_result = det_loader.read()
                if detection_result is None:
                    break
                    
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = detection_result
                
                if orig_img is None:
                    print("End of video reached")
                    break
                
                frame_count += 1
                
                # 人体検出結果確認
                if boxes is None or boxes.nelement() == 0:
                    print(f"Frame {frame_count}: No humans detected")
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                
                print(f"Frame {frame_count}: Detected {boxes.size(0)} person(s)")
                
                # 2Dポーズ推定
                inps = inps.to(args.device)
                datalen = inps.size(0)
                
                # バッチ処理
                leftover = 0
                if datalen % args.posebatch:
                    leftover = 1
                num_batches = datalen // args.posebatch + leftover
                
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * args.posebatch:min((j + 1) * args.posebatch, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                
                hm = torch.cat(hm)
                hm = hm.cpu()
                
                # 結果保存
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                
                # 進捗表示
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    progress = frame_count / video_info['frame_count'] * 100
                    print(f"Progress: {frame_count}/{video_info['frame_count']} frames "
                          f"({progress:.1f}%) - {fps:.2f} FPS")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # 結果保存完了まで待機
    print("Waiting for result saving to complete...")
    while writer.running():
        time.sleep(1)
        remaining = writer.count()
        if remaining > 0:
            print(f'Saving remaining {remaining} results...', end='\r')
    
    # クリーンアップ
    writer.stop()
    det_loader.stop()
    
    # 統計情報
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n=== Processing Complete ===")
    print(f"Processed frames: {frame_count}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # 結果ファイル確認
    json_path = os.path.join(args.outdir, f"{args.outname}.json")
    if os.path.exists(json_path):
        print(f"Results saved to: {json_path}")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(json_path) / 1024  # KB
        print(f"Result file size: {file_size:.1f} KB")
        
        return json_path
    else:
        print("Warning: Result file not found")
        return None

def main():
    args = setup_args()
    
    try:
        result_path = video_to_alphapose(args)
        if result_path:
            print(f"\nSuccess! AlphaPose results saved to: {result_path}")
            print("\nNext steps:")
            print("1. Use convert_alphapose_to_motionbert.py to prepare MotionBert input")
            print("2. Run 3D pose estimation with MotionBert")
        else:
            print("Failed to generate results")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())