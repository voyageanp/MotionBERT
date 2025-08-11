#!/usr/bin/env python3
"""
MotionBERT Model Download Script
Downloads pretrained models from OneDrive links specified in README.md
"""

import os
import sys
import requests
from urllib.parse import urlparse, parse_qs
import zipfile
from pathlib import Path

# Model definitions from README.md
MODELS = {
    'pretrain': {
        'MotionBERT': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2',
            'config': 'configs/pretrain/MB_pretrain.yaml',
            'description': 'MotionBERT (162MB)'
        },
        'MotionBERT-Lite': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn',
            'config': 'configs/pretrain/MB_lite.yaml',
            'description': 'MotionBERT-Lite (61MB)'
        }
    },
    'pose3d': {
        'H36M-scratch': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgSvNejMQ0OHxMGZC?e=KcwBk1',
            'config': 'configs/pose3d/MB_train_h36m.yaml',
            'description': '3D Pose (H36M-SH, scratch) - 39.2mm MPJPE'
        },
        'H36M-ft': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgSoTqtyR5Zsgi8_Z?e=rn4VJf',
            'config': 'configs/pose3d/MB_ft_h36m.yaml', 
            'description': '3D Pose (H36M-SH, ft) - 37.2mm MPJPE'
        }
    },
    'action': {
        'NTU60-xsub': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgTX23yT_NO7RiZz-?e=nX6w2j',
            'config': 'configs/action/MB_ft_NTU60_xsub.yaml',
            'description': 'Action Recognition (x-sub, ft) - 97.2% Top1 Acc'
        },
        'NTU60-xview': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgTaNiXw2Nal-g37M?e=lSkE4T',
            'config': 'configs/action/MB_ft_NTU60_xview.yaml',
            'description': 'Action Recognition (x-view, ft) - 93.0% Top1 Acc'
        }
    },
    'mesh': {
        'PW3D-ft': {
            'url': 'https://1drv.ms/f/s!AvAdh0LSjEOlgTmgYNslCDWMNQi9?e=WjcB1F',
            'config': 'configs/mesh/MB_ft_pw3d.yaml',
            'description': 'Mesh (with 3DPW, ft) - 88.1mm MPVE'
        }
    }
}

def convert_onedrive_url(share_url):
    """Convert OneDrive share URL to direct download URL"""
    try:
        # Extract the file ID from OneDrive URL
        if 'onedrive.live.com' in share_url or '1drv.ms' in share_url:
            # For 1drv.ms URLs, we need to follow redirects first
            if '1drv.ms' in share_url:
                response = requests.head(share_url, allow_redirects=True)
                share_url = response.url
            
            # Extract resid from the URL
            if '/redir' in share_url:
                parsed = urlparse(share_url)
                query_params = parse_qs(parsed.query)
                if 'resid' in query_params:
                    resid = query_params['resid'][0]
                    # Convert to direct download URL
                    download_url = f"https://api.onedrive.com/v1.0/shares/s!{resid.split('!')[1]}/root/content"
                    return download_url
        
        return None
    except Exception as e:
        print(f"Error converting URL: {e}")
        return None

def download_file(url, filepath, description=""):
    """Download file from URL with progress bar"""
    print(f"Downloading {description}...")
    print(f"URL: {url}")
    print(f"Destination: {filepath}")
    
    try:
        # Convert OneDrive URL if needed
        download_url = convert_onedrive_url(url)
        if not download_url:
            print(f"Warning: Could not convert OneDrive URL. Trying original URL...")
            download_url = url
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✓ Successfully downloaded {description}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {description}: {e}")
        return False

def extract_if_zip(filepath):
    """Extract zip file if the downloaded file is a zip"""
    if zipfile.is_zipfile(filepath):
        print(f"Extracting {filepath}...")
        extract_dir = os.path.splitext(filepath)[0]
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✓ Extracted to {extract_dir}")
        return True
    return False

def main():
    """Main download function"""
    print("MotionBERT Model Downloader")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python download_models.py <category> [model_name]")
        print("\nAvailable categories and models:")
        for category, models in MODELS.items():
            print(f"\n{category}:")
            for model_name, model_info in models.items():
                print(f"  - {model_name}: {model_info['description']}")
        return
    
    category = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if category not in MODELS:
        print(f"Error: Unknown category '{category}'")
        print(f"Available categories: {list(MODELS.keys())}")
        return
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    
    models_to_download = {}
    if model_name:
        if model_name in MODELS[category]:
            models_to_download[model_name] = MODELS[category][model_name]
        else:
            print(f"Error: Unknown model '{model_name}' in category '{category}'")
            print(f"Available models: {list(MODELS[category].keys())}")
            return
    else:
        models_to_download = MODELS[category]
    
    # Download models
    for name, info in models_to_download.items():
        print(f"\n{'='*60}")
        category_dir = checkpoint_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Use model name as filename (will be zip file)
        filepath = category_dir / f"{name}.zip"
        
        success = download_file(info['url'], str(filepath), info['description'])
        
        if success:
            # Try to extract if it's a zip file
            extract_if_zip(str(filepath))
            
            print(f"Model config: {info['config']}")
        
        print(f"{'='*60}")
    
    print("\n✓ Download process completed!")
    print(f"Models saved to: {checkpoint_dir.absolute()}")

if __name__ == "__main__":
    main()