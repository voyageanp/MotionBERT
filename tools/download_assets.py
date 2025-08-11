import argparse
import hashlib
import os
import sys
import tarfile
import zipfile
from typing import Optional, Tuple

from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import yaml
from tqdm import tqdm


def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stream_download(url: str, dst_path: str, chunk_size: int = 1024 * 1024) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dst_path) or "download")
        with open(dst_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()


def is_archive(filename: str) -> Tuple[bool, Optional[str]]:
    lower = filename.lower()
    if lower.endswith('.zip'):
        return True, 'zip'
    if lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz') or lower.endswith('.tar.bz2'):
        return True, 'tar'
    return False, None


def extract_archive(archive_path: str, dest_dir: str) -> None:
    ensure_dir(dest_dir)
    lower = archive_path.lower()
    if lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz') or lower.endswith('.tar.bz2'):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")


def download_one(name: str, spec: dict, root: str, overwrite: bool = False, extract: Optional[bool] = None) -> None:
    url = spec['url']
    rel_path = spec.get('path') or os.path.basename(url)
    sha256 = spec.get('sha256')
    auto_extract = spec.get('extract', False)
    do_extract = extract if extract is not None else auto_extract

    dst_path = os.path.join(root, rel_path)
    dst_dir = dst_path if dst_path.endswith(os.sep) else os.path.dirname(dst_path)
    ensure_dir(dst_dir or '.')

    # If target is a directory and we plan to extract, download to a temp file in that directory
    dl_to = dst_path
    if do_extract:
        basename = os.path.basename(url)
        dl_to = os.path.join(dst_dir or '.', basename)

    if os.path.exists(dl_to) and not overwrite:
        print(f"[skip] {name}: exists at {dl_to}")
    else:
        print(f"[down] {name}: {url} -> {dl_to}") 
        try:
            stream_download(url, dl_to)
        except (HTTPError, URLError) as e:
            print(f"[err ] {name}: failed to download {url}: {e}") 
            sys.exit(1)

    if sha256 is not None:
        print(f"[hash] {name}: verifying sha256...") 
        got = sha256sum(dl_to)
        if got.lower() != sha256.lower():
            print(f"[err ] {name}: sha256 mismatch. expected={sha256} got={got}") 
            sys.exit(2)
        print(f"[ok  ] {name}: sha256 verified") 

    if do_extract:
        print(f"[xtr ] {name}: extracting {dl_to} -> {dst_dir}") 
        extract_archive(dl_to, dst_dir)
    elif dl_to != dst_path:
        # Move file to requested path if it differs
        ensure_dir(os.path.dirname(dst_path))
        os.replace(dl_to, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Download MotionBERT assets via a YAML manifest.")
    parser.add_argument("--manifest", required=True, help="Path to YAML manifest.")
    parser.add_argument("--dest", default=".", help="Destination root directory.")
    parser.add_argument("--select", nargs="*", help="Specific asset keys to download (default: all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--extract", action="store_true", help="Force extract archives (if applicable).")
    args = parser.parse_args()

    with open(args.manifest, 'r') as f:
        manifest = yaml.safe_load(f)

    assets = manifest.get('assets', {})
    if not assets:
        print("No assets found in manifest (expected key: 'assets').") 
        sys.exit(1)

    keys = args.select if args.select else list(assets.keys())
    missing = [k for k in keys if k not in assets]
    if missing:
        print(f"Unknown asset keys: {missing}") 
        sys.exit(1)

    for k in keys:
        download_one(k, assets[k], args.dest, overwrite=args.overwrite, extract=args.extract)

    print("Done.")


if __name__ == "__main__":
    main()
