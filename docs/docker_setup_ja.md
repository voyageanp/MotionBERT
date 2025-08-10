# MotionBERT Docker セットアップ（日本語）

## 概要

MotionBERTをDockerコンテナで実行するためのセットアップ手順を説明します。

## 前提条件

- Docker
- Docker Compose  
- NVIDIA Docker Runtime（GPU使用の場合）

### NVIDIA Docker Runtime のインストール

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 基本的な使い方

### 1. コンテナのビルドと起動

```bash
# リポジトリのクローン（まだの場合）
git clone https://github.com/Walter0807/MotionBERT.git
cd MotionBERT

# Dockerコンテナのビルド
docker-compose build motionbert

# コンテナの起動（バックグラウンド）
docker-compose up -d motionbert

# コンテナに接続
docker-compose exec motionbert bash
```

### 2. 直接的なDocker実行

```bash
# イメージのビルド
docker build -t motionbert .

# コンテナの実行（GPU使用）
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoint:/workspace/checkpoint \
  -p 8888:8888 \
  motionbert
```

## 開発環境での使用

### Jupyter Lab の起動

```bash
# Jupyter専用コンテナの起動
docker-compose up -d jupyter

# ブラウザでアクセス
# http://localhost:8889
```

### TensorBoard の起動

```bash
# コンテナ内で実行
tensorboard --logdir=checkpoint/logs --host=0.0.0.0 --port=6006

# ブラウザでアクセス
# http://localhost:6006
```

## データとモデルの準備

### 1. データディレクトリの作成

```bash
# ホスト側で実行
mkdir -p data/motion3d
mkdir -p data/motion2d
mkdir -p checkpoint
mkdir -p output
```

### 2. 事前学習済みモデルのダウンロード

```bash
# コンテナ内で実行
cd /workspace
wget https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2
# 適切なディレクトリに解凍
```

## 基本的な実行例

### 1. 動画からの3Dポーズ推定

```bash
# コンテナ内で実行
python infer_wild.py \
  --vid_path data/input_video.mp4 \
  --json_path data/alphapose_results.json \
  --out_path output/pose_results/
```

### 2. 学習の実行

```bash
# 事前学習
python train.py \
  --config configs/pretrain/MB_pretrain.yaml \
  -c checkpoint/pretrain/MB_pretrain

# ファインチューニング
python train.py \
  --config configs/pose3d/MB_ft_h36m.yaml \
  -c checkpoint/pose3d/MB_ft_h36m
```

## トラブルシューティング

### GPU認識の確認

```bash
# コンテナ内で実行
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### メモリ不足の対処

```bash
# より小さなバッチサイズで実行
python train.py --config configs/pretrain/MB_pretrain.yaml --batch_size 16
```

### 権限エラーの対処

```bash
# ホスト側で実行
sudo chown -R $USER:$USER data/ checkpoint/ output/
```

## カスタマイズ

### Dockerfileの修正

```dockerfile
# 追加のパッケージをインストールしたい場合
RUN pip install your_package_name

# システムパッケージを追加したい場合  
RUN apt-get update && apt-get install -y your_system_package
```

### Docker Composeの設定変更

```yaml
# docker-compose.yml
services:
  motionbert:
    # メモリ制限の設定
    mem_limit: 16g
    
    # GPU指定
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # 複数GPU使用
```

## ボリューム構成

```
ホスト側                    コンテナ側
./                     →   /workspace/          # ソースコード
./data/               →   /workspace/data/      # データセット
./checkpoint/         →   /workspace/checkpoint/ # モデル
./output/            →   /workspace/output/     # 結果
```

## 注意事項

- GPU使用にはNVIDIA Docker Runtimeが必要
- 大きなデータセットは`.dockerignore`で除外推奨
- 開発時はソースコードをボリュームマウントで共有
- 本番環境では適切なセキュリティ設定を適用

## よく使うDockerコマンド

```bash
# コンテナの状態確認
docker-compose ps

# ログの確認
docker-compose logs motionbert

# コンテナの停止
docker-compose down

# イメージの再ビルド
docker-compose build --no-cache motionbert

# コンテナ内でのシェル実行
docker-compose exec motionbert bash
```