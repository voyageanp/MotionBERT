# MotionBERT セットアップと動かし方（日本語）

## 環境構築

### 1. 仮想環境の作成と設定

```bash
# 仮想環境作成
conda create -n motionbert python=3.7 anaconda
conda activate motionbert

# PyTorch インストール (CUDA 11.6の場合)
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 事前学習済みモデルのダウンロード

[OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2)から必要なモデルをダウンロードして、適切なディレクトリに配置してください。

## 基本的な使い方

### 1. 動画からの推論（最も簡単）

#### 3Dポーズ推定
```bash
# 事前に AlphaPose で 2D姿勢検出が必要
python infer_wild.py \
--vid_path video.mp4 \
--json_path alphapose_results.json \
--out_path output/
```

#### メッシュ復元
```bash
python infer_wild_mesh.py \
--vid_path video.mp4 \
--json_path alphapose_results.json \
--out_path output/
```

### 2. 学習

#### 事前学習
```bash
python train.py \
--config configs/pretrain/MB_pretrain.yaml \
-c checkpoint/pretrain/MB_pretrain
```

#### 3Dポーズ推定のファインチューニング
```bash
python train.py \
--config configs/pose3d/MB_ft_h36m.yaml \
-c checkpoint/pose3d/MB_ft_h36m
```

#### 行動認識
```bash
python train_action.py \
--config configs/action/MB_ft_NTU60_xsub.yaml \
-c checkpoint/action/MB_ft_NTU60_xsub
```

#### メッシュ復元
```bash
python train_mesh.py \
--config configs/mesh/MB_ft_pw3d.yaml \
-c checkpoint/mesh/MB_ft_pw3d
```

## よく使うコマンドとオプション

### 学習関連
```bash
# 学習の再開
python train.py --config config.yaml -c checkpoint_dir -r checkpoint.bin

# 評価のみ実行
python train.py --config config.yaml -e checkpoint.bin

# 特定のエポックから再開
python train.py --config config.yaml -c checkpoint_dir -r latest_epoch.bin
```

### 推論関連
```bash
# 特定の人物にフォーカス（複数人が映っている場合）
python infer_wild.py --vid_path video.mp4 --json_path results.json --focus 0

# ピクセル座標系で出力
python infer_wild.py --vid_path video.mp4 --json_path results.json --pixel

# クリップ長を変更（デフォルト243フレーム）
python infer_wild.py --vid_path video.mp4 --json_path results.json --clip_len 81
```

## データセットの準備

各タスク用のデータセット準備については、以下のドキュメントを参照：

- 事前学習: `docs/pretrain.md`
- 3Dポーズ推定: `docs/pose3d.md`
- 行動認識: `docs/action.md`
- メッシュ復元: `docs/mesh.md`

## ファイル構成

```
MotionBERT/
├── configs/           # 設定ファイル（YAML形式）
│   ├── pretrain/      # 事前学習用設定
│   ├── pose3d/        # 3Dポーズ推定用設定
│   ├── action/        # 行動認識用設定
│   └── mesh/          # メッシュ復元用設定
├── lib/               # コアライブラリ
│   ├── model/         # モデル定義
│   ├── data/          # データセット・データローダー
│   └── utils/         # ユーティリティ関数
├── tools/             # データ前処理スクリプト
├── train.py           # メイン学習スクリプト
├── infer_wild.py      # 動画推論スクリプト（3Dポーズ）
└── infer_wild_mesh.py # 動画推論スクリプト（メッシュ）
```

## トラブルシューティング

### よくあるエラー

1. **CUDA out of memory**: `batch_size` を小さくしてください
2. **キーポイント形式エラー**: AlphaPoseでHalpe形式（26キーポイント）を使用してください
3. **動画読み込みエラー**: `imageio-ffmpeg` がインストールされていることを確認してください

### 推奨設定

- GPU: 8GB以上のVRAM推奨
- メモリ: 16GB以上推奨
- Python: 3.7（テスト済み）