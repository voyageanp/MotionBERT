# 動画からテンソル変換プロセス詳解（日本語）

## 概要

動画ファイルからMotionBERTが処理可能な `[B, F, J, C]` 形式のテンソルへの変換プロセスを詳しく解説します。

## 全体フロー

```
動画ファイル (.mp4) 
    ↓ AlphaPose
2D姿勢検出結果 (JSON)
    ↓ キーポイント形式変換
H36M 17関節形式
    ↓ 正規化・クリッピング  
MotionBERT入力テンソル [B,F,J,C]
```

## 各ステップの詳細

### Step 1: 2D姿勢検出 - AlphaPose

#### 使用モデル
```bash
# AlphaPose Fast Pose モデル (Halpe dataset trained)
# 出力: 26キーポイント × 3チャンネル [x, y, confidence]
```

#### AlphaPoseの実行
```bash
# AlphaPoseのインストールと実行例
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose
python scripts/demo_inference.py \
    --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
    --indir path/to/video.mp4 \
    --outdir output_directory/
```

#### 出力形式
```json
[
  {
    "image_id": "frame_000001.jpg",
    "category_id": 1, 
    "keypoints": [x1, y1, c1, x2, y2, c2, ..., x26, y26, c26],
    "score": 0.95,
    "idx": 0
  },
  ...
]
```

**Halpe 26キーポイント構成:**
```python
# lib/data/dataset_wild.py:18-44
halpe_keypoints = {
    0: "Nose",      1: "LEye",      2: "REye",      3: "LEar",      4: "REar",
    5: "LShoulder", 6: "RShoulder", 7: "LElbow",    8: "RElbow",    9: "LWrist",
    10: "RWrist",   11: "LHip",     12: "RHip",     13: "LKnee",    14: "RKnee", 
    15: "LAnkle",   16: "RAnkle",   17: "Head",     18: "Neck",     19: "Hip",
    20: "LBigToe",  21: "RBigToe",  22: "LSmallToe", 23: "RSmallToe",
    24: "LHeel",    25: "RHeel"
}
```

### Step 2: JSON読み込みとパース

#### 実装箇所: `lib/data/dataset_wild.py:67-86`

```python
def read_input(json_path, vid_size, scale_range, focus):
    # JSONファイル読み込み
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    
    kpts_all = []
    for item in results:
        # 特定人物のフォーカス
        if focus != None and item['idx'] != focus:
            continue
            
        # キーポイント抽出・リシェイプ
        kpts = np.array(item['keypoints']).reshape([-1, 3])  # [26, 3]
        kpts_all.append(kpts)
    
    # 全フレーム統合
    kpts_all = np.array(kpts_all)  # [T, 26, 3]
    return kpts_all
```

**次元変化:**
```
JSON → [T, 26, 3]
T = 動画の総フレーム数
26 = Halpeキーポイント数  
3 = [x, y, confidence]
```

### Step 3: キーポイント形式変換 (Halpe → H36M)

#### 実装箇所: `lib/data/dataset_wild.py:15-65`

```python
def halpe2h36m(x):
    """
    Input:  x [T, 26, 3] (Halpe 26 keypoints)  
    Output: y [T, 17, 3] (H36M 17 keypoints)
    """
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    
    # 変換マッピング (Halpe → H36M)
    y[:,0,:] = x[:,19,:]   # Hip (中心)
    y[:,1,:] = x[:,12,:]   # RHip  
    y[:,2,:] = x[:,14,:]   # RKnee
    y[:,3,:] = x[:,16,:]   # RAnkle
    y[:,4,:] = x[:,11,:]   # LHip
    y[:,5,:] = x[:,13,:]   # LKnee  
    y[:,6,:] = x[:,15,:]   # LAnkle
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5  # Spine (平均値)
    y[:,8,:] = x[:,18,:]   # Thorax
    y[:,9,:] = x[:,0,:]    # Head/Nose
    y[:,10,:] = x[:,17,:]  # Head top
    y[:,11,:] = x[:,5,:]   # LShoulder
    y[:,12,:] = x[:,7,:]   # LElbow
    y[:,13,:] = x[:,9,:]   # LWrist
    y[:,14,:] = x[:,6,:]   # RShoulder
    y[:,15,:] = x[:,8,:]   # RElbow  
    y[:,16,:] = x[:,10,:]  # RWrist
    
    return y
```

**H36M 17関節構成:**
```python
h36m_keypoints = {
    0: "Hip",        1: "RHip",       2: "RKnee",      3: "RAnkle",
    4: "LHip",       5: "LKnee",      6: "LAnkle",     7: "Spine",
    8: "Thorax",     9: "Nose",       10: "Head",      11: "LShoulder", 
    12: "LElbow",    13: "LWrist",    14: "RShoulder", 15: "RElbow",
    16: "RWrist"
}
```

**次元変化:**
```
[T, 26, 3] → [T, 17, 3]
不要な関節を削除、一部は計算で生成（例：Spine = (Neck + Hip) / 2）
```

### Step 4: 座標正規化

#### 実装箇所: `lib/data/dataset_wild.py:78-85`

```python
def normalize_coordinates(kpts_all, vid_size, scale_range):
    if vid_size:
        w, h = vid_size  # 動画の幅・高さ
        scale = min(w, h) / 2.0
        
        # 画像中心を原点に移動
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        
        # [-1, 1] 範囲に正規化
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        
    if scale_range:
        # さらなるスケール正規化
        kpts_all = crop_scale(kpts_all, scale_range)
        
    return kpts_all.astype(np.float32)
```

**正規化の効果:**
```python
# 正規化前（ピクセル座標）
x_pixel = [640, 360]  # 1280x720動画での例

# 正規化後
x_normalized = [(640 - 640) / 360, (360 - 360) / 360] = [0.0, 0.0]

# 一般的な変換式
x_norm = (x_pixel - w/2) / (min(w,h)/2)  
y_norm = (y_pixel - h/2) / (min(w,h)/2)
```

### Step 5: 時系列クリッピング

#### 実装箇所: `lib/data/dataset_wild.py:88-101`

```python
class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.clip_len = clip_len  # 通常243フレーム
        self.vid_all = read_input(json_path, vid_size, scale_range, focus)
        
    def __len__(self):
        # クリップ数を計算
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        # 指定クリップを抽出  
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]  # [clip_len, 17, 3]
```

**クリッピング例:**
```python
# 500フレームの動画で clip_len=243 の場合
clip_0: frames [0:243]    # 243フレーム
clip_1: frames [243:486]  # 243フレーム  
clip_2: frames [486:500]  # 14フレーム（最後のクリップ）

# 総クリップ数: ceil(500/243) = 3
```

### Step 6: 最終テンソル形成

#### DataLoaderでの統合
```python
# infer_wild.py:41-63
testloader_params = {
    'batch_size': 1,      # 通常1（動画推論）
    'shuffle': False,     # 時系列順を保持
    'num_workers': 8,
    'pin_memory': True
}

test_loader = DataLoader(wild_dataset, **testloader_params)

# バッチ処理
for batch_input in test_loader:
    N, T = batch_input.shape[:2]  # [1, clip_len]
    # batch_input.shape = [1, clip_len, 17, 3]
```

## 最終的なテンソル形状

```python
# MotionBERTへの入力テンソル
tensor_shape = [B, F, J, C]

B = 1        # バッチサイズ（動画推論では通常1）
F = 243      # フレーム数（クリップ長）
J = 17       # 関節数（H36M形式）  
C = 3        # チャンネル数 [x, y, confidence] または [x, y, z]

# 実例
input_tensor = torch.tensor([1, 243, 17, 3])  # float32
```

## チャンネル情報の詳細

```python
# C = 3 の内容
channel_0 = x_coordinate    # x座標（正規化済み、[-1,1]範囲）
channel_1 = y_coordinate    # y座標（正規化済み、[-1,1]範囲）  
channel_2 = confidence      # 信頼度（[0,1]範囲、AlphaPose出力）

# 推論時の注意点
if args.no_conf:
    # 信頼度チャンネルを除外する場合
    batch_input = batch_input[:, :, :, :2]  # [B, F, J, 2]
```

## エラー処理と注意点

### 1. キーポイント欠損への対応
```python
# dataset_wild.py で confidence=0 のキーポイントを適切に処理
missing_keypoint = [0.0, 0.0, 0.0]  # x, y, conf すべて0
```

### 2. 複数人物の処理
```python  
# focus パラメータで特定人物を指定
wild_dataset = WildDetDataset(json_path, focus=0)  # ID=0の人物のみ
```

### 3. 動画長による処理
```python
# 短い動画: パディング処理が必要な場合がある
# 長い動画: 複数クリップに自動分割
```

この一連の処理により、任意の動画ファイルがMotionBERTで処理可能な統一形式のテンソルに変換されます。