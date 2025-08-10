# MotionBERT モデル構造詳細解説（日本語）

## 概要

MotionBERTは、人体動作の統一的な表現学習を目的としたTransformerベースのモデルです。DSTformer（Dual-Scale Transformer）という独自のアーキテクチャを採用し、空間的・時間的な情報を効率的に処理します。

## 全体アーキテクチャ

```
Input [B,F,J,C] → DSTformer (Backbone) → Task-specific Head → Output
```

## 入力から出力までの次元変化

### 1. 入力 (Input)

```python
Shape: [B, F, J, C]
B = Batch size（バッチサイズ）
F = Frames（時間長、最大243フレーム）  
J = Joints（17関節、H36M形式）
C = Channels（通常3、x,y,z座標またはx,y,confidence）

例: [64, 243, 17, 3]
```

### 2. DSTformerでの処理

#### Step 1: Joint Embedding（関節埋め込み）

```python
# lib/model/DSTformer.py:333
x = self.joints_embed(x)  # Linear(3, 512)

次元変化: [B×F, J, C] → [B×F, J, 512]
例: [64×243, 17, 3] → [64×243, 17, 512]
```

#### Step 2: Positional Encoding（位置エンコーディング）

```python
# 空間的位置エンコーディング（関節位置）
x = x + self.pos_embed  # [1, 17, 512]

# 時間的位置エンコーディング（時間位置）  
x = x + self.temp_embed[:,:F,:,:]  # [1, 243, 1, 512]

結果: [B×F, J, 512] (位置情報が追加)
```

#### Step 3: Dual-Scale Transformer Blocks

```python
# 5つのTransformerブロック (depth=5)
for blk_st, blk_ts in zip(self.blocks_st, self.blocks_ts):
    x_st = blk_st(x, F)    # Spatial-Temporal順
    x_ts = blk_ts(x, F)    # Temporal-Spatial順  
    x = fuse(x_st, x_ts)   # アテンション重み付き融合

維持される次元: [B×F, J, 512]
```

**各ブロック内の処理:**

- **Spatial Attention**: 関節間の関係をモデル化
  - Query, Key, Value: `[B×F, num_heads, J, 512/num_heads]`
  - Attention Map: `[B×F, num_heads, J, J]`

- **Temporal Attention**: 時間軸の関係をモデル化  
  - データ再構成: `[B, num_heads, J, F, 512/num_heads]`
  - Attention Map: `[B, num_heads, J, F, F]`

#### Step 4: 最終表現生成

```python
x = self.pre_logits(x)  # Linear(512, 512) + Tanh()

最終出力: [B, F, J, 512]
例: [64, 243, 17, 512]
```

### 3. タスク別ヘッドでの次元変化

#### A. 3D Pose Estimation（3Dポーズ推定）

```python
# DSTformerから直接出力
x = self.head(x)  # Linear(512, 3)

次元変化: [B, F, J, 512] → [B, F, J, 3]
例: [64, 243, 17, 512] → [64, 243, 17, 3]
```

#### B. Action Recognition（行動認識）

```python
# model_action.py:15-28
def forward(self, feat):
    N, M, T, J, C = feat.shape  # [64, 1, 243, 17, 512]
    
    # 時間軸で平均化
    feat = feat.permute(0, 1, 3, 4, 2)  # (N, M, J, C, T)
    feat = feat.mean(dim=-1)            # (N, M, J, C)
    
    # 関節次元を展開
    feat = feat.reshape(N, M, -1)       # (N, M, J×C)
    
    # クリップ軸で平均化  
    feat = feat.mean(dim=1)             # (N, J×C)
    
    # 分類ヘッド
    feat = self.fc1(feat)               # Linear(17×512, 2048)
    feat = self.fc2(feat)               # Linear(2048, num_classes)

最終次元変化:
[N, M, T, J, C] → [N, M, J×C] → [N, J×C] → [N, num_classes]
例: [64, 1, 243, 17, 512] → [64, 8704] → [64, 60]
```

#### C. Mesh Recovery（メッシュ復元）

```python
# model_mesh.py:37-80
def forward(self, feat):
    N, T, J, C = feat.shape  # [64, 243, 17, 512]
    NT = N * T
    
    # Pose branch（ポーズ分岐）
    feat_pose = feat.reshape(NT, -1)     # (N×T, J×C)
    pred_pose = self.head_pose(feat_pose) # Linear(8704, 144) # 24joints×6D
    
    # Shape branch（体型分岐）
    feat_shape = feat.mean(dim=1)        # 時間軸平均 (N, J×C)
    pred_shape = self.head_shape(feat_shape) # Linear(8704, 10)
    
    # SMPL モデルで頂点生成
    pred_vertices = SMPL(pose=pred_pose, shape=pred_shape)

最終次元変化:
[N, T, J, C] → Pose: [N×T, 144], Shape: [N, 10]
→ SMPL → Vertices: [N×T, 6890, 3]
```

## 重要な設計特徴

### 1. Dual-Scale Attention

```python
# 2つの処理順序を並列実行
x_st = SpatialAttention → TemporalAttention  # 空間→時間
x_ts = TemporalAttention → SpatialAttention  # 時間→空間

# アテンション重みで融合
alpha = AttentionWeights(x_st, x_ts)  # [B×F, J, 512, 2]
x = x_st * alpha[:,:,:,0] + x_ts * alpha[:,:,:,1]
```

### 2. Multi-Head Attention詳細

#### Spatial Attention
```python
# 関節間の関係モデリング
Q, K, V = [B×F, num_heads, J, head_dim]
attn = (Q @ K.T) * scale  # [B×F, num_heads, J, J]
out = attn @ V            # [B×F, num_heads, J, head_dim]
```

#### Temporal Attention  
```python
# 時間軸の関係モデリング
Q, K, V = [B, num_heads, J, F, head_dim]
attn = (Q @ K.T) * scale  # [B, num_heads, J, F, F]  
out = attn @ V            # [B, num_heads, J, F, head_dim]
```

### 3. パラメータ数

- **DSTformer Backbone**: 約162MB
- **DSTformer-Lite**: 約61MB（軽量版）
- **主要パラメータ**:
  - `dim_feat`: 512（特徴次元）
  - `depth`: 5（Transformerレイヤ数）
  - `num_heads`: 8（マルチヘッド数）
  - `maxlen`: 243（最大シーケンス長）

## 設定可能パラメータ

### コアパラメータ（configs/*.yaml）

```yaml
# モデル構造
dim_feat: 512        # 特徴次元
dim_rep: 512         # 表現次元  
depth: 5             # Transformerレイヤ数
num_heads: 8         # アテンションヘッド数
maxlen: 243          # 最大シーケンス長
mlp_ratio: 2         # MLP隠れ層比率

# 学習設定
batch_size: 64       # バッチサイズ
learning_rate: 0.0005 # 学習率
epochs: 90           # エポック数
dropout: 0.0         # ドロップアウト率
```

## 処理フロー図

```
[64,243,17,3] 入力
    ↓ Joint Embedding
[64×243,17,512] 
    ↓ + Position Encoding  
[64×243,17,512] 位置情報付き
    ↓ Dual-Scale Transformer × 5
[64×243,17,512] 特徴抽出済み
    ↓ reshape + pre_logits
[64,243,17,512] 最終表現
    ↓ Task Head
[64,243,17,3] (Pose) / [64,60] (Action) / [64×243,6890,3] (Mesh)
```

この構造により、MotionBERTは時空間の複雑な依存関係を効率的にモデル化し、様々な人体動作解析タスクに適用可能な汎用的な表現を学習できます。