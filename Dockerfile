# MotionBERT Docker Environment
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda

# 作業ディレクトリ設定
WORKDIR /workspace

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージの更新
RUN pip install --upgrade pip setuptools wheel

# 基本的なPythonライブラリのインストール
RUN pip install \
    tensorboardX \
    tqdm \
    easydict \
    prettytable \
    opencv-python \
    imageio-ffmpeg \
    matplotlib \
    roma \
    ipdb \
    pytorch-metric-learning

# SMPLXとその依存関係のインストール
RUN pip install "smplx[all]"

# Jupyter notebook関連（開発用）
RUN pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    cython_bbox

# AlphaPose用の追加パッケージ（オプション）
RUN pip install \
    yacs \
    pycocotools \
    scipy \
    torchvision

# ソースコードをコピー
COPY . /workspace/

# 権限設定
RUN chmod -R a+rX /workspace

# エントリーポイント
CMD ["/bin/bash"]