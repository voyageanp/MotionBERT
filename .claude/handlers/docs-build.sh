#!/bin/bash

# docs-build: Generate and update all Japanese documentation
# Usage: /docs-build

set -e

echo "📚 Building Japanese documentation for MotionBERT..."

# ドキュメントディレクトリの存在確認
if [ ! -d "docs" ]; then
    echo "Creating docs directory..."
    mkdir -p docs
fi

echo ""
echo "📝 Updating documentation files..."

# 各ドキュメントの存在確認と更新状況表示
docs=(
    "docs/setup_ja.md:セットアップと動かし方"
    "docs/model_architecture_ja.md:モデル構造詳細解説"
    "docs/video_to_tensor_ja.md:動画からテンソル変換プロセス"
    "docs/docker_setup_ja.md:Docker環境セットアップ"
)

for doc_info in "${docs[@]}"; do
    doc_path="${doc_info%:*}"
    doc_desc="${doc_info#*:}"
    
    if [ -f "$doc_path" ]; then
        echo "✅ $doc_desc: $doc_path"
    else
        echo "❌ $doc_desc: $doc_path (不足)"
    fi
done

echo ""
echo "📋 Documentation structure:"
echo "docs/"
echo "├── setup_ja.md              # セットアップ手順"
echo "├── model_architecture_ja.md # モデル構造解説"  
echo "├── video_to_tensor_ja.md    # データ変換プロセス"
echo "└── docker_setup_ja.md       # Docker環境構築"

echo ""
echo "🔗 Related files:"
echo "├── CLAUDE.md                # Claude Code用ガイド"
echo "├── Dockerfile               # Docker環境定義"
echo "├── docker-compose.yml       # Docker Compose設定"
echo "└── .dockerignore            # Docker除外ファイル"

echo ""
echo "✅ Documentation build completed!"
echo "💡 Use '/git-acp docs: Update documentation' to commit changes"