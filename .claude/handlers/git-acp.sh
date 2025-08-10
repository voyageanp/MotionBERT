#!/bin/bash

# git-acp: Add, Commit, and Push in one command
# Usage: /git-acp [commit message]

set -e

# コミットメッセージを取得
COMMIT_MSG="$*"

# コミットメッセージが空の場合はデフォルトメッセージを使用
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update files $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "🔍 Checking git status..."
git status

echo ""
echo "📝 Adding all changes..."
git add .

echo ""
echo "💾 Committing with message: '$COMMIT_MSG'"
git commit -m "$COMMIT_MSG"

echo ""
echo "🚀 Pushing to remote..."
git push

echo ""
echo "✅ Successfully completed git add, commit, and push!"
echo "📊 Final status:"
git status