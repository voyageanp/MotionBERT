# Claude Code カスタムコマンド

このディレクトリには、MotionBERT開発用のClaude Codeカスタムコマンドが含まれています。

## 利用可能コマンド

### `/git-acp [コミットメッセージ]`
Git の add, commit, push を一連で実行します。

```bash
# 使用例
/git-acp "Add Docker support"
/git-acp "Fix model architecture"
/git-acp  # メッセージ省略時は自動生成
```

**実行内容:**
1. `git add .` - すべての変更をステージング
2. `git commit -m "メッセージ"` - 変更をコミット
3. `git push` - リモートにプッシュ
4. 最終状態を表示

### `/docs-build`
日本語ドキュメントの構造と状態を確認します。

```bash
# 使用例
/docs-build
```

**実行内容:**
- 各ドキュメントファイルの存在確認
- ドキュメント構造の表示
- 関連ファイルの一覧表示

## ファイル構造

```
.claude/
├── commands                 # コマンド定義ファイル
├── handlers/               # コマンド実装
│   ├── git-acp.sh         # Git操作スクリプト
│   └── docs-build.sh      # ドキュメント確認スクリプト
└── README.md              # このファイル
```

## 注意事項

- スクリプトは実行権限が必要です（`chmod +x`）
- Git操作前に必ず現在の状態を確認してください
- コミットメッセージは意味のある内容にしてください

## トラブルシューティング

### コマンドが認識されない場合
Claude Codeを再起動してください。

### 権限エラーが発生する場合
```bash
chmod +x .claude/handlers/*.sh
```

### Git操作でエラーが発生する場合
- リモートリポジトリへのアクセス権限を確認
- 認証情報が正しく設定されているか確認