# Gitコマンドリファレンス

## 基本的なGitコマンド

### 初期化と設定
```bash
# リポジトリを初期化
git init

# ユーザー名を設定
git config --global user.name "Your Name"

# メールアドレスを設定
git config --global user.email "your.email@example.com"

# 設定を確認
git config --list
```

### 基本的な操作
```bash
# ファイルの状態を確認
git status

# ファイルをステージングエリアに追加
git add <filename>
git add .  # すべてのファイルを追加

# 変更をコミット
git commit -m "コミットメッセージ"

# コミット履歴を表示
git log
git log --oneline  # 簡潔に表示
```

### ブランチ操作
```bash
# ブランチ一覧を表示
git branch

# 新しいブランチを作成
git branch <branch-name>

# ブランチを切り替え
git checkout <branch-name>
git switch <branch-name>  # 新しい方法

# ブランチを作成して切り替え
git checkout -b <branch-name>
git switch -c <branch-name>
```

### リモート操作
```bash
# リモートリポジトリを追加
git remote add origin <repository-url>

# リモートリポジトリを確認
git remote -v

# リモートから取得
git fetch origin

# リモートから取得してマージ
git pull origin <branch-name>

# リモートにプッシュ
git push origin <branch-name>
```

## CursorエディタでのGit操作

### 1. ソースコントロールパネル
- `Ctrl+Shift+G` でソースコントロールパネルを開く
- 変更されたファイルが表示される

### 2. 変更のステージング
- ファイル名の横の「+」ボタンをクリック
- または、ファイルを右クリックして「Stage Changes」

### 3. コミット
- メッセージを入力して「✓」ボタンをクリック
- または、`Ctrl+Enter`

### 4. プッシュ/プル
- ソースコントロールパネルの「...」メニューから選択

## よく使うGitHub操作

### 1. リポジトリの作成
1. GitHub.comにアクセス
2. 「New repository」をクリック
3. リポジトリ名と説明を入力
4. 公開/非公開を選択
5. 「Create repository」をクリック

### 2. 既存プロジェクトをGitHubにプッシュ
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/username/repository.git
git push -u origin main
```

### 3. プルリクエストの作成
1. GitHubでリポジトリにアクセス
2. 「Pull requests」タブをクリック
3. 「New pull request」をクリック
4. ベースブランチと比較ブランチを選択
5. タイトルと説明を入力
6. 「Create pull request」をクリック

## トラブルシューティング

### よくある問題と解決方法

#### 1. 認証エラー
```bash
# Personal Access Tokenを使用
git remote set-url origin https://username:token@github.com/username/repository.git
```

#### 2. マージコンフリクト
```bash
# コンフリクトを解決後
git add .
git commit -m "Resolve merge conflicts"
```

#### 3. 間違ったコミットを取り消し
```bash
# 直前のコミットを取り消し（変更は保持）
git reset --soft HEAD~1

# 直前のコミットを取り消し（変更も削除）
git reset --hard HEAD~1
```

## ベストプラクティス

1. **コミットメッセージ**
   - 明確で簡潔なメッセージ
   - 現在形で記述（例：「Add new feature」）

2. **ブランチ戦略**
   - `main`ブランチは常に安定版
   - 機能開発は`feature/`ブランチ
   - バグ修正は`fix/`ブランチ

3. **定期的な同期**
   - 作業開始前に`git pull`
   - 作業完了後に`git push`

4. **コミットの粒度**
   - 論理的な単位でコミット
   - 小さく、頻繁にコミット

