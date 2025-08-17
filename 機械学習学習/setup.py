"""
機械学習学習環境のセットアップスクリプト
必要なライブラリのインストールと環境の確認を行います
"""

import sys
import subprocess
import importlib
import platform


def check_python_version():
    """Pythonのバージョンを確認"""
    print("=== Pythonバージョンの確認 ===")
    print(f"Python バージョン: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("警告: Python 3.8以上が推奨されます")
        return False
    else:
        print("✓ Python バージョンは要件を満たしています")
        return True


def install_package(package):
    """パッケージのインストール"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"✗ {package}のインストールに失敗しました")
        return False


def check_package(package_name):
    """パッケージのインストール状況を確認"""
    try:
        # パッケージ名の変換
        if package_name == "scikit-learn":
            importlib.import_module("sklearn")
        else:
            importlib.import_module(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False


def install_required_packages():
    """必要なパッケージをインストール"""
    print("\n=== 必要なパッケージのインストール ===")
    
    # 基本パッケージ
    basic_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ]
    
    # オプションパッケージ
    optional_packages = [
        "jupyter",
        "ipykernel",
        "plotly",
        "scipy",
        "statsmodels"
    ]
    
    # 深層学習パッケージ（オプション）
    deep_learning_packages = [
        "tensorflow",
        "keras"
    ]
    
    print("基本パッケージをインストール中...")
    for package in basic_packages:
        if not check_package(package.replace("-", "_")):
            print(f"{package}をインストール中...")
            install_package(package)
        else:
            print(f"✓ {package}は既にインストールされています")
    
    print("\nオプションパッケージをインストール中...")
    for package in optional_packages:
        if not check_package(package.replace("-", "_")):
            print(f"{package}をインストール中...")
            install_package(package)
        else:
            print(f"✓ {package}は既にインストールされています")
    
    print("\n深層学習パッケージをインストール中...")
    for package in deep_learning_packages:
        if not check_package(package.replace("-", "_")):
            print(f"{package}をインストール中...")
            install_package(package)
        else:
            print(f"✓ {package}は既にインストールされています")


def verify_installation():
    """インストールの確認"""
    print("\n=== インストールの確認 ===")
    
    packages_to_check = {
        "numpy": "数値計算",
        "pandas": "データ分析",
        "matplotlib": "可視化",
        "seaborn": "統計的可視化",
        "scikit-learn": "機械学習",
        "jupyter": "Jupyter環境",
        "tensorflow": "深層学習"
    }
    
    all_installed = True
    
    for package, description in packages_to_check.items():
        if check_package(package):
            print(f"✓ {package} ({description})")
        else:
            print(f"✗ {package} ({description}) - インストールされていません")
            all_installed = False
    
    return all_installed


def create_sample_notebook():
    """サンプルノートブックの作成"""
    print("\n=== サンプルノートブックの作成 ===")
    
    try:
        # 基本的な機械学習の例
        sample_code = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 機械学習の最初の一歩\\n",
    "\\n",
    "このノートブックでは、scikit-learnを使って基本的な機械学習を行います。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 必要なライブラリのインポート\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "from sklearn.datasets import load_iris\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "from sklearn.ensemble import RandomForestClassifier\\n",
    "from sklearn.metrics import accuracy_score, classification_report\\n",
    "\\n",
    "print(\\"ライブラリのインポートが完了しました\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# アイリスデータセットの読み込み\\n",
    "iris = load_iris()\\n",
    "X = iris.data\\n",
    "y = iris.target\\n",
    "\\n",
    "print(f\\"データサイズ: {X.shape}\\")\\n",
    "print(f\\"特徴量名: {iris.feature_names}\\")\\n",
    "print(f\\"クラス名: {iris.target_names}\\")\\n",
    "print(f\\"目的変数: {np.unique(y)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# データの分割\\n",
    "X_train, X_test, y_train, y_test = train_test_split(\\n",
    "    X, y, test_size=0.2, random_state=42\\n",
    ")\\n",
    "\\n",
    "print(f\\"訓練データ: {X_train.shape}\\")\\n",
    "print(f\\"テストデータ: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ランダムフォレストモデルの学習\\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\\n",
    "model.fit(X_train, y_train)\\n",
    "\\n",
    "print(\\"モデルの学習が完了しました\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 予測と評価\\n",
    "y_pred = model.predict(X_test)\\n",
    "accuracy = accuracy_score(y_test, y_pred)\\n",
    "\\n",
    "print(f\\"正解率: {accuracy:.4f}\\")\\n",
    "print(\\"\\n詳細レポート:\\")\\n",
    "print(classification_report(y_test, y_pred, target_names=iris.target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 特徴量の重要度を可視化\\n",
    "importances = model.feature_importances_\\n",
    "indices = np.argsort(importances)[::-1]\\n",
    "\\n",
    "plt.figure(figsize=(10, 6))\\n",
    "plt.title(\\"特徴量の重要度\\")\\n",
    "plt.bar(range(len(importances)), importances[indices])\\n",
    "plt.xticks(range(len(importances)), [iris.feature_names[i] for i in indices], rotation=45)\\n",
    "plt.xlabel(\\"特徴量\\")\\n",
    "plt.ylabel(\\"重要度\\")\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
        
        with open('notebooks/02_機械学習入門.ipynb', 'w', encoding='utf-8') as f:
            f.write(sample_code)
        
        print("✓ サンプルノートブックを作成しました")
        return True
        
    except Exception as e:
        print(f"✗ サンプルノートブックの作成に失敗しました: {e}")
        return False


def main():
    """メイン関数"""
    print("機械学習学習環境のセットアップを開始します")
    print("=" * 50)
    
    # Pythonバージョンの確認
    if not check_python_version():
        print("Pythonのバージョンが要件を満たしていません")
        return
    
    # システム情報の表示
    print(f"\n=== システム情報 ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python実行パス: {sys.executable}")
    
    # パッケージのインストール
    install_required_packages()
    
    # インストールの確認
    if verify_installation():
        print("\n✓ すべてのパッケージが正常にインストールされました")
        
        # サンプルノートブックの作成
        create_sample_notebook()
        
        print("\n=== セットアップ完了 ===")
        print("以下のコマンドでJupyterを起動できます:")
        print("jupyter notebook")
        print("\nまたは、以下のコマンドでサンプルスクリプトを実行できます:")
        print("python src/data_preprocessing.py")
        print("python src/model_training.py")
        
    else:
        print("\n✗ 一部のパッケージのインストールに失敗しました")
        print("手動でインストールを試してください")


if __name__ == "__main__":
    main()
