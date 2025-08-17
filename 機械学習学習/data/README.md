# データディレクトリ

このディレクトリには、機械学習で使用するデータセットやサンプルデータを格納します。

## ディレクトリ構成

```
data/
├── README.md          # このファイル
├── raw/               # 生データ（元のデータセット）
├── processed/         # 前処理済みデータ
├── external/          # 外部データソース
└── interim/           # 中間処理データ
```

## データの種類

### 1. 生データ（raw/）
- 元のデータセット
- 変更を加えない
- バージョン管理に含める

### 2. 前処理済みデータ（processed/）
- 前処理が完了したデータ
- 機械学習で直接使用可能
- バージョン管理に含める

### 3. 外部データ（external/）
- 外部APIやデータベースから取得
- 定期的に更新される可能性
- .gitignoreに追加することを推奨

### 4. 中間処理データ（interim/）
- 処理途中のデータ
- 一時的な保存用
- .gitignoreに追加することを推奨

## 推奨データセット

### 初心者向け
1. **Iris（アイリス）データセット**
   - 花の種類を分類する問題
   - 150サンプル、4特徴量
   - scikit-learnに標準搭載

2. **Boston Housing（ボストン住宅価格）データセット**
   - 住宅価格を予測する回帰問題
   - 506サンプル、13特徴量
   - scikit-learnに標準搭載

3. **Digits（手書き数字）データセット**
   - 手書き数字を認識する分類問題
   - 1797サンプル、64特徴量
   - scikit-learnに標準搭載

### 中級者向け
1. **MNIST**
   - 手書き数字の大規模データセット
   - 60,000訓練サンプル、10,000テストサンプル
   - 画像認識の基本

2. **CIFAR-10**
   - カラー画像の分類データセット
   - 10クラス、50,000訓練サンプル
   - 画像認識の応用

3. **Titanic**
   - 生存予測の分類問題
   - 891サンプル、複数の特徴量
   - Kaggleの有名なコンペティション

## データの取得方法

### 1. scikit-learnの組み込みデータセット
```python
from sklearn.datasets import load_iris, load_boston, load_digits

# データの読み込み
iris = load_iris()
X = iris.data
y = iris.target
```

### 2. pandasでの読み込み
```python
import pandas as pd

# CSVファイルの読み込み
data = pd.read_csv('data.csv')

# Excelファイルの読み込み
data = pd.read_excel('data.xlsx')
```

### 3. オンラインデータソース
- **Kaggle**: 機械学習コンペティションとデータセット
- **UCI Machine Learning Repository**: 学術的なデータセット
- **Google Dataset Search**: データセットの検索エンジン

## データの前処理

### 基本的な前処理手順
1. **データの読み込みと確認**
2. **欠損値の処理**
3. **カテゴリ変数のエンコーディング**
4. **特徴量のスケーリング**
5. **データの分割（訓練・テスト）**

### 前処理の例
```python
from src.data_preprocessing import (
    load_and_explore_data,
    handle_missing_values,
    encode_categorical_variables,
    scale_numerical_features,
    split_data
)

# データの読み込み
data = load_and_explore_data('data.csv')

# 欠損値の処理
cleaned_data = handle_missing_values(data, strategy='mean')

# カテゴリ変数のエンコーディング
encoded_data, encoders = encode_categorical_variables(cleaned_data)

# 数値特徴量のスケーリング
scaled_data, scaler = scale_numerical_features(encoded_data)

# データの分割
X_train, X_test, y_train, y_test = split_data(scaled_data, 'target_column')
```

## 注意事項

1. **データの品質**: 欠損値、異常値、データ型の確認
2. **データのサイズ**: メモリ使用量の確認
3. **データの機密性**: 個人情報や機密情報の取り扱い
4. **ライセンス**: データセットの使用条件の確認

## 次のステップ

1. 基本的なデータセットで学習を開始
2. データの前処理手法を習得
3. 実際のデータセットでの実践
4. データの可視化と分析

データは機械学習の成功の鍵です。適切なデータの選択と前処理を心がけましょう！

