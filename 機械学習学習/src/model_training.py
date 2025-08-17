"""
機械学習モデルの学習と評価のためのユーティリティ関数
基本的な分類・回帰問題に対応しています
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def train_classification_model(X_train, y_train, model_type='logistic', **kwargs):
    """
    分類モデルの学習
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        model_type (str): モデルの種類
        **kwargs: モデルのパラメータ
        
    Returns:
        object: 学習済みモデル
    """
    models = {
        'logistic': LogisticRegression(random_state=42, **kwargs),
        'decision_tree': DecisionTreeClassifier(random_state=42, **kwargs),
        'random_forest': RandomForestClassifier(random_state=42, **kwargs),
        'svm': SVC(random_state=42, **kwargs),
        'knn': KNeighborsClassifier(**kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
    
    model = models[model_type]
    model.fit(X_train, y_train)
    
    print(f"{model_type}モデルの学習が完了しました")
    return model


def train_regression_model(X_train, y_train, model_type='linear', **kwargs):
    """
    回帰モデルの学習
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        model_type (str): モデルの種類
        **kwargs: モデルのパラメータ
        
    Returns:
        object: 学習済みモデル
    """
    models = {
        'linear': LinearRegression(**kwargs),
        'decision_tree': DecisionTreeRegressor(random_state=42, **kwargs),
        'random_forest': RandomForestRegressor(random_state=42, **kwargs),
        'svm': SVR(**kwargs),
        'knn': KNeighborsRegressor(**kwargs)
    }
    
    if model_type not in models:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
    
    model = models[model_type]
    model.fit(X_train, y_train)
    
    print(f"{model_type}モデルの学習が完了しました")
    return model


def evaluate_classification_model(model, X_test, y_test, model_name="モデル"):
    """
    分類モデルの評価
    
    Args:
        model: 学習済みモデル
        X_test: テストデータの特徴量
        y_test: テストデータの目的変数
        model_name (str): モデル名
        
    Returns:
        dict: 評価指標
    """
    # 予測
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # 評価指標の計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 結果の表示
    print(f"=== {model_name}の評価結果 ===")
    print(f"正解率 (Accuracy): {accuracy:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall): {recall:.4f}")
    print(f"F1スコア: {f1:.4f}")
    
    # 詳細レポート
    print("\n=== 詳細レポート ===")
    print(classification_report(y_test, y_pred))
    
    # 混同行列
    print("\n=== 混同行列 ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 混同行列の可視化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - 混同行列')
    plt.ylabel('実際の値')
    plt.xlabel('予測値')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def evaluate_regression_model(model, X_test, y_test, model_name="モデル"):
    """
    回帰モデルの評価
    
    Args:
        model: 学習済みモデル
        X_test: テストデータの特徴量
        y_test: テストデータの目的変数
        model_name (str): モデル名
        
    Returns:
        dict: 評価指標
    """
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価指標の計算
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 結果の表示
    print(f"=== {model_name}の評価結果 ===")
    print(f"平均二乗誤差 (MSE): {mse:.4f}")
    print(f"平均二乗誤差の平方根 (RMSE): {rmse:.4f}")
    print(f"決定係数 (R²): {r2:.4f}")
    
    # 予測vs実際の値の散布図
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('実際の値')
    plt.ylabel('予測値')
    plt.title(f'{model_name} - 予測vs実際の値')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 残差プロット
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('予測値')
    plt.ylabel('残差')
    plt.title(f'{model_name} - 残差プロット')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'residuals': residuals
    }


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    クロスバリデーションによるモデル評価
    
    Args:
        model: 評価対象のモデル
        X: 特徴量
        y: 目的変数
        cv (int): 分割数
        scoring (str): 評価指標
        
    Returns:
        dict: クロスバリデーション結果
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"=== クロスバリデーション結果 ({cv}分割) ===")
    print(f"スコア: {scores}")
    print(f"平均スコア: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
    """
    ハイパーパラメータのチューニング
    
    Args:
        model: ベースモデル
        param_grid (dict): パラメータグリッド
        X_train: 訓練データの特徴量
        y_train: 訓練データの目的変数
        cv (int): 分割数
        scoring (str): 評価指標
        
    Returns:
        object: 最適化されたモデル
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
    )
    
    print("ハイパーパラメータのチューニングを開始します...")
    grid_search.fit(X_train, y_train)
    
    print(f"最適なパラメータ: {grid_search.best_params_}")
    print(f"最適なスコア: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def compare_models(X_train, X_test, y_train, y_test, model_types=None):
    """
    複数のモデルを比較
    
    Args:
        X_train, X_test: 訓練・テストデータの特徴量
        y_train, y_test: 訓練・テストデータの目的変数
        model_types (list): 比較するモデルの種類
        
    Returns:
        dict: 各モデルの評価結果
    """
    if model_types is None:
        model_types = ['logistic', 'decision_tree', 'random_forest', 'svm', 'knn']
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"{model_type}モデルの学習と評価")
        print(f"{'='*50}")
        
        try:
            # モデルの学習
            model = train_classification_model(X_train, y_train, model_type)
            
            # モデルの評価
            result = evaluate_classification_model(model, X_test, y_test, model_type)
            results[model_type] = result
            
            # クロスバリデーション
            cv_result = cross_validate_model(model, X_train, y_train)
            results[model_type]['cv_result'] = cv_result
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            results[model_type] = {'error': str(e)}
    
    return results


def plot_feature_importance(model, feature_names, model_name="モデル"):
    """
    特徴量の重要度を可視化
    
    Args:
        model: 学習済みモデル
        feature_names (list): 特徴量名のリスト
        model_name (str): モデル名
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("このモデルでは特徴量の重要度を取得できません")
        return
    
    # 重要度の降順でソート
    indices = np.argsort(importances)[::-1]
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name} - 特徴量の重要度')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('特徴量')
    plt.ylabel('重要度')
    plt.tight_layout()
    plt.show()
    
    # 重要度の詳細表示
    print(f"\n=== {model_name} - 特徴量の重要度 ===")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    # サンプルデータで動作確認
    print("=== 機械学習モデルのテスト ===")
    
    # サンプルデータの作成（分類問題）
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, random_state=42
    )
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"データサイズ: {X.shape}")
    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    
    # 複数モデルの比較
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # 最良モデルの特定
    best_model = None
    best_score = 0
    
    for model_name, result in results.items():
        if 'error' not in result and result['accuracy'] > best_score:
            best_score = result['accuracy']
            best_model = model_name
    
    if best_model:
        print(f"\n最良のモデル: {best_model} (正解率: {best_score:.4f})")
        
        # 最良モデルで特徴量の重要度を表示
        model = train_classification_model(X_train, y_train, best_model)
        feature_names = [f"特徴量_{i}" for i in range(X.shape[1])]
        plot_feature_importance(model, feature_names, best_model)

