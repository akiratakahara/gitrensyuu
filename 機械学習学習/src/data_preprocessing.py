"""
データ前処理のためのユーティリティ関数
機械学習の前処理でよく使う機能をまとめています
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_explore_data(file_path):
    """
    データを読み込んで基本的な情報を表示
    
    Args:
        file_path: データファイルのパスまたはDataFrame
        
    Returns:
        pandas.DataFrame: 読み込んだデータ
    """
    try:
        # 既にDataFrameの場合はそのまま使用
        if isinstance(file_path, pd.DataFrame):
            data = file_path
        # ファイルパスの場合
        elif isinstance(file_path, str):
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("サポートされていないファイル形式です")
        else:
            raise ValueError("ファイルパスまたはDataFrameを指定してください")
        
        print("=== データの基本情報 ===")
        print(f"データサイズ: {data.shape}")
        print(f"列名: {list(data.columns)}")
        print("\n=== データの最初の5行 ===")
        print(data.head())
        print("\n=== データ型 ===")
        print(data.dtypes)
        print("\n=== 基本統計量 ===")
        print(data.describe())
        print("\n=== 欠損値の確認 ===")
        print(data.isnull().sum())
        
        return data
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None


def handle_missing_values(data, strategy='drop'):
    """
    欠損値の処理
    
    Args:
        data (pandas.DataFrame): 処理対象のデータ
        strategy (str): 処理方法 ('drop', 'mean', 'median', 'mode')
        
    Returns:
        pandas.DataFrame: 処理後のデータ
    """
    if strategy == 'drop':
        # 欠損値を含む行を削除
        cleaned_data = data.dropna()
        print(f"欠損値を削除: {len(data) - len(cleaned_data)}行を削除")
        
    elif strategy == 'mean':
        # 数値列の欠損値を平均値で補完
        cleaned_data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                mean_val = data[col].mean()
                cleaned_data[col].fillna(mean_val, inplace=True)
                print(f"{col}: 平均値 {mean_val:.2f} で補完")
                
    elif strategy == 'median':
        # 数値列の欠損値を中央値で補完
        cleaned_data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                cleaned_data[col].fillna(median_val, inplace=True)
                print(f"{col}: 中央値 {median_val:.2f} で補完")
                
    elif strategy == 'mode':
        # カテゴリ列の欠損値を最頻値で補完
        cleaned_data = data.copy()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if data[col].isnull().any():
                mode_val = data[col].mode()[0]
                cleaned_data[col].fillna(mode_val, inplace=True)
                print(f"{col}: 最頻値 '{mode_val}' で補完")
    
    return cleaned_data


def encode_categorical_variables(data, columns=None):
    """
    カテゴリ変数を数値にエンコーディング
    
    Args:
        data (pandas.DataFrame): 処理対象のデータ
        columns (list): エンコーディング対象の列名（Noneの場合は自動判定）
        
    Returns:
        pandas.DataFrame: エンコーディング後のデータ
        dict: エンコーディングのマッピング情報
    """
    if columns is None:
        # オブジェクト型の列を自動判定
        columns = data.select_dtypes(include=['object']).columns.tolist()
    
    encoded_data = data.copy()
    encoders = {}
    
    for col in columns:
        if col in data.columns:
            # LabelEncoderを使用
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
            print(f"{col}: {len(le.classes_)}カテゴリをエンコーディング")
    
    return encoded_data, encoders


def scale_numerical_features(data, columns=None, scaler_type='standard'):
    """
    数値特徴量のスケーリング
    
    Args:
        data (pandas.DataFrame): 処理対象のデータ
        columns (list): スケーリング対象の列名（Noneの場合は自動判定）
        scaler_type (str): スケーリング方法 ('standard', 'minmax')
        
    Returns:
        pandas.DataFrame: スケーリング後のデータ
        object: スケーラーオブジェクト
    """
    if columns is None:
        # 数値型の列を自動判定
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    scaled_data = data.copy()
    scaled_data[columns] = scaler.fit_transform(data[columns])
    
    print(f"{len(columns)}列を{scaler_type}スケーリングしました")
    
    return scaled_data, scaler


def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    データを訓練データとテストデータに分割
    
    Args:
        data (pandas.DataFrame): 分割対象のデータ
        target_column (str): 目的変数の列名
        test_size (float): テストデータの割合
        random_state (int): 乱数シード
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_column not in data.columns:
        raise ValueError(f"目的変数 '{target_column}' が見つかりません")
    
    # 特徴量と目的変数を分離
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"データ分割完了:")
    print(f"  訓練データ: {X_train.shape}")
    print(f"  テストデータ: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def create_sample_dataset():
    """
    サンプルデータセットを作成（テスト用）
    
    Returns:
        pandas.DataFrame: サンプルデータ
    """
    np.random.seed(42)
    
    # サンプルデータの作成
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['高校', '大学', '大学院'], n_samples),
        'city': np.random.choice(['東京', '大阪', '名古屋', '福岡'], n_samples),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    }
    
    # 欠損値を追加
    data['age'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['income'][np.random.choice(n_samples, 30, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    
    # 年齢を整数に調整（欠損値を除外）
    df['age'] = df['age'].round().fillna(0).astype(int)
    
    return df


if __name__ == "__main__":
    # サンプルデータで動作確認
    print("=== データ前処理のテスト ===")
    
    # サンプルデータ作成
    sample_data = create_sample_dataset()
    print("サンプルデータを作成しました")
    
    # データの探索
    print("\n" + "="*50)
    data = load_and_explore_data(sample_data)
    
    # 欠損値の処理
    print("\n" + "="*50)
    cleaned_data = handle_missing_values(data, strategy='mean')
    
    # カテゴリ変数のエンコーディング
    print("\n" + "="*50)
    encoded_data, encoders = encode_categorical_variables(cleaned_data)
    
    # 数値特徴量のスケーリング
    print("\n" + "="*50)
    scaled_data, scaler = scale_numerical_features(encoded_data, ['age', 'income'])
    
    # データ分割
    print("\n" + "="*50)
    X_train, X_test, y_train, y_test = split_data(scaled_data, 'satisfaction')
    
    print("\nデータ前処理が完了しました！")
