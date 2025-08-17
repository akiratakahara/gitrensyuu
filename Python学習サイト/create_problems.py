import json
import os

def create_sample_problems():
    """東京大学のPython教材を参考にした体系的問題データを作成"""
    
    problems = [
        # レベル1: 基礎編（初心者向け）
        {
            "id": 1,
            "title": "数値演算の基礎",
            "description": "変数 `result` に 15 + 25 * 2 の結果を代入してください。演算子の優先順位に注意してください。",
            "code_template": "result = ___",
            "answer": "65",
            "explanation": "演算子の優先順位により、まず 25 * 2 = 50 が計算され、その後 15 + 50 = 65 となります。",
            "level": 1,
            "category": "数値演算",
            "hint": "掛け算は足し算より優先順位が高いです"
        },
        {
            "id": 2,
            "title": "変数の定義と代入",
            "description": "変数 `name` に文字列 'Python' を代入してください。",
            "code_template": "name = ___",
            "answer": "Python",
            "explanation": "Pythonでは変数に値を代入する際、`=` 演算子を使用します。文字列は引用符で囲みます。",
            "level": 1,
            "category": "変数・データ型",
            "hint": "文字列は引用符で囲む必要があります"
        },
        {
            "id": 3,
            "title": "関数の定義",
            "description": "引数 `x` を受け取り、その2倍を返す関数 `double` を定義してください。",
            "code_template": "def double(x):\n    return ___",
            "answer": "x * 2",
            "explanation": "関数内で `return` 文を使用して値を返します。`x * 2` で引数の2倍を計算できます。",
            "level": 1,
            "category": "関数",
            "hint": "引数に2を掛けてください"
        },
        {
            "id": 4,
            "title": "if文による条件分岐",
            "description": "変数 `age` が 18 以上かどうかを判定する条件を書いてください。",
            "code_template": "if age ___ 18:",
            "answer": ">=",
            "explanation": "`>=` は「以上」を表す比較演算子です。`age >= 18` で18歳以上かどうかを判定できます。",
            "level": 1,
            "category": "制御構文",
            "hint": "18以上を表す演算子を考えてください"
        },
        {
            "id": 5,
            "title": "assert文によるテスト",
            "description": "変数 `x` が正の数であることを確認するassert文を書いてください。",
            "code_template": "assert x ___ 0",
            "answer": ">",
            "explanation": "`assert x > 0` で、xが正の数であることを確認できます。条件がFalseの場合、AssertionErrorが発生します。",
            "level": 1,
            "category": "テスト・デバッグ",
            "hint": "正の数は0より大きいです"
        },
        
        # レベル2: 中級編（実践的）
        {
            "id": 6,
            "title": "文字列の操作",
            "description": "文字列 `text` の長さを取得してください。",
            "code_template": "length = ___",
            "answer": "len(text)",
            "explanation": "`len()` 関数で文字列の長さ（文字数）を取得できます。",
            "level": 2,
            "category": "文字列操作",
            "hint": "長さを取得する関数名を考えてください"
        },
        {
            "id": 7,
            "title": "リストの作成と操作",
            "description": "リスト `numbers` の各要素を2倍にする処理を書いてください。",
            "code_template": "for num in numbers:\n    print(num * ___)",
            "answer": "2",
            "explanation": "for文でリストの各要素にアクセスし、それぞれに2を掛けて出力します。",
            "level": 2,
            "category": "リスト・ループ",
            "hint": "各要素を2倍にする必要があります"
        },
        {
            "id": 8,
            "title": "辞書の値の取得",
            "description": "辞書 `person` からキー 'name' の値を取得してください。",
            "code_template": "name = person['___']",
            "answer": "name",
            "explanation": "辞書から値を取得するには、角括弧内にキーを指定します。`person['name']` で名前の値を取得できます。",
            "level": 2,
            "category": "辞書・データ構造",
            "hint": "キーの名前をそのまま書いてください"
        },
        {
            "id": 9,
            "title": "例外処理",
            "description": "エラーが発生した場合の処理を書いてください。",
            "code_template": "try:\n    # 処理\n___ ValueError:\n    print('エラーが発生しました')",
            "answer": "except",
            "explanation": "`except` キーワードで特定の例外をキャッチします。`except ValueError:` でValueErrorが発生した場合の処理を定義できます。",
            "level": 2,
            "category": "例外処理",
            "hint": "try文と対になるキーワードです"
        },
        {
            "id": 10,
            "title": "リスト内包表記",
            "description": "リスト `numbers` の各要素を2倍にした新しいリストを作成してください。",
            "code_template": "doubled = [num * 2 for num in ___]",
            "answer": "numbers",
            "explanation": "リスト内包表記では、元のリストの名前を指定します。`[num * 2 for num in numbers]` で各要素を2倍にしたリストを作成できます。",
            "level": 2,
            "category": "リスト内包表記",
            "hint": "元のリストの変数名を書いてください"
        },
        
        # レベル3: 上級編（応用）
        {
            "id": 11,
            "title": "クラスの定義",
            "description": "`Person` クラスを定義し、コンストラクタで `name` と `age` を受け取るようにしてください。",
            "code_template": "class Person:\n    def __init__(self, name, age):\n        self.name = ___\n        self.age = ___",
            "answer": "name, age",
            "explanation": "コンストラクタでは、受け取った引数をインスタンス変数に代入します。`self.name = name` と `self.age = age` で設定できます。",
            "level": 3,
            "category": "クラス・オブジェクト指向",
            "hint": "引数名をそのまま使用します"
        },
        {
            "id": 12,
            "title": "ファイルの読み込み",
            "description": "ファイル 'data.txt' を読み込んで内容を表示してください。",
            "code_template": "___ open('data.txt', 'r') as f:\n    content = f.read()\n    print(content)",
            "answer": "with",
            "explanation": "`with` 文を使用することで、ファイルの自動クローズが保証されます。リソースの適切な管理が可能です。",
            "level": 3,
            "category": "ファイル操作",
            "hint": "リソース管理のためのキーワードです"
        },
        {
            "id": 13,
            "title": "モジュールのインポート",
            "description": "数学関数を使用するために `math` モジュールをインポートしてください。",
            "code_template": "___ math",
            "answer": "import",
            "explanation": "`import math` でmathモジュールをインポートできます。これにより、`math.sqrt()`, `math.pi` などの数学関数や定数が使用可能になります。",
            "level": 3,
            "category": "モジュール・パッケージ",
            "hint": "モジュールを読み込むキーワードです"
        },
        {
            "id": 14,
            "title": "ジェネレータ関数",
            "description": "1からnまでの数を生成するジェネレータ関数 `count_up` を作成してください。",
            "code_template": "def count_up(n):\n    for i in range(1, n + 1):\n        ___ i",
            "answer": "yield",
            "explanation": "ジェネレータ関数では `yield` キーワードを使用して値を生成します。`yield i` で各値を一つずつ返すことができます。",
            "level": 3,
            "category": "ジェネレータ",
            "hint": "値を返すキーワードです"
        },
        {
            "id": 15,
            "title": "デコレータ",
            "description": "関数の実行時間を計測するデコレータ `timer` を作成してください。",
            "code_template": "def timer(func):\n    def wrapper(*args, **kwargs):\n        import time\n        start = time.___()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'実行時間: {end - start}秒')\n        return result\n    return wrapper",
            "answer": "time",
            "explanation": "`time.time()` で現在時刻を取得します。開始時刻と終了時刻の差で実行時間を計算できます。",
            "level": 3,
            "category": "デコレータ",
            "hint": "時刻を取得する関数名です"
        },
        
        # レベル4: プロフェッショナル編（実践・応用）
        {
            "id": 16,
            "title": "NumPy配列の作成",
            "description": "NumPyを使用して1から10までの配列を作成してください。",
            "code_template": "import numpy as np\narray = np.___(1, 11)",
            "answer": "arange",
            "explanation": "`np.arange(1, 11)` で1から10までの配列を作成できます。arangeは指定された範囲の連続した数値の配列を生成します。",
            "level": 4,
            "category": "データ分析・NumPy",
            "hint": "範囲を指定して配列を作成する関数です"
        },
        {
            "id": 17,
            "title": "pandas DataFrameの作成",
            "description": "pandasを使用して、列名が 'name' と 'age' のDataFrameを作成してください。",
            "code_template": "import pandas as pd\ndf = pd.DataFrame({\n    'name': ['Alice', 'Bob'],\n    'age': [25, 30]\n})",
            "answer": "DataFrame",
            "explanation": "`pd.DataFrame()` で辞書形式のデータからDataFrameを作成できます。列名とデータを辞書で指定します。",
            "level": 4,
            "category": "データ分析・pandas",
            "hint": "pandasの主要なデータ構造です"
        },
        {
            "id": 18,
            "title": "機械学習モデルの学習",
            "description": "scikit-learnを使用して、データを訓練データとテストデータに分割してください。",
            "code_template": "from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=___)",
            "answer": "0.2",
            "explanation": "`test_size=0.2` でデータの20%をテストデータとして分割します。残りの80%が訓練データとなります。",
            "level": 4,
            "category": "機械学習・scikit-learn",
            "hint": "テストデータの割合を小数で指定します"
        },
        {
            "id": 19,
            "title": "matplotlibでのグラフ作成",
            "description": "matplotlibを使用して線グラフを作成してください。",
            "code_template": "import matplotlib.pyplot as plt\nplt.___(x, y)\nplt.show()",
            "answer": "plot",
            "explanation": "`plt.plot(x, y)` でx座標とy座標のデータから線グラフを作成できます。`plt.show()` でグラフを表示します。",
            "level": 4,
            "category": "データ可視化・matplotlib",
            "hint": "線グラフを作成する関数です"
        },
        {
            "id": 20,
            "title": "Flaskルートの定義",
            "description": "Flaskでホームページのルートを定義してください。",
            "code_template": "@app.route('/')\ndef home():\n    return 'Hello, World!'",
            "answer": "route",
            "explanation": "`@app.route('/')` でルートパス '/' へのアクセスを処理する関数を定義できます。これにより、Webアプリケーションのエンドポイントを作成できます。",
            "level": 4,
            "category": "Web開発・Flask",
            "hint": "ルートを定義するデコレータです"
        }
    ]
    
    # データディレクトリが存在しない場合は作成
    os.makedirs('data', exist_ok=True)
    
    # JSONファイルに保存
    with open('data/problems.json', 'w', encoding='utf-8') as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
    
    print(f"{len(problems)}個の問題データを作成しました！")
    print("ファイル: data/problems.json")
    print("\n学習レベル:")
    print("- レベル1: 基礎編（初心者向け）")
    print("- レベル2: 中級編（実践的）")
    print("- レベル3: 上級編（応用）")
    print("- レベル4: プロフェッショナル編（実践・応用）")

if __name__ == "__main__":
    create_sample_problems()
