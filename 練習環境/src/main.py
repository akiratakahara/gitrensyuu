#!/usr/bin/env python3
"""
Git練習用のサンプルPythonスクリプト
"""

def hello_git():
    """Gitの練習用の挨拶関数"""
    print("Hello, Git!")
    print("CursorエディタでGitを使いこなそう！")
    return "Git練習中"

def calculate_sum(a, b):
    """2つの数の合計を計算する関数"""
    return a + b

def main():
    """メイン関数"""
    print("=== Git練習プロジェクト ===")
    hello_git()
    
    # 簡単な計算の例
    result = calculate_sum(5, 3)
    print(f"5 + 3 = {result}")
    
    print("=== 練習完了 ===")

if __name__ == "__main__":
    main()

