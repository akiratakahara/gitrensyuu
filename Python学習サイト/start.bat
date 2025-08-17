@echo off
echo Python学習サイトを起動しています...
echo.

REM 依存関係のインストール確認
echo 依存関係をチェックしています...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Flaskがインストールされていません。インストールしています...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 依存関係のインストールに失敗しました。
        pause
        exit /b 1
    )
)

REM 問題データの作成
echo 問題データを作成しています...
python create_problems.py
if %errorlevel% neq 0 (
    echo 問題データの作成に失敗しました。
    pause
    exit /b 1
)

REM アプリケーションの起動
echo.
echo Python学習サイトを起動しています...
echo ブラウザで http://localhost:5000 にアクセスしてください。
echo.
echo 停止するには Ctrl+C を押してください。
echo.
python app.py

pause
