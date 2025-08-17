from flask import Flask, render_template, request, jsonify, session
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'python_learning_secret_key_2024'

# 問題データの読み込み
def load_problems():
    """問題データを読み込む"""
    try:
        with open('data/problems.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# ユーザーの進捗を管理
def get_user_progress():
    """ユーザーの進捗を取得"""
    if 'progress' not in session:
        session['progress'] = {
            'completed': [],
            'current_level': 1,
            'score': 0,
            'start_date': datetime.now().isoformat()
        }
    return session['progress']

def update_user_progress(problem_id, is_correct):
    """ユーザーの進捗を更新"""
    progress = get_user_progress()
    
    if problem_id not in progress['completed']:
        progress['completed'].append(problem_id)
        if is_correct:
            progress['score'] += 10
    
    # レベルアップの判定
    if len(progress['completed']) >= 5 and progress['current_level'] == 1:
        progress['current_level'] = 2
    elif len(progress['completed']) >= 10 and progress['current_level'] == 2:
        progress['current_level'] = 3
    
    session['progress'] = progress

@app.route('/')
def index():
    """ホームページ"""
    progress = get_user_progress()
    return render_template('index.html', progress=progress)

@app.route('/problems')
def problems():
    """問題一覧ページ"""
    problems_data = load_problems()
    progress = get_user_progress()
    
    # レベル別に問題を分類
    level_problems = {}
    for problem in problems_data:
        level = problem.get('level', 1)
        if level not in level_problems:
            level_problems[level] = []
        level_problems[level].append(problem)
    
    return render_template('problems.html', 
                         level_problems=level_problems, 
                         progress=progress)

@app.route('/problem/<int:problem_id>')
def problem_detail(problem_id):
    """個別問題ページ"""
    problems_data = load_problems()
    problem = None
    
    for p in problems_data:
        if p['id'] == problem_id:
            problem = p
            break
    
    if not problem:
        return "問題が見つかりません", 404
    
    progress = get_user_progress()
    return render_template('problem_detail.html', 
                         problem=problem, 
                         progress=progress)

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """回答をチェック"""
    data = request.get_json()
    problem_id = data.get('problem_id')
    user_answer = data.get('answer')
    
    problems_data = load_problems()
    problem = None
    
    for p in problems_data:
        if p['id'] == problem_id:
            problem = p
            break
    
    if not problem:
        return jsonify({'error': '問題が見つかりません'}), 404
    
    # 回答をチェック
    is_correct = check_answer(problem, user_answer)
    
    # 進捗を更新
    update_user_progress(problem_id, is_correct)
    
    return jsonify({
        'correct': is_correct,
        'explanation': problem.get('explanation', ''),
        'progress': get_user_progress()
    })

def check_answer(problem, user_answer):
    """回答が正しいかチェック"""
    correct_answer = problem.get('answer', '')
    
    # 文字列の場合は大文字小文字を無視
    if isinstance(correct_answer, str) and isinstance(user_answer, str):
        return user_answer.strip().lower() == correct_answer.strip().lower()
    
    # 数値の場合は型変換を試行
    try:
        if isinstance(correct_answer, (int, float)):
            user_num = float(user_answer)
            return abs(user_num - correct_answer) < 0.001
    except (ValueError, TypeError):
        pass
    
    return str(user_answer) == str(correct_answer)

@app.route('/progress')
def progress():
    """進捗ページ"""
    progress_data = get_user_progress()
    problems_data = load_problems()
    
    # 完了した問題の詳細を取得
    completed_problems = []
    for problem_id in progress_data['completed']:
        for problem in problems_data:
            if problem['id'] == problem_id:
                completed_problems.append(problem)
                break
    
    return render_template('progress.html', 
                         progress=progress_data, 
                         completed_problems=completed_problems)

@app.route('/reset_progress')
def reset_progress():
    """進捗をリセット"""
    session.pop('progress', None)
    return jsonify({'message': '進捗がリセットされました'})

if __name__ == '__main__':
    # 問題データが存在しない場合は作成
    if not os.path.exists('data/problems.json'):
        from create_problems import create_sample_problems
        create_sample_problems()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
