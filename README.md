# 筋トレログアプリ

`Streamlit` で動く筋トレログ管理アプリです。Google スプレッドシートにログを保存し、自然言語入力の解析とトレーニング助言を `Gemini` で行います。

## 機能

- 手入力フォームで筋トレログを追加
- 自然言語入力を LLM で構造化し、そのまま保存
- `training_logs / exercise_master / advice_history` の 3 シートを自動作成
- 種目マスターをアプリ上で追加・更新
- 直近 2 週間のログと昨日・今日の相談履歴を使った AI アドバイス
- 追加・更新・削除の CRUD

## 起動

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

`.streamlit/secrets.toml` は [`.streamlit/secrets.toml.example`](/Users/tokinorishimodaira/Desktop/outof-work/muscle_training_app/.streamlit/secrets.toml.example) と同じ構造にしてください。

## 疎通確認

```bash
python scripts/integration_check.py --secrets .streamlit/secrets.toml --write-test
```

# muscle_training
