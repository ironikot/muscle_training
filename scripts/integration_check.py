from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muscle_training_app.domain import json_dumps, summarize_recent_progress, summarize_today_logs
from muscle_training_app.gemini_client import GeminiClient
from muscle_training_app.settings import load_settings
from muscle_training_app.sheets_repo import GoogleSheetsRepository


def load_secrets(path: Path) -> None:
    if not path.exists():
        return
    data = tomllib.loads(path.read_text())
    for key in (
        "GOOGLE_API_KEY",
        "GEMINI_AI_STUDIO_API_KEY",
        "SPREADSHEET_URL",
        "APP_TIMEZONE",
        "GEMINI_MODEL_CANDIDATES",
    ):
        if key in data and not os.environ.get(key):
            os.environ[key] = str(data[key])
    service_account = data.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if isinstance(service_account, dict) and not os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"):
        os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(service_account, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--secrets",
        default=".streamlit/secrets.toml",
        help="Streamlit secrets.toml のパス",
    )
    parser.add_argument(
        "--write-test",
        action="store_true",
        help="一時ログを書いて削除する CRUD テストも実行する",
    )
    args = parser.parse_args()

    load_secrets(Path(args.secrets))
    settings = load_settings()
    repository = GoogleSheetsRepository(
        spreadsheet_url=settings.spreadsheet_url,
        service_account_json_path=(
            str(settings.service_account_json_path)
            if settings.service_account_json_path
            else None
        ),
        service_account_info=settings.service_account_info,
    )
    repository.ensure_schema()
    meta = repository.get_meta()
    print("Spreadsheet:", meta.title)
    print("Worksheets:", ", ".join(meta.worksheet_titles))
    print("Exercises:", ", ".join(repository.list_exercises(active_only=False)))

    if args.write_test:
        today = datetime.now(settings.timezone).date().isoformat()
        before = repository.load_logs(days=14)
        repository.append_log_rows(
            [
                {
                    "date": today,
                    "exercise": "ベンチプレス",
                    "set_number": 99,
                    "weight_kg": 40,
                    "reps": 1,
                    "rpe": 6,
                    "rest_seconds": 60,
                    "note": "integration_check_temp",
                }
            ]
        )
        after_append = repository.load_logs(days=14)
        temp_row = next(
            row
            for row in after_append
            if row["セット番号"] == 99 and row["ノート"] == "integration_check_temp"
        )
        repository.delete_log_row(temp_row["_row_number"])
        after_delete = repository.load_logs(days=14)
        print(
            "CRUD write test:",
            f"before={len(before)} appended={len(after_append)} after_delete={len(after_delete)}",
        )

    client = GeminiClient(
        api_keys=settings.api_keys,
        model_candidates=settings.model_candidates,
    )
    model = settings.model_candidates[0]
    thinking_level = settings.model_thinking_levels[model][0]
    print("Gemini probe:", json_dumps(client.probe(model=model, thinking_level=thinking_level)))
    parsed = client.parse_workout_log(
        text="今日ベンチプレス70kg5回を2セット、RPE8。懸垂10回を2セット。",
        exercise_names=repository.list_exercises(active_only=False),
        now=datetime.now(settings.timezone),
        model=model,
        thinking_level=thinking_level,
    )
    print("Parse sample:", json_dumps(parsed))
    recent_logs = repository.load_logs(days=14)
    advice = client.generate_training_advice(
        question="今日の内容を踏まえて、次回のベンチプレス重量設定を提案して。",
        now=datetime.now(settings.timezone),
        today_summary=summarize_today_logs(recent_logs, datetime.now(settings.timezone).date()),
        trend_summary=summarize_recent_progress(recent_logs, lookback_days=14),
        recent_logs=recent_logs,
        recent_advice=repository.load_advice_history(days=2),
        model=model,
        thinking_level=thinking_level,
    )
    print("Advice sample:", json_dumps(advice))


if __name__ == "__main__":
    main()
