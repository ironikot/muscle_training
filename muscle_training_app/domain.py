from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

APP_ROOT: Final = Path(__file__).resolve().parent.parent

DEFAULT_SPREADSHEET_URL: Final = (
    "https://docs.google.com/spreadsheets/d/"
    "1phGdYE9UmB-jSDBPO88iyiXQAfMwh8OOgMj2dQvXgHQ/edit?gid=0#gid=0"
)
DEFAULT_MODEL_CANDIDATES: Final = (
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
)
MODEL_THINKING_LEVELS: Final = {
    "gemini-3-flash-preview": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro-preview": ["low", "medium", "high"],
}
DEFAULT_EXERCISES: Final = [
    "ベンチプレス",
    "加重懸垂",
    "懸垂",
    "スミスマシン",
    "レッグマシン",
]

LOG_SHEET_NAME: Final = "training_logs"
LOG_HEADERS: Final = [
    "日付",
    "種目",
    "セット番号",
    "重さ_kg",
    "回数",
    "RPE",
    "休憩秒",
    "ボリューム_kg",
    "ノート",
    "作成日時",
]
EXERCISE_SHEET_NAME: Final = "exercise_master"
EXERCISE_HEADERS: Final = ["種目名", "有効", "並び順", "作成日時"]
ADVICE_SHEET_NAME: Final = "advice_history"
ADVICE_HEADERS: Final = [
    "実行日時",
    "相談内容",
    "回答",
    "回答JSON",
    "今日のメニュー",
    "参照開始日",
    "参照終了日",
    "参照ログ件数",
    "モデル",
]
PROFILE_SHEET_NAME: Final = "advice_profile"
PROFILE_HEADERS: Final = ["key", "value", "updated_at"]
DEFAULT_GOAL_TEXT: Final = (
    "片手懸垂とマッスルアップができるようになりたくて、"
    "ベンチプレスは100kgを目指したい、体脂肪率は17％台に落としたい、"
    "一日にできる筋トレ時間は25分ほどです。"
)
DEFAULT_NOTE_TEXT: Final = (
    "筋トレのログの斜度ウォークは、ウォーキングマシンで15度の斜度をつけて歩くものです。"
)

RPE_OPTIONS: Final = [None, *[index / 2 for index in range(0, 21)]]
RPE_DESCRIPTIONS: Final = {
    6.0: "あと4回以上余裕",
    7.0: "あと3回はいけそう",
    8.0: "あと2回はいけそう",
    9.0: "あと1回ギリギリ",
    10.0: "完全限界",
}


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    return None


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_optional_int(value: Any) -> int | None:
    number = normalize_optional_float(value)
    return int(number) if number is not None else None


def normalize_log_record(record: dict[str, Any], default_date: date | None = None) -> dict[str, Any]:
    base_date = default_date or date.today()
    parsed = parse_date(record.get("date") or record.get("日付")) or base_date
    exercise = normalize_text(record.get("exercise") or record.get("種目"))
    note = normalize_text(record.get("note") or record.get("ノート"))
    set_number = int(float(record.get("set_number") or record.get("セット番号") or 1))
    weight_kg = float(record.get("weight_kg") or record.get("重さ_kg") or 0)
    reps = int(float(record.get("reps") or record.get("回数") or 0))
    rpe = normalize_optional_float(record.get("rpe") or record.get("RPE"))
    rest_seconds = normalize_optional_int(record.get("rest_seconds") or record.get("休憩秒"))
    return {
        "date": parsed.isoformat(),
        "exercise": exercise,
        "set_number": set_number,
        "weight_kg": weight_kg,
        "reps": reps,
        "rpe": rpe,
        "rest_seconds": rest_seconds,
        "note": note,
    }


def format_number(value: Any) -> str:
    if value in (None, ""):
        return ""
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def format_rpe(value: float | None) -> str:
    if value is None:
        return "未入力"
    base = format_number(value)
    description = RPE_DESCRIPTIONS.get(float(int(value)) if float(value).is_integer() else value)
    return f"{base} | {description}" if description else base


def next_set_number(logs: list[dict[str, Any]], log_date: date, exercise: str) -> int:
    target_date = log_date.isoformat()
    candidates = [
        int(row.get("セット番号", 0) or 0)
        for row in logs
        if row.get("日付") == target_date and row.get("種目") == exercise
    ]
    return (max(candidates) if candidates else 0) + 1


def summarize_today_logs(logs: list[dict[str, Any]], target_date: date) -> str:
    target_key = target_date.isoformat()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logs:
        if row.get("日付") == target_key:
            grouped[normalize_text(row.get("種目"))].append(row)
    if not grouped:
        return "今日はまだログがありません。"

    lines: list[str] = []
    for exercise in sorted(grouped):
        rows = sorted(grouped[exercise], key=lambda row: int(row.get("セット番号", 0) or 0))
        total_volume = sum(float(row.get("ボリューム_kg", 0) or 0) for row in rows)
        set_text = ", ".join(
            f"{format_number(row.get('重さ_kg'))}kg x {int(float(row.get('回数') or 0))}"
            for row in rows
        )
        lines.append(
            f"{exercise}: {len(rows)}セット, {set_text}, 総ボリューム {format_number(total_volume)}kg"
        )
    return "\n".join(lines)


def summarize_recent_progress(logs: list[dict[str, Any]], *, lookback_days: int = 14) -> str:
    if not logs:
        return "直近のログはありません。"

    latest_date = max((parse_date(row.get("日付")) for row in logs), default=None)
    if latest_date is None:
        return "直近のログはありません。"
    start_date = latest_date - timedelta(days=max(lookback_days - 1, 0))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logs:
        row_date = parse_date(row.get("日付"))
        if row_date and row_date >= start_date:
            grouped[normalize_text(row.get("種目"))].append(row)

    if not grouped:
        return "直近のログはありません。"

    lines: list[str] = []
    for exercise in sorted(grouped):
        rows = grouped[exercise]
        session_dates = sorted({row["日付"] for row in rows})
        top_weight = max(float(row.get("重さ_kg", 0) or 0) for row in rows)
        total_sets = len(rows)
        total_volume = sum(float(row.get("ボリューム_kg", 0) or 0) for row in rows)
        latest_rows = [row for row in rows if row.get("日付") == session_dates[-1]]
        latest_top = max(
            latest_rows,
            key=lambda row: (float(row.get("重さ_kg", 0) or 0), float(row.get("回数", 0) or 0)),
        )
        lines.append(
            (
                f"{exercise}: {len(session_dates)}日, {total_sets}セット, "
                f"最大重量 {format_number(top_weight)}kg, 総ボリューム {format_number(total_volume)}kg, "
                f"最新 {session_dates[-1]} {format_number(latest_top.get('重さ_kg'))}kg x "
                f"{int(float(latest_top.get('回数') or 0))}"
            )
        )
    return "\n".join(lines)


def build_session_volume_rows(logs: list[dict[str, Any]], exercise: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in logs:
        if row.get("種目") == exercise:
            grouped[row["日付"]].append(row)

    result: list[dict[str, Any]] = []
    for log_date in sorted(grouped):
        rows = grouped[log_date]
        result.append(
            {
                "日付": log_date,
                "総ボリューム_kg": sum(float(row.get("ボリューム_kg", 0) or 0) for row in rows),
                "最大重量_kg": max(float(row.get("重さ_kg", 0) or 0) for row in rows),
                "総セット数": len(rows),
            }
        )
    return result


def build_prompt_ready_logs(logs: list[dict[str, Any]], *, limit: int = 200) -> list[dict[str, Any]]:
    ordered = sorted(
        logs,
        key=lambda row: (
            row.get("日付", ""),
            normalize_text(row.get("種目")),
            int(row.get("セット番号", 0) or 0),
        ),
        reverse=True,
    )
    compact: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        compact.append(
            {
                "date": row.get("日付"),
                "exercise": row.get("種目"),
                "set_number": int(row.get("セット番号", 0) or 0),
                "weight_kg": normalize_optional_float(row.get("重さ_kg")),
                "reps": normalize_optional_int(row.get("回数")),
                "rpe": normalize_optional_float(row.get("RPE")),
                "rest_seconds": normalize_optional_int(row.get("休憩秒")),
                "volume_kg": normalize_optional_float(row.get("ボリューム_kg")),
                "note": normalize_text(row.get("ノート")),
            }
        )
    return compact


def build_prompt_ready_advice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "executed_at": row.get("実行日時"),
            "question": row.get("相談内容"),
            "answer": row.get("回答"),
        }
        for row in rows
    ]
