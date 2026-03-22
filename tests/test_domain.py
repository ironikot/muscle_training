from __future__ import annotations

from datetime import date

from muscle_training_app.domain import (
    build_session_volume_rows,
    next_set_number,
    normalize_log_record,
    summarize_recent_progress,
    summarize_today_logs,
)


def test_normalize_log_record() -> None:
    row = normalize_log_record(
        {
            "date": "2026-03-23",
            "exercise": "ベンチプレス",
            "set_number": "2",
            "weight_kg": "70",
            "reps": "5",
            "rpe": "8.5",
            "rest_seconds": "180",
            "note": "最後のセット失敗",
        }
    )
    assert row["date"] == "2026-03-23"
    assert row["exercise"] == "ベンチプレス"
    assert row["set_number"] == 2
    assert row["weight_kg"] == 70.0
    assert row["reps"] == 5
    assert row["rpe"] == 8.5
    assert row["rest_seconds"] == 180


def test_next_set_number() -> None:
    logs = [
        {"日付": "2026-03-23", "種目": "ベンチプレス", "セット番号": 1},
        {"日付": "2026-03-23", "種目": "ベンチプレス", "セット番号": 2},
    ]
    assert next_set_number(logs, date(2026, 3, 23), "ベンチプレス") == 3


def test_summarize_today_logs() -> None:
    logs = [
        {
            "日付": "2026-03-23",
            "種目": "ベンチプレス",
            "セット番号": 1,
            "重さ_kg": 70,
            "回数": 5,
            "ボリューム_kg": 350,
        },
        {
            "日付": "2026-03-23",
            "種目": "ベンチプレス",
            "セット番号": 2,
            "重さ_kg": 70,
            "回数": 5,
            "ボリューム_kg": 350,
        },
    ]
    summary = summarize_today_logs(logs, date(2026, 3, 23))
    assert "ベンチプレス" in summary
    assert "総ボリューム 700kg" in summary


def test_summarize_recent_progress() -> None:
    logs = [
        {
            "日付": "2026-03-20",
            "種目": "ベンチプレス",
            "セット番号": 1,
            "重さ_kg": 67.5,
            "回数": 5,
            "ボリューム_kg": 337.5,
        },
        {
            "日付": "2026-03-23",
            "種目": "ベンチプレス",
            "セット番号": 1,
            "重さ_kg": 70,
            "回数": 5,
            "ボリューム_kg": 350,
        },
    ]
    summary = summarize_recent_progress(logs)
    assert "最大重量 70kg" in summary
    assert "最新 2026-03-23 70kg x 5" in summary


def test_build_session_volume_rows() -> None:
    logs = [
        {
            "日付": "2026-03-23",
            "種目": "ベンチプレス",
            "セット番号": 1,
            "重さ_kg": 70,
            "回数": 5,
            "ボリューム_kg": 350,
        },
        {
            "日付": "2026-03-23",
            "種目": "ベンチプレス",
            "セット番号": 2,
            "重さ_kg": 72.5,
            "回数": 3,
            "ボリューム_kg": 217.5,
        },
    ]
    rows = build_session_volume_rows(logs, "ベンチプレス")
    assert rows[0]["総ボリューム_kg"] == 567.5
    assert rows[0]["最大重量_kg"] == 72.5
