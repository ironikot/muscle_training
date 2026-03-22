from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import gspread
from google.oauth2.service_account import Credentials

from .domain import (
    ADVICE_HEADERS,
    ADVICE_SHEET_NAME,
    DEFAULT_EXERCISES,
    EXERCISE_HEADERS,
    EXERCISE_SHEET_NAME,
    LOG_HEADERS,
    LOG_SHEET_NAME,
    format_number,
    normalize_log_record,
    normalize_optional_float,
    normalize_optional_int,
    normalize_text,
    parse_date,
)

GOOGLE_SCOPES: Final = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
)


@dataclass(frozen=True)
class SpreadsheetMeta:
    title: str
    worksheet_titles: list[str]


class GoogleSheetsRepository:
    def __init__(
        self,
        *,
        spreadsheet_url: str,
        service_account_json_path: str | None = None,
        service_account_info: dict[str, Any] | None = None,
    ) -> None:
        if service_account_info is not None:
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=GOOGLE_SCOPES,
            )
        elif service_account_json_path is not None:
            credentials = Credentials.from_service_account_file(
                service_account_json_path,
                scopes=GOOGLE_SCOPES,
            )
        else:
            raise ValueError("Google 認証情報が必要です。")
        self._client = gspread.authorize(credentials)
        self._spreadsheet = self._client.open_by_url(spreadsheet_url)

    def ensure_schema(self) -> None:
        self._ensure_headers(LOG_SHEET_NAME, LOG_HEADERS, cols=12)
        self._ensure_headers(EXERCISE_SHEET_NAME, EXERCISE_HEADERS, cols=6)
        self._ensure_headers(ADVICE_SHEET_NAME, ADVICE_HEADERS, cols=12)
        self._seed_default_exercises()

    def get_meta(self) -> SpreadsheetMeta:
        return SpreadsheetMeta(
            title=self._spreadsheet.title,
            worksheet_titles=[worksheet.title for worksheet in self._spreadsheet.worksheets()],
        )

    def load_logs(self, *, days: int | None = None) -> list[dict[str, Any]]:
        worksheet = self._worksheet(LOG_SHEET_NAME)
        rows = worksheet.get_all_values()
        if not rows:
            return []

        data_rows = rows[1:]
        normalized: list[dict[str, Any]] = []
        for row_number, values in enumerate(data_rows, start=2):
            row = {header: values[index] if index < len(values) else "" for index, header in enumerate(LOG_HEADERS)}
            log_date = parse_date(row.get("日付"))
            if log_date is None:
                continue
            normalized.append(
                {
                    "_row_number": row_number,
                    "日付": log_date.isoformat(),
                    "種目": normalize_text(row.get("種目")),
                    "セット番号": normalize_optional_int(row.get("セット番号")) or 0,
                    "重さ_kg": normalize_optional_float(row.get("重さ_kg")) or 0,
                    "回数": normalize_optional_int(row.get("回数")) or 0,
                    "RPE": normalize_optional_float(row.get("RPE")),
                    "休憩秒": normalize_optional_int(row.get("休憩秒")),
                    "ボリューム_kg": normalize_optional_float(row.get("ボリューム_kg")) or 0,
                    "ノート": normalize_text(row.get("ノート")),
                    "作成日時": normalize_text(row.get("作成日時")),
                }
            )

        normalized.sort(
            key=lambda item: (item["日付"], item["種目"], int(item["セット番号"] or 0)),
            reverse=True,
        )
        if days is None or not normalized:
            return normalized

        latest = max(parse_date(row["日付"]) for row in normalized if parse_date(row["日付"]))
        if latest is None:
            return normalized
        cutoff = latest.toordinal() - max(days - 1, 0)
        return [
            row
            for row in normalized
            if parse_date(row["日付"]) and parse_date(row["日付"]).toordinal() >= cutoff
        ]

    def append_log_rows(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        worksheet = self._worksheet(LOG_SHEET_NAME)
        start_row = len(worksheet.get_all_values()) + 1
        payload: list[list[Any]] = []
        for offset, record in enumerate(records):
            payload.append(self._serialize_log_row(record, row_number=start_row + offset))
        worksheet.append_rows(payload, value_input_option="USER_ENTERED")

    def update_log_row(self, row_number: int, record: dict[str, Any], *, created_at: str = "") -> None:
        worksheet = self._worksheet(LOG_SHEET_NAME)
        worksheet.update(
            f"A{row_number}:J{row_number}",
            [self._serialize_log_row(record, row_number=row_number, created_at=created_at or "=NOW()")],
            value_input_option="USER_ENTERED",
        )

    def delete_log_row(self, row_number: int) -> None:
        worksheet = self._worksheet(LOG_SHEET_NAME)
        worksheet.delete_rows(row_number)

    def load_exercise_records(self) -> list[dict[str, Any]]:
        worksheet = self._worksheet(EXERCISE_SHEET_NAME)
        rows = worksheet.get_all_values()
        if not rows:
            return []

        result: list[dict[str, Any]] = []
        for row_number, values in enumerate(rows[1:], start=2):
            if not any(values):
                continue
            row = {
                header: values[index] if index < len(values) else ""
                for index, header in enumerate(EXERCISE_HEADERS)
            }
            result.append(
                {
                    "_row_number": row_number,
                    "種目名": normalize_text(row.get("種目名")),
                    "有効": str(row.get("有効", "TRUE")).strip().upper() != "FALSE",
                    "並び順": normalize_optional_int(row.get("並び順")) or row_number - 1,
                    "作成日時": normalize_text(row.get("作成日時")),
                }
            )
        result.sort(key=lambda item: (int(item["並び順"]), item["種目名"]))
        return result

    def list_exercises(self, *, active_only: bool = True) -> list[str]:
        rows = self.load_exercise_records()
        names = [row["種目名"] for row in rows if row["種目名"] and (row["有効"] or not active_only)]
        return names

    def add_exercise(self, name: str) -> None:
        normalized_name = normalize_text(name)
        if not normalized_name:
            return
        existing = {value.casefold() for value in self.list_exercises(active_only=False)}
        if normalized_name.casefold() in existing:
            return
        worksheet = self._worksheet(EXERCISE_SHEET_NAME)
        rows = self.load_exercise_records()
        next_order = (max((int(row["並び順"]) for row in rows), default=0) + 1) if rows else 1
        worksheet.append_row(
            [normalized_name, "TRUE", next_order, "=NOW()"],
            value_input_option="USER_ENTERED",
        )

    def save_exercise_records(self, records: list[dict[str, Any]]) -> None:
        worksheet = self._worksheet(EXERCISE_SHEET_NAME)
        ordered = sorted(
            (
                {
                    "種目名": normalize_text(row.get("種目名")),
                    "有効": bool(row.get("有効", True)),
                    "並び順": int(float(row.get("並び順") or 0)),
                    "作成日時": normalize_text(row.get("作成日時")) or "=NOW()",
                }
                for row in records
                if normalize_text(row.get("種目名"))
            ),
            key=lambda item: (item["並び順"], item["種目名"]),
        )
        values = [EXERCISE_HEADERS] + [
            [row["種目名"], "TRUE" if row["有効"] else "FALSE", row["並び順"], row["作成日時"]]
            for row in ordered
        ]
        worksheet.clear()
        worksheet.update(values, value_input_option="USER_ENTERED")

    def load_advice_history(self, *, days: int = 2) -> list[dict[str, Any]]:
        worksheet = self._worksheet(ADVICE_SHEET_NAME)
        rows = worksheet.get_all_values()
        if not rows:
            return []

        normalized: list[dict[str, Any]] = []
        for values in rows[1:]:
            row = {
                header: values[index] if index < len(values) else ""
                for index, header in enumerate(ADVICE_HEADERS)
            }
            executed_date = parse_date(normalize_text(row.get("実行日時"))[:10])
            if executed_date is None:
                continue
            normalized.append(row)

        normalized.sort(key=lambda row: normalize_text(row.get("実行日時")), reverse=True)
        if not normalized:
            return []
        latest = parse_date(normalized[0]["実行日時"][:10])
        if latest is None:
            return normalized
        cutoff = latest.toordinal() - max(days - 1, 0)
        return [
            row
            for row in normalized
            if parse_date(normalize_text(row.get("実行日時"))[:10])
            and parse_date(normalize_text(row.get("実行日時"))[:10]).toordinal() >= cutoff
        ]

    def append_advice_history(self, record: dict[str, Any]) -> None:
        worksheet = self._worksheet(ADVICE_SHEET_NAME)
        worksheet.append_row(
            [record.get(header, "") for header in ADVICE_HEADERS],
            value_input_option="USER_ENTERED",
        )

    def _serialize_log_row(
        self,
        record: dict[str, Any],
        *,
        row_number: int,
        created_at: str = "=NOW()",
    ) -> list[Any]:
        normalized = normalize_log_record(record)
        return [
            normalized["date"],
            normalized["exercise"],
            normalized["set_number"],
            float(normalized["weight_kg"]),
            int(normalized["reps"]),
            "" if normalized["rpe"] is None else float(normalized["rpe"]),
            "" if normalized["rest_seconds"] is None else int(normalized["rest_seconds"]),
            f"=D{row_number}*E{row_number}",
            normalized["note"],
            created_at,
        ]

    def _seed_default_exercises(self) -> None:
        existing = {name.casefold() for name in self.list_exercises(active_only=False)}
        if existing:
            missing = [name for name in DEFAULT_EXERCISES if name.casefold() not in existing]
        else:
            missing = DEFAULT_EXERCISES
        for name in missing:
            self.add_exercise(name)

    def _worksheet(self, sheet_name: str):
        worksheet = self._try_get_worksheet(sheet_name)
        if worksheet is not None:
            return worksheet
        return self._spreadsheet.add_worksheet(title=sheet_name, rows=300, cols=16)

    def _try_get_worksheet(self, sheet_name: str):
        try:
            return self._spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            return None

    def _ensure_headers(self, sheet_name: str, headers: list[str], *, cols: int) -> None:
        worksheet = self._worksheet(sheet_name)
        existing = worksheet.row_values(1)
        if not existing:
            worksheet.update([headers], value_input_option="USER_ENTERED")
            worksheet.freeze(rows=1)
            return
        if existing[: len(headers)] != headers:
            worksheet.update("A1", [headers], value_input_option="USER_ENTERED")
        worksheet.freeze(rows=1)
