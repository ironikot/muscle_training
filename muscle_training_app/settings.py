from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .domain import (
    APP_ROOT,
    DEFAULT_MODEL_CANDIDATES,
    DEFAULT_SPREADSHEET_URL,
    MODEL_THINKING_LEVELS,
)

ENV_PATH: Final = APP_ROOT / ".env"
DEFAULT_SERVICE_ACCOUNT_PATH: Final = (
    APP_ROOT / "endless-recorder-477707-d48e664f0b26.json"
)


@dataclass(frozen=True)
class AppSettings:
    spreadsheet_url: str
    service_account_json_path: Path | None
    service_account_info: dict[str, object] | None
    api_keys: tuple[str, ...]
    timezone: ZoneInfo
    model_candidates: tuple[str, ...]
    model_thinking_levels: dict[str, tuple[str, ...]]


def _collect_api_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for key_name in ("GEMINI_AI_STUDIO_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(key_name, "").strip()
        if value and value not in keys:
            keys.append(value)
    return tuple(keys)


def _load_model_candidates() -> tuple[str, ...]:
    raw = os.getenv("GEMINI_MODEL_CANDIDATES", "").strip()
    if not raw:
        return DEFAULT_MODEL_CANDIDATES
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or DEFAULT_MODEL_CANDIDATES


def _build_model_thinking_levels(
    model_candidates: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for model in model_candidates:
        levels = MODEL_THINKING_LEVELS.get(model)
        if levels:
            result[model] = tuple(levels)
    return result


def load_settings() -> AppSettings:
    load_dotenv(ENV_PATH)
    spreadsheet_url = os.getenv("SPREADSHEET_URL", DEFAULT_SPREADSHEET_URL).strip()
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    service_account_info: dict[str, object] | None = None
    service_account_json_path: Path | None = None

    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON は JSON オブジェクトである必要があります。")
        service_account_info = parsed
    else:
        candidate = Path(
            os.getenv("GOOGLE_SHEETS_SERVICE_ACCOUNT_PATH", str(DEFAULT_SERVICE_ACCOUNT_PATH))
        )
        if not candidate.exists():
            raise RuntimeError(
                "Google Sheets 認証情報が見つかりません。"
                "GOOGLE_SERVICE_ACCOUNT_JSON または GOOGLE_SHEETS_SERVICE_ACCOUNT_PATH を設定してください。"
            )
        service_account_json_path = candidate

    api_keys = _collect_api_keys()
    if not api_keys:
        raise RuntimeError(
            "GEMINI_AI_STUDIO_API_KEY または GOOGLE_API_KEY を設定してください。"
        )

    model_candidates = _load_model_candidates()

    return AppSettings(
        spreadsheet_url=spreadsheet_url,
        service_account_json_path=service_account_json_path,
        service_account_info=service_account_info,
        api_keys=api_keys,
        timezone=ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Tokyo")),
        model_candidates=model_candidates,
        model_thinking_levels=_build_model_thinking_levels(model_candidates),
    )
