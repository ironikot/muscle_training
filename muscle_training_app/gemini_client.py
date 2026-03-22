from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import requests

from .domain import (
    build_prompt_ready_advice,
    build_prompt_ready_logs,
    json_dumps,
    normalize_log_record,
    normalize_text,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAPIError(RuntimeError):
    """Gemini API 呼び出し失敗。"""


class GeminiClient:
    def __init__(
        self,
        api_keys: tuple[str, ...],
        model_candidates: tuple[str, ...],
        timeout_seconds: int = 90,
    ) -> None:
        self._api_keys = api_keys
        self._model_candidates = model_candidates
        self._timeout_seconds = timeout_seconds
        self._preferred_key_index = 0
        self._preferred_model_index = 0

    def parse_workout_log(
        self,
        *,
        text: str,
        exercise_names: list[str],
        now: datetime,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        prompt = f"""
あなたは筋トレログ抽出アシスタントです。
以下の自然文から筋トレログを抽出し、必ず JSON のみで返してください。

現在日時: {now.isoformat(timespec="seconds")}
既存の候補種目: {", ".join(exercise_names) if exercise_names else "なし"}

出力 JSON スキーマ:
{{
  "records": [
    {{
      "date": "YYYY-MM-DD",
      "exercise": "ベンチプレス",
      "set_number": 1,
      "weight_kg": 70,
      "reps": 5,
      "rpe": 8.5,
      "rest_seconds": 180,
      "note": "最後のセット失敗"
    }}
  ],
  "new_exercises": ["新しい種目名"]
}}

ルール:
- date は相対表現を現在日時基準で解決してください。日付が明示されなければ今日を入れてください。
- 1セットごとに1レコードへ分解してください。
- set_number が明示されなければ、同一 date + exercise 内で 1,2,3... と振ってください。
- 重さは kg 数値のみ、回数は数値のみ。
- 自重種目で加重なしなら weight_kg は 0 にしてください。
- RPE と rest_seconds と note は不明なら null または空文字にしてください。
- 既存候補種目と一致するものは表記を優先してください。
- 候補にないが明確に新規種目なら new_exercises に含めてください。
- 抽出できる筋トレログがなければ records を空配列にしてください。

入力:
\"\"\"{text.strip()}\"\"\"
""".strip()
        payload = self._generate_json(
            prompt=prompt,
            temperature=0.1,
            model=model,
            thinking_level=thinking_level,
        )
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, list):
            raise GeminiAPIError("records が配列ではありません。")

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(raw_records, start=1):
            if not isinstance(row, dict):
                continue
            normalized_row = normalize_log_record(row, default_date=now.date())
            if not normalized_row["exercise"] or normalized_row["reps"] <= 0:
                continue
            if not row.get("set_number") and index:
                normalized_row["set_number"] = index
            normalized.append(normalized_row)

        new_exercises = [
            normalize_text(value)
            for value in payload.get("new_exercises", [])
            if normalize_text(value)
        ]
        return {
            "records": normalized,
            "new_exercises": new_exercises,
        }

    def generate_training_advice(
        self,
        *,
        question: str,
        now: datetime,
        today_summary: str,
        trend_summary: str,
        recent_logs: list[dict[str, Any]],
        recent_advice: list[dict[str, Any]],
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        prompt = f"""
あなたは筋トレの記録をもとに、現実的で安全な助言をするコーチです。
回答は必ず JSON のみで返してください。

現在日時: {now.isoformat(timespec="seconds")}
今日のメニュー要約:
{today_summary}

直近2週間の推移要約:
{trend_summary}

直近2週間のログ(JSON):
{json_dumps(build_prompt_ready_logs(recent_logs))}

昨日と今日の相談履歴(JSON):
{json_dumps(build_prompt_ready_advice(recent_advice))}

ユーザーの相談:
{question.strip()}

出力 JSON スキーマ:
{{
  "summary": "結論の要約",
  "advice": ["具体的な提案1", "具体的な提案2"],
  "next_workout_focus": "次回の重点",
  "caution": "注意点"
}}

ルール:
- 今日の内容、過去2週間の負荷推移、相談履歴を踏まえて答えてください。
- 重量、回数、RPE の調整案は、可能なら具体的な数値で返してください。
- 無理な断定は避け、記録不足ならその前提を短く明示してください。
- advice は 2〜4 件の短い文にしてください。
""".strip()
        payload = self._generate_json(
            prompt=prompt,
            temperature=0.3,
            model=model,
            thinking_level=thinking_level,
        )
        advice = payload.get("advice", [])
        if not isinstance(advice, list):
            advice = []
        return {
            "summary": normalize_text(payload.get("summary")),
            "advice": [normalize_text(item) for item in advice if normalize_text(item)],
            "next_workout_focus": normalize_text(payload.get("next_workout_focus")),
            "caution": normalize_text(payload.get("caution")),
            "_raw": payload,
        }

    def probe(
        self,
        *,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        payload = self._generate_json(
            prompt='{"status":"ok","message":"疎通確認"} と同じキーで JSON を返してください。',
            temperature=0.0,
            model=model,
            thinking_level=thinking_level,
        )
        return {
            "status": normalize_text(payload.get("status")),
            "message": normalize_text(payload.get("message")),
        }

    def _generate_json(
        self,
        *,
        prompt: str,
        temperature: float,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for model_index in self._candidate_model_indexes(preferred_model=model):
            model_name = self._model_candidates[model_index]
            for key_index in self._candidate_key_indexes():
                api_key = self._api_keys[key_index]
                try:
                    generation_config: dict[str, Any] = {
                        "responseMimeType": "application/json",
                        "temperature": temperature,
                    }
                    if thinking_level:
                        generation_config["thinkingConfig"] = {
                            "thinkingLevel": thinking_level,
                        }
                    response = requests.post(
                        f"{GEMINI_BASE_URL}/{model_name}:generateContent",
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": api_key,
                        },
                        json={
                            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                            "generationConfig": generation_config,
                        },
                        timeout=self._timeout_seconds,
                    )
                    if response.status_code >= 400:
                        errors.append(f"{model_name}: {self._extract_error_message(response)}")
                        continue
                    text = self._extract_text(response.json())
                    parsed = self._coerce_json(text)
                    self._preferred_model_index = model_index
                    self._preferred_key_index = key_index
                    return parsed
                except requests.RequestException as exc:
                    errors.append(f"{model_name}: {exc}")
                except (ValueError, KeyError) as exc:
                    errors.append(f"{model_name}: {exc}")
        raise GeminiAPIError(" / ".join(errors) if errors else "Gemini API 呼び出しに失敗しました。")

    def _candidate_key_indexes(self) -> list[int]:
        indexes = list(range(len(self._api_keys)))
        preferred = self._preferred_key_index
        if preferred in indexes:
            indexes.remove(preferred)
            indexes.insert(0, preferred)
        return indexes

    def _candidate_model_indexes(self, *, preferred_model: str | None = None) -> list[int]:
        indexes = list(range(len(self._model_candidates)))
        if preferred_model and preferred_model in self._model_candidates:
            preferred = self._model_candidates.index(preferred_model)
        else:
            preferred = self._preferred_model_index
        if preferred in indexes:
            indexes.remove(preferred)
            indexes.insert(0, preferred)
        return indexes

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return normalize_text(error.get("message")) or response.text[:200]
        return response.text[:200]

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini 応答に candidates がありません。")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise ValueError("Gemini 応答に本文がありません。")
        return text

    @staticmethod
    def _coerce_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("JSON オブジェクトではありません。")
        return parsed
