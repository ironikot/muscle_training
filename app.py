from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth


def _streamlit_secrets_into_environ() -> None:
    try:
        secrets = st.secrets
    except (RuntimeError, FileNotFoundError, TypeError):
        return

    service_account_secret = secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if service_account_secret and not os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"):
        if hasattr(service_account_secret, "to_dict"):
            os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(
                service_account_secret.to_dict(),
                ensure_ascii=False,
            )
        elif isinstance(service_account_secret, dict):
            os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = json.dumps(
                service_account_secret,
                ensure_ascii=False,
            )
        else:
            os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = str(service_account_secret)

    for key in (
        "GEMINI_AI_STUDIO_API_KEY",
        "GOOGLE_API_KEY",
        "SPREADSHEET_URL",
        "APP_TIMEZONE",
        "GEMINI_MODEL_CANDIDATES",
    ):
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])


_streamlit_secrets_into_environ()

from muscle_training_app.domain import (
    DEFAULT_GOAL_TEXT,
    DEFAULT_NOTE_TEXT,
    LOG_HEADERS,
    MODEL_THINKING_LEVELS,
    RPE_OPTIONS,
    build_session_volume_rows,
    format_number,
    format_rpe,
    json_dumps,
    next_set_number,
    normalize_log_record,
    normalize_text,
    parse_date,
    summarize_recent_progress,
    summarize_today_logs,
)
from muscle_training_app.gemini_client import GeminiAPIError, GeminiClient
from muscle_training_app.settings import AppSettings, load_settings
from muscle_training_app.sheets_repo import GoogleSheetsRepository

st.set_page_config(page_title="筋トレログアプリ", page_icon="🏋️", layout="wide")

REPOSITORY_CACHE_VERSION = "profile-v1"
GEMINI_CLIENT_CACHE_VERSION = "advice-profile-v1"


@st.cache_resource(show_spinner=False)
def get_settings() -> AppSettings:
    return load_settings()


@st.cache_resource(show_spinner=False)
def get_repository(cache_version: str = REPOSITORY_CACHE_VERSION) -> GoogleSheetsRepository:
    del cache_version
    settings = get_settings()
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
    return repository


def load_profile_settings() -> dict[str, str]:
    profile = repository_read(
        lambda repository: repository.load_profile()
        if hasattr(repository, "load_profile")
        else {
            "goal": DEFAULT_GOAL_TEXT,
            "note": DEFAULT_NOTE_TEXT,
        },
        fallback={
            "goal": DEFAULT_GOAL_TEXT,
            "note": DEFAULT_NOTE_TEXT,
        },
    )
    return profile or {
        "goal": DEFAULT_GOAL_TEXT,
        "note": DEFAULT_NOTE_TEXT,
    }


def save_profile_settings(*, goal: str, note: str) -> bool:
    repository = get_repository()
    if hasattr(repository, "save_profile"):
        try:
            repository.save_profile(goal=goal, note=note)
            return True
        except Exception:
            reset_repository_connection()
            repository = get_repository()
            if hasattr(repository, "save_profile"):
                repository.save_profile(goal=goal, note=note)
                return True
    st.warning("プロファイル保存機能の初期化が未完了です。アプリを再読み込みしてください。")
    return False


def clear_advice_question() -> None:
    st.session_state["advice_question"] = ""


def mark_planned_minutes_updated() -> None:
    st.session_state["advice_planned_minutes_set_at"] = current_time().isoformat(timespec="seconds")


def reset_repository_connection() -> None:
    get_repository.clear()


def reset_gemini_client() -> None:
    get_gemini_client.clear()


def repository_read(operation, *, fallback):
    try:
        return operation(get_repository())
    except Exception:
        reset_repository_connection()
        try:
            return operation(get_repository())
        except Exception:
            return fallback


def repository_write(operation) -> bool:
    try:
        operation(get_repository())
        return True
    except Exception:
        reset_repository_connection()
        try:
            operation(get_repository())
            return True
        except Exception:
            return False


@st.cache_resource(show_spinner=False)
def get_gemini_client(cache_version: str = GEMINI_CLIENT_CACHE_VERSION) -> GeminiClient:
    del cache_version
    settings = get_settings()
    return GeminiClient(
        api_keys=settings.api_keys,
        model_candidates=settings.model_candidates,
    )


def current_time() -> datetime:
    return datetime.now(get_settings().timezone)


def available_exercises() -> list[str]:
    return repository_read(
        lambda repository: repository.list_exercises(active_only=True),
        fallback=[],
    )


def logs_to_dataframe(logs: list[dict[str, Any]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(columns=LOG_HEADERS)
    frame = pd.DataFrame(logs)
    columns = ["日付", "種目", "セット番号", "重さ_kg", "回数", "RPE", "休憩秒", "ボリューム_kg", "ノート", "作成日時"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def editable_logs_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        normalized = normalize_log_record(row)
        if not normalized["exercise"] or normalized["weight_kg"] < 0 or normalized["reps"] <= 0:
            continue
        records.append(normalized)
    return records


def ensure_session_state() -> None:
    st.session_state.setdefault("app_opened_at", current_time().isoformat(timespec="seconds"))
    st.session_state.setdefault("parsed_log_records", [])
    st.session_state.setdefault("parsed_new_exercises", [])
    st.session_state.setdefault("last_advice", None)
    st.session_state.setdefault("advice_question", "")
    st.session_state.setdefault("manual_date", current_time().date())
    st.session_state.setdefault("manual_exercise", "")
    st.session_state.setdefault("manual_set_number", 1)
    st.session_state.setdefault("manual_weight_kg", 40.0)
    st.session_state.setdefault("manual_reps", 5)
    st.session_state.setdefault("manual_rpe", None)
    st.session_state.setdefault("manual_rest_seconds", 180)
    st.session_state.setdefault("manual_note", "")
    st.session_state.setdefault("manual_prefill_marker", "")
    st.session_state.setdefault("advice_planned_minutes", 25)
    st.session_state.setdefault(
        "advice_planned_minutes_set_at",
        current_time().isoformat(timespec="seconds"),
    )


def hydrate_manual_entry_defaults(logs: list[dict[str, Any]]) -> None:
    if not logs:
        return
    today_key = current_time().date().isoformat()
    today_logs = [row for row in logs if row.get("日付") == today_key]
    latest_today_log = (
        max(today_logs, key=lambda row: int(row.get("_row_number", 0) or 0))
        if today_logs
        else None
    )
    marker = f"{today_key}:{latest_today_log.get('_row_number') if latest_today_log else 'none'}"
    if st.session_state.get("manual_prefill_marker") == marker:
        return
    st.session_state["manual_date"] = current_time().date()
    if latest_today_log:
        st.session_state["manual_exercise"] = latest_today_log.get("種目") or st.session_state.get("manual_exercise") or ""
        st.session_state["manual_set_number"] = int(latest_today_log.get("セット番号", 0) or 0) + 1
        st.session_state["manual_weight_kg"] = float(latest_today_log.get("重さ_kg", 40.0) or 40.0)
        st.session_state["manual_reps"] = int(latest_today_log.get("回数", 5) or 5)
        st.session_state["manual_rpe"] = latest_today_log.get("RPE")
        st.session_state["manual_rest_seconds"] = int(latest_today_log.get("休憩秒", 180) or 180)
        st.session_state["manual_note"] = latest_today_log.get("ノート", "") or ""
    else:
        st.session_state["manual_set_number"] = 1
    st.session_state["manual_prefill_marker"] = marker


def _secret_section_to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return {key: _secret_section_to_dict(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {key: _secret_section_to_dict(item) for key, item in value.items()}
    return value


def get_authenticator() -> stauth.Authenticate:
    try:
        auth_config = _secret_section_to_dict(st.secrets["auth"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("`auth` 設定が secrets.toml にありません。") from exc

    credentials = auth_config.get("credentials")
    cookie = auth_config.get("cookie", {})
    if not credentials:
        raise RuntimeError("`auth.credentials` が secrets.toml にありません。")
    return stauth.Authenticate(
        credentials,
        cookie.get("name", "muscle_training_cookie"),
        cookie.get("key", "change-this-cookie-key"),
        float(cookie.get("expiry_days", 30)),
        auto_hash=False,
    )


def require_login() -> None:
    authenticator = get_authenticator()
    authenticator.login(
        location="main",
        key="muscle_training_login",
        fields={
            "Form name": "ログイン",
            "Username": "ID",
            "Password": "パスワード",
            "Login": "ログイン",
        },
    )
    name = st.session_state.get("name")
    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if authentication_status:
        st.sidebar.caption("ログイン中")
        st.sidebar.write(f"{name} ({username})")
        authenticator.logout("ログアウト", "sidebar", key="muscle_training_logout")
        return
    if authentication_status is False:
        st.error("ID またはパスワードが違います。")
    else:
        st.info("ログインしてください。")
    st.stop()


def render_sidebar() -> tuple[str, str]:
    settings = get_settings()
    st.sidebar.caption("接続先")
    meta = repository_read(lambda repository: repository.get_meta(), fallback=None)
    if meta is not None:
        st.sidebar.write(meta.title)
    else:
        st.sidebar.warning("スプレッドシート接続を確認できません。権限または secrets を確認してください。")
    st.sidebar.caption("セッション開始")
    st.sidebar.write(st.session_state["app_opened_at"])
    selected_model = st.sidebar.selectbox(
        "Gemini モデル",
        options=list(settings.model_candidates),
    )
    model_thinking_levels = getattr(settings, "model_thinking_levels", None) or {
        model: tuple(levels) for model, levels in MODEL_THINKING_LEVELS.items()
    }
    thinking_levels = list(model_thinking_levels.get(selected_model, ("low",)))
    selected_thinking_level = st.sidebar.selectbox(
        "Thinking level",
        options=thinking_levels,
        index=0,
        help="Gemini 3 系の thinking level を指定します。",
    )
    if meta is not None:
        st.sidebar.caption("ワークシート")
        st.sidebar.write(", ".join(meta.worksheet_titles))
    return selected_model, selected_thinking_level


def render_dashboard(logs: list[dict[str, Any]]) -> None:
    st.subheader("ダッシュボード")
    today = current_time().date()
    today_logs = [row for row in logs if row.get("日付") == today.isoformat()]
    today_volume = sum(float(row.get("ボリューム_kg", 0) or 0) for row in today_logs)
    two_week_volume = sum(float(row.get("ボリューム_kg", 0) or 0) for row in logs)
    exercises = sorted({row.get("種目") for row in logs if row.get("種目")})

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("今日の種目数", len({row.get("種目") for row in today_logs if row.get("種目")}))
    metric2.metric("今日のセット数", len(today_logs))
    metric3.metric("今日の総ボリューム", f"{format_number(today_volume)} kg")
    metric4.metric("直近2週間ボリューム", f"{format_number(two_week_volume)} kg")

    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True):
            st.caption("今日のメニュー")
            st.write(summarize_today_logs(logs, today))
    with right:
        with st.container(border=True):
            st.caption("直近2週間の推移要約")
            st.write(summarize_recent_progress(logs, lookback_days=14))

    st.caption("直近ログ")
    st.dataframe(logs_to_dataframe(logs[:30]), use_container_width=True, hide_index=True)

    if exercises:
        selected_exercise = st.selectbox("推移を見る種目", exercises)
        volume_rows = build_session_volume_rows(logs, selected_exercise)
        if volume_rows:
            chart_df = pd.DataFrame(volume_rows)
            chart_df["日付"] = pd.to_datetime(chart_df["日付"])
            chart_df["総ボリューム_kg"] = pd.to_numeric(chart_df["総ボリューム_kg"])
            chart_df["最大重量_kg"] = pd.to_numeric(chart_df["最大重量_kg"])
            melted = chart_df.melt(
                id_vars=["日付"],
                value_vars=["総ボリューム_kg", "最大重量_kg"],
                var_name="指標",
                value_name="値",
            )
            chart = (
                alt.Chart(melted)
                .mark_line(point=True)
                .encode(
                    x=alt.X("日付:T", title="日付"),
                    y=alt.Y("値:Q", title="kg"),
                    color=alt.Color("指標:N", title="指標"),
                    tooltip=["日付:T", "指標:N", "値:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(chart_df, use_container_width=True, hide_index=True)


def render_manual_entry(logs: list[dict[str, Any]]) -> None:
    st.subheader("手入力")
    exercises = available_exercises()
    hydrate_manual_entry_defaults(logs)
    if not exercises:
        st.warning("種目マスターが空です。先に種目を追加してください。")
        return
    if not st.session_state["manual_exercise"] or st.session_state["manual_exercise"] not in exercises:
        st.session_state["manual_exercise"] = exercises[0]
    if st.session_state["manual_set_number"] < 1:
        st.session_state["manual_set_number"] = next_set_number(
            logs,
            st.session_state["manual_date"],
            st.session_state["manual_exercise"],
        )

    with st.form("manual_entry_form"):
        left, right = st.columns(2)
        with left:
            selected_date = st.date_input("日付", value=st.session_state["manual_date"])
            exercise = st.selectbox(
                "種目",
                options=exercises,
                index=exercises.index(st.session_state["manual_exercise"]),
            )
            set_number = st.number_input(
                "セット番号",
                min_value=1,
                value=int(st.session_state["manual_set_number"]),
                step=1,
            )
            weight_kg = st.number_input(
                "重さ(kg)",
                min_value=0.0,
                value=float(st.session_state["manual_weight_kg"]),
                step=2.5,
            )
        with right:
            reps = st.number_input(
                "回数",
                min_value=1,
                value=int(st.session_state["manual_reps"]),
                step=1,
            )
            rpe = st.selectbox(
                "RPE",
                options=RPE_OPTIONS,
                index=RPE_OPTIONS.index(st.session_state["manual_rpe"])
                if st.session_state["manual_rpe"] in RPE_OPTIONS
                else 0,
                format_func=format_rpe,
            )
            rest_seconds = st.number_input(
                "休憩秒",
                min_value=0,
                value=int(st.session_state["manual_rest_seconds"]),
                step=15,
                help="0 の場合は未入力扱いにします。",
            )
            note = st.text_input("ノート", value=st.session_state["manual_note"])

        submitted = st.form_submit_button("このセットを追加", type="primary")

    st.session_state["manual_date"] = selected_date
    st.session_state["manual_exercise"] = exercise
    st.session_state["manual_set_number"] = int(set_number)
    st.session_state["manual_weight_kg"] = float(weight_kg)
    st.session_state["manual_reps"] = int(reps)
    st.session_state["manual_rpe"] = rpe
    st.session_state["manual_rest_seconds"] = int(rest_seconds)
    st.session_state["manual_note"] = note

    if submitted:
        if repository_write(
            lambda repository: repository.append_log_rows(
                [
                    {
                        "date": selected_date.isoformat(),
                        "exercise": exercise,
                        "set_number": int(set_number),
                        "weight_kg": float(weight_kg),
                        "reps": int(reps),
                        "rpe": rpe,
                        "rest_seconds": None if int(rest_seconds) == 0 else int(rest_seconds),
                        "note": note,
                    }
                ]
            )
        ):
            st.session_state["manual_set_number"] = int(set_number) + 1
            st.session_state["manual_prefill_marker"] = ""
            st.success("ログを追加しました。")
            st.rerun()
        st.error("ログ追加に失敗しました。スプレッドシート接続を確認して再試行してください。")

    st.divider()
    st.caption("種目マスター")
    new_exercise = st.text_input("新しい種目名")
    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("種目を追加"):
            if normalize_text(new_exercise):
                if repository_write(lambda repository: repository.add_exercise(new_exercise)):
                    st.success("種目を追加しました。")
                    st.rerun()
                st.error("種目追加に失敗しました。スプレッドシート接続を確認してください。")
            else:
                st.warning("種目名を入力してください。")

    master_rows = repository_read(lambda repository: repository.load_exercise_records(), fallback=[])
    if master_rows:
        master_df = pd.DataFrame(master_rows)[["種目名", "有効", "並び順", "作成日時"]]
        edited_master = st.data_editor(
            master_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "種目名": st.column_config.TextColumn(required=True),
                "有効": st.column_config.CheckboxColumn(),
                "並び順": st.column_config.NumberColumn(min_value=1, step=1),
                "作成日時": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True,
            key="exercise_master_editor",
        )
        if st.button("種目マスターを保存"):
            if repository_write(
                lambda repository: repository.save_exercise_records(edited_master.to_dict(orient="records"))
            ):
                st.success("種目マスターを更新しました。")
                st.rerun()
            st.error("種目マスターの保存に失敗しました。")


def render_natural_language_entry(selected_model: str, selected_thinking_level: str) -> None:
    st.subheader("自然言語入力")
    st.caption("例: 今日ベンチ70kg5回を3セット、RPE8。懸垂10回を2セット。")
    text = st.text_area(
        "筋トレ内容",
        height=140,
        placeholder="今日ベンチプレス70kg5回を3セット、最後だけRPE9。懸垂は自重で10回を2セット。",
    )

    if st.button("LLMで解析", type="primary"):
        if not normalize_text(text):
            st.warning("筋トレ内容を入力してください。")
            return
        try:
            parsed = get_gemini_client().parse_workout_log(
                text=text,
                exercise_names=available_exercises(),
                now=current_time(),
                model=selected_model,
                thinking_level=selected_thinking_level,
            )
        except GeminiAPIError as exc:
            st.error(f"解析に失敗しました: {exc}")
        else:
            st.session_state["parsed_log_records"] = parsed["records"]
            st.session_state["parsed_new_exercises"] = parsed["new_exercises"]

    records = st.session_state.get("parsed_log_records", [])
    if records:
        if st.session_state.get("parsed_new_exercises"):
            st.info(
                "新規候補種目: " + ", ".join(st.session_state["parsed_new_exercises"])
            )
        parsed_df = pd.DataFrame(records).rename(
            columns={
                "date": "日付",
                "exercise": "種目",
                "set_number": "セット番号",
                "weight_kg": "重さ_kg",
                "reps": "回数",
                "rpe": "RPE",
                "rest_seconds": "休憩秒",
                "note": "ノート",
            }
        )
        for column in ("ボリューム_kg", "作成日時"):
            if column not in parsed_df.columns:
                parsed_df[column] = ""
        edited_df = st.data_editor(
            parsed_df[["日付", "種目", "セット番号", "重さ_kg", "回数", "RPE", "休憩秒", "ノート"]],
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "日付": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "セット番号": st.column_config.NumberColumn(min_value=1, step=1),
                "重さ_kg": st.column_config.NumberColumn(min_value=0.0, step=2.5),
                "回数": st.column_config.NumberColumn(min_value=1, step=1),
                "RPE": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.5),
                "休憩秒": st.column_config.NumberColumn(min_value=0, step=15),
            },
            key="parsed_logs_editor",
        )
        if st.button("解析結果を保存"):
            records_to_save = editable_logs_to_records(edited_df)
            def save_parsed_logs(repository: GoogleSheetsRepository) -> None:
                for exercise_name in st.session_state.get("parsed_new_exercises", []):
                    repository.add_exercise(exercise_name)
                for record in records_to_save:
                    repository.add_exercise(record["exercise"])
                repository.append_log_rows(records_to_save)

            if repository_write(save_parsed_logs):
                st.session_state["parsed_log_records"] = []
                st.session_state["parsed_new_exercises"] = []
                st.success(f"{len(records_to_save)} 件のログを保存しました。")
                st.rerun()
            st.error("解析結果の保存に失敗しました。")


def render_log_management(logs: list[dict[str, Any]]) -> None:
    st.subheader("ログ管理")
    if not logs:
        st.info("ログがまだありません。")
        return

    selectable = logs[:100]
    labels = {
        row["_row_number"]: (
            f"{row['日付']} | {row['種目']} | Set {row['セット番号']} | "
            f"{format_number(row['重さ_kg'])}kg x {row['回数']}"
        )
        for row in selectable
    }
    row_number = st.selectbox(
        "編集するログ",
        options=list(labels.keys()),
        format_func=lambda value: labels[value],
    )
    target = next(row for row in selectable if row["_row_number"] == row_number)
    exercises = available_exercises()
    if target["種目"] not in exercises:
        exercises = [target["種目"], *exercises]

    with st.form("edit_log_form"):
        left, right = st.columns(2)
        with left:
            selected_date = st.date_input("日付", value=parse_date(target["日付"]))
            exercise = st.selectbox(
                "種目",
                options=exercises,
                index=exercises.index(target["種目"]) if target["種目"] in exercises else 0,
            )
            set_number = st.number_input(
                "セット番号",
                min_value=1,
                value=int(target["セット番号"]),
                step=1,
            )
            weight_kg = st.number_input(
                "重さ(kg)",
                min_value=0.0,
                value=float(target["重さ_kg"]),
                step=2.5,
            )
        with right:
            reps = st.number_input("回数", min_value=1, value=int(target["回数"]), step=1)
            rpe = st.selectbox(
                "RPE",
                options=RPE_OPTIONS,
                index=RPE_OPTIONS.index(target["RPE"]) if target["RPE"] in RPE_OPTIONS else 0,
                format_func=format_rpe,
            )
            rest_seconds = st.number_input(
                "休憩秒",
                min_value=0,
                value=int(target["休憩秒"] or 0),
                step=15,
            )
            note = st.text_input("ノート", value=target["ノート"])

        update_clicked = st.form_submit_button("更新", type="primary")

    if update_clicked:
        if repository_write(
            lambda repository: repository.update_log_row(
                row_number,
                {
                    "date": selected_date.isoformat(),
                    "exercise": exercise,
                    "set_number": int(set_number),
                    "weight_kg": float(weight_kg),
                    "reps": int(reps),
                    "rpe": rpe,
                    "rest_seconds": None if int(rest_seconds) == 0 else int(rest_seconds),
                    "note": note,
                },
                created_at=target["作成日時"] or "=NOW()",
            )
        ):
            st.success("ログを更新しました。")
            st.rerun()
        st.error("ログ更新に失敗しました。")

    confirm_delete = st.checkbox("このログを削除する")
    if st.button("削除", disabled=not confirm_delete):
        if repository_write(lambda repository: repository.delete_log_row(row_number)):
            st.success("ログを削除しました。")
            st.rerun()
        st.error("ログ削除に失敗しました。スプレッドシート接続を確認して再試行してください。")


def render_advice_tab(
    logs: list[dict[str, Any]],
    selected_model: str,
    selected_thinking_level: str,
) -> None:
    st.subheader("AI相談")
    today = current_time().date()
    today_summary = summarize_today_logs(logs, today)
    trend_summary = summarize_recent_progress(logs, lookback_days=14)
    recent_advice = repository_read(
        lambda repository: repository.load_advice_history(days=2),
        fallback=[],
    )
    profile = load_profile_settings()

    left, right = st.columns([1, 1])
    with left:
        with st.container(border=True):
            st.caption("今日のメニュー")
            st.write(today_summary)
    with right:
        with st.container(border=True):
            st.caption("昨日・今日の相談履歴")
            if recent_advice:
                advice_df = pd.DataFrame(recent_advice)[["実行日時", "相談内容", "回答"]]
                st.dataframe(advice_df, use_container_width=True, hide_index=True)
            else:
                st.write("相談履歴はまだありません。")

    st.divider()
    st.caption("相談の前提")
    with st.form("advice_profile_form"):
        goal_text = st.text_area(
            "① なりたい姿・目標",
            value=profile["goal"],
            height=120,
        )
        note_text = st.text_area(
            "② 備考",
            value=profile["note"],
            height=100,
        )
        save_profile_clicked = st.form_submit_button("目標・備考を保存")
    if save_profile_clicked:
        if save_profile_settings(goal=goal_text, note=note_text):
            st.success("目標・備考を保存しました。")
            st.rerun()

    planned_minutes = st.number_input(
        "今日のジム予定滞在時間(分)",
        min_value=1,
        value=int(st.session_state.get("advice_planned_minutes", 25)),
        step=1,
        key="advice_planned_minutes",
        on_change=mark_planned_minutes_updated,
        help="デフォルトは25分です。",
    )
    planned_set_at = st.session_state.get("advice_planned_minutes_set_at", current_time().isoformat(timespec="seconds"))
    try:
        planned_set_at_dt = datetime.fromisoformat(planned_set_at)
    except ValueError:
        planned_set_at_dt = current_time()
        planned_set_at = planned_set_at_dt.isoformat(timespec="seconds")
        st.session_state["advice_planned_minutes_set_at"] = planned_set_at
    elapsed_minutes = max(
        0,
        int((current_time() - planned_set_at_dt).total_seconds() // 60),
    )
    remaining_minutes = max(0, int(planned_minutes) - elapsed_minutes)
    st.caption(
        f"予定滞在時間の入力時刻: {planned_set_at} / 相談時点の推定残り時間: {remaining_minutes} 分"
    )

    question = st.text_area(
        "相談内容",
        key="advice_question",
        height=140,
        placeholder="今日のベンチプレスが重かった。次回の重量設定と補助種目を提案して。",
    )
    ask_col, clear_col = st.columns([1, 1])
    with ask_col:
        ask_clicked = st.button("アドバイスをもらう", type="primary")
    with clear_col:
        st.button("前回入力をクリア", on_click=clear_advice_question)

    if ask_clicked:
        if not normalize_text(question):
            st.warning("相談内容を入力してください。")
            return
        try:
            advice_client = get_gemini_client()
            advice = advice_client.generate_training_advice(
                question=question,
                now=current_time(),
                goal_text=profile["goal"],
                note_text=profile["note"],
                planned_stay_minutes=int(planned_minutes),
                planned_stay_recorded_at=planned_set_at,
                remaining_minutes=remaining_minutes,
                today_summary=today_summary,
                trend_summary=trend_summary,
                recent_logs=logs,
                recent_advice=recent_advice,
                model=selected_model,
                thinking_level=selected_thinking_level,
            )
        except TypeError as exc:
            reset_gemini_client()
            try:
                advice = get_gemini_client().generate_training_advice(
                    question=question,
                    now=current_time(),
                    goal_text=profile["goal"],
                    note_text=profile["note"],
                    planned_stay_minutes=int(planned_minutes),
                    planned_stay_recorded_at=planned_set_at,
                    remaining_minutes=remaining_minutes,
                    today_summary=today_summary,
                    trend_summary=trend_summary,
                    recent_logs=logs,
                    recent_advice=recent_advice,
                    model=selected_model,
                    thinking_level=selected_thinking_level,
                )
            except TypeError as retry_exc:
                st.error(f"AI相談の内部エラーです: {retry_exc}")
                return
        except GeminiAPIError as exc:
            st.error(f"アドバイス生成に失敗しました: {exc}")
        else:
            executed_at = current_time().isoformat(timespec="seconds")
            saved = repository_write(
                lambda repository: repository.append_advice_history(
                    {
                        "実行日時": executed_at,
                        "相談内容": question.strip(),
                        "回答": "\n".join([advice["summary"], *advice["advice"]]).strip(),
                        "回答JSON": json_dumps(advice["_raw"]),
                        "今日のメニュー": today_summary,
                        "参照開始日": min((row["日付"] for row in logs), default=today.isoformat()),
                        "参照終了日": max((row["日付"] for row in logs), default=today.isoformat()),
                        "参照ログ件数": len(logs),
                        "モデル": f"{selected_model} ({selected_thinking_level})",
                    }
                )
            )
            if not saved:
                st.error("相談履歴の保存に失敗しました。スプレッドシート接続を確認してください。")
                return
            st.session_state["last_advice"] = advice
            st.success("アドバイスを保存しました。")
            st.rerun()

    advice = st.session_state.get("last_advice")
    if advice:
        with st.container(border=True):
            st.markdown(f"**要約**\n\n{advice['summary'] or '要約なし'}")
            if advice["advice"]:
                st.markdown("**提案**")
                for item in advice["advice"]:
                    st.write(f"- {item}")
            st.markdown(f"**次回の重点**\n\n{advice['next_workout_focus'] or 'なし'}")
            st.markdown(f"**注意点**\n\n{advice['caution'] or 'なし'}")


def main() -> None:
    require_login()
    ensure_session_state()
    selected_model, selected_thinking_level = render_sidebar()
    recent_logs = repository_read(
        lambda repository: repository.load_logs(days=14),
        fallback=[],
    )
    all_logs = repository_read(
        lambda repository: repository.load_logs(days=None),
        fallback=[],
    )

    st.title("筋トレログアプリ")
    st.caption("手入力、自然言語入力、LLMアドバイス、Google Sheets 保存をまとめた Streamlit アプリです。")

    tab_dashboard, tab_manual, tab_nl, tab_manage, tab_advice = st.tabs(
        ["ダッシュボード", "手入力", "自然言語入力", "ログ管理", "AI相談"]
    )
    with tab_dashboard:
        render_dashboard(recent_logs)
    with tab_manual:
        render_manual_entry(recent_logs)
    with tab_nl:
        render_natural_language_entry(selected_model, selected_thinking_level)
    with tab_manage:
        render_log_management(all_logs)
    with tab_advice:
        render_advice_tab(recent_logs, selected_model, selected_thinking_level)


if __name__ == "__main__":
    main()
