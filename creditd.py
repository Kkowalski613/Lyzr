from pathlib import Path
import io
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TOTAL_ANNUAL_CREDITS = 250_000
MONTHLY_CREDIT_LIMIT = 12_000
BURST_THRESHOLD = 10
CSV_CHUNK_SIZE = 150_000
MAX_ROWS = 1_000_000
RAW_EXPORT_LIMIT = 50_000
INPUT_MESSAGE_CANDIDATES = (
    "input_messages",
    "input_message",
    "input",
    "messages",
    "prompt",
    "user_input",
)
OUTPUT_MESSAGE_CANDIDATES = (
    "output_message",
    "output_messages",
    "response_message",
    "assistant_message",
    "assistant_response",
    "output",
    "response",
)
SESSION_ID_CANDIDATES = (
    "session_id",
    "session",
    "conversation_id",
    "chat_id",
    "run_id",
)

st.set_page_config(
    page_title="Credit Consumption Dashboard",
    layout="wide",
    page_icon="💳"
)

# ---------- Helpers ----------

def normalize_column_name(col: str) -> str:
    """
    Normalize a column name to a lowercase, underscore form for robust matching.
    Handles non-string/blank headers without raising.
    """
    return str(col).strip().lower().replace(" ", "_")


def find_local_roles_workbook() -> Path | None:
    """
    Locate a local Prophet roles workbook, preferring common filenames and falling
    back to any *.xlsx that looks like a Prophet roles file.
    """
    candidates = [
        Path("prophet roles.xlsx"),
        Path("Prophet Roles.xlsx"),
        Path("Prophet roles.xlsx"),
        Path("Prophet Final Roles.xlsx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in Path(".").glob("*.xlsx"):
        name = path.name.lower()
        if "prophet" in name and "role" in name:
            return path

    for path in Path(".").glob("*.xlsx"):
        if "prophet" in path.name.lower():
            return path

    return None


def extract_latency_ms(df: pd.DataFrame) -> pd.Series:
    """
    Look for a latency column and return a Series in milliseconds.
    Supports common variants like latency (seconds), latency_ms, duration_ms, etc.
    """
    if df.empty:
        return pd.Series(dtype="float")

    latency_aliases = {
        "latency_ms": "ms",
        "latency": "s",
        "latency_seconds": "s",
        "latency_s": "s",
        "response_time_ms": "ms",
        "duration_ms": "ms",
        "elapsed_ms": "ms",
    }

    normalized_cols = {col: normalize_column_name(col) for col in df.columns}
    selected_col = None
    selected_unit = "ms"

    for original, normalized in normalized_cols.items():
        if normalized in latency_aliases:
            selected_col = original
            selected_unit = latency_aliases[normalized]
            break

    if selected_col is None:
        return pd.Series(pd.NA, index=df.index, dtype="float")

    latency_series = pd.to_numeric(df[selected_col], errors="coerce")
    if selected_unit == "s":
        latency_series = latency_series * 1000

    # Negative latencies are treated as missing
    latency_series = latency_series.where(latency_series >= 0)
    return latency_series


def parse_date_safe(series: pd.Series) -> pd.Series:
    # Your created_at now looks like "12/1/25"
    # Let pandas infer the format; errors become NaT
    return pd.to_datetime(series, errors="coerce")


def get_month_key(dt_series: pd.Series) -> pd.Series:
    # Format like "2025-12"
    return dt_series.dt.to_period("M").astype(str)


TIME_GRANULARITIES = ("Monthly", "Weekly", "Daily")


def format_period_label(period: str, granularity: str) -> str:
    """
    Human-friendly label for a period key based on the selected granularity.
    """
    try:
        if granularity == "Monthly":
            ts = pd.Period(period, freq="M").to_timestamp()
            return ts.strftime("%B %Y")
        if granularity == "Weekly":
            start = pd.Period(period, freq="W-SUN").start_time
            end = start + pd.Timedelta(days=6)
            return f"{start:%b %d, %Y} – {end:%b %d, %Y}"
        if granularity == "Daily":
            ts = pd.to_datetime(period)
            return ts.strftime("%b %d, %Y")
    except Exception:
        return str(period)

    return str(period)


def period_sort_key(period: str, granularity: str) -> pd.Timestamp:
    """
    Produce a sortable timestamp for chronological ordering of period keys.
    """
    try:
        if granularity == "Monthly":
            return pd.Period(period, freq="M").start_time
        if granularity == "Weekly":
            return pd.Period(period, freq="W-SUN").start_time
        if granularity == "Daily":
            return pd.to_datetime(period)
    except Exception:
        return pd.Timestamp.min

    return pd.Timestamp.min


def build_period_options(df: pd.DataFrame, granularity: str) -> tuple[list[str], dict[str, str]]:
    """
    Return sorted period keys and labels for the given granularity.
    """
    column_map = {"Monthly": "month_key", "Weekly": "week_key", "Daily": "date_key"}
    column = column_map.get(granularity)
    if column is None or column not in df.columns:
        return [], {}

    unique_periods = df[column].dropna().astype(str).unique()
    sorted_periods = sorted(unique_periods, key=lambda p: period_sort_key(p, granularity))
    labels = {p: format_period_label(p, granularity) for p in sorted_periods}
    return sorted_periods, labels


def compute_rank_map(df: pd.DataFrame) -> dict[str, int]:
    """
    Create a mapping of user -> rank (#) based on total credits within a period.
    """
    if df.empty or "email" not in df.columns:
        return {}

    filtered = df[df["email"] != "(no email)"].copy()
    if filtered.empty:
        return {}

    grouped = (
        filtered.groupby("email", dropna=False)["credits"]
        .sum()
        .round(2)
        .reset_index()
        .rename(columns={"email": "User", "credits": "Credits"})
        .sort_values("Credits", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.index + 1
    return dict(zip(grouped["User"].str.lower(), grouped["rank"]))


def export_to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Build an Excel workbook from provided sheet name -> dataframe mapping.
    Large raw sheets are capped to RAW_EXPORT_LIMIT rows to avoid huge files.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            trimmed = df
            if len(trimmed) > RAW_EXPORT_LIMIT:
                trimmed = trimmed.head(RAW_EXPORT_LIMIT)
            safe_name = name[:31]  # Excel sheet name limit
            trimmed.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def format_rank_change(current_rank: int, previous_rank: int | None) -> str:
    """
    Format the rank delta with arrows for quick scanning.
    """
    if previous_rank is None:
        return "—"

    delta = previous_rank - current_rank
    if delta > 0:
        return f"▲{delta}"
    if delta < 0:
        return f"▼{abs(delta)}"
    return "—"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean & normalize your export using the actual column names:

    log_id          -> log_id          (used to filter out bad rows)
    agent_id        -> agent_id
    call_type       -> call_type       (toggle embeddings)
    language_model  -> language_model  (model breakdown)
    email           -> email           (user breakdown)
    actions         -> credits         (totals)
    created_at      -> created_at      (date, for month grouping)
    """

    df = df.copy()

    # ---- Ensure the columns we expect are present ----
    required_cols = [
        "log_id",
        "agent_id",
        "call_type",
        "language_model",
        "email",
        "actions",      # this is your credits column
        "created_at",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error("Missing required columns in CSV: " + ", ".join(missing))
        return pd.DataFrame()

    # 1) Normalize log_id and filter out "bad" ones
    df["log_id"] = df["log_id"].astype(str).str.strip()
    bad_log_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    invalid_log_mask = (
        df["log_id"].str.lower().isin(bad_log_values)
        | (df["log_id"].str.len() < 8)            # “proper log ID” heuristic
    )
    df = df[~invalid_log_mask].copy()

    # 2) Credits = actions
    df["credits"] = pd.to_numeric(df["actions"], errors="coerce")
    df = df[df["credits"].notna() & (df["credits"] > 0)].copy()

    # 3) Call type & model
    df["call_type"] = df["call_type"].astype(str).str.strip()
    df["language_model"] = df["language_model"].astype(str).str.strip()
    df = df[
        (df["call_type"] != "") &
        (df["language_model"] != "")
    ].copy()

    # 4) Timestamp (you fixed this to proper datetimes)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df[df["created_at"].notna()].copy()

    # 5) Month key for the selector
    df["month_key"] = df["created_at"].dt.to_period("M").astype(str)
    df["week_key"] = df["created_at"].dt.to_period("W-SUN").astype(str)
    df["date_key"] = df["created_at"].dt.date.astype(str)

    # 6) Email normalization (avoid NaN showing up)
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    bad_email_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    df.loc[df["email"].str.lower().isin(bad_email_values), "email"] = "(no email)"

    # 7) Agent ID normalization
    df["agent_id"] = df["agent_id"].astype(str).str.strip().str.lower()
    bad_agent_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    df.loc[df["agent_id"].str.lower().isin(bad_agent_values), "agent_id"] = "(unassigned)"

    # 8) Latency (optional, kept as-is even when missing)
    df["latency_ms"] = extract_latency_ms(df)

    return df


def detect_burst_users(df: pd.DataFrame, threshold: int = BURST_THRESHOLD) -> set[str]:
    """
    Flag users with more than `threshold` messages inside any single hour.
    """
    if df.empty or "created_at" not in df.columns:
        return set()

    hourly_counts = (
        df[df["email"] != "(no email)"]
        .assign(hour_window=lambda d: d["created_at"].dt.floor("H"))
        .groupby(["email", "hour_window"])
        .size()
    )
    return set(hourly_counts[hourly_counts > threshold].index.get_level_values("email"))


def load_roles_mapping(uploaded_roles) -> pd.DataFrame:
    """
    Load the Prophet roles workbook (uploaded or local) and normalize key columns.
    Expected columns: email, role, discipline, office (case-insensitive).
    """
    local_roles = find_local_roles_workbook()
    source = uploaded_roles if uploaded_roles is not None else local_roles
    source_label = getattr(uploaded_roles, "name", None) if uploaded_roles is not None else (local_roles.name if local_roles else None)

    if source is None:
        return pd.DataFrame()

    try:
        roles_df = pd.read_excel(source)
    except Exception as exc:  # pragma: no cover - UI warning
        label = source_label or "Prophet roles workbook"
        st.warning(f"Could not read {label}: {exc}")
        return pd.DataFrame()

    # Normalize common alternate headers
    alias_map = {
        "column1": "email",          # sample workbook
        "column m": "email",
        "column_m": "email",
        "broader role": "role",
        "broader_role": "role",
    }

    normalized_columns = {}
    for col in roles_df.columns:
        normalized = normalize_column_name(col)
        normalized_columns[col] = alias_map.get(normalized, normalized)
    roles_df = roles_df.rename(columns=normalized_columns)

    if "email" not in roles_df.columns:
        label = source_label or "Prophet roles workbook"
        st.warning(f"{label} is missing an 'email' column; skipping enrichment.")
        return pd.DataFrame()

    for col in ("role", "discipline", "office"):
        if col not in roles_df.columns:
            roles_df[col] = ""

    roles_df["email"] = roles_df["email"].astype(str).str.strip().str.lower()
    roles_df["role"] = roles_df["role"].astype(str).str.strip()
    roles_df["discipline"] = roles_df["discipline"].astype(str).str.strip()
    roles_df["office"] = roles_df["office"].astype(str).str.strip()

    return roles_df[["email", "role", "discipline", "office"]]


def find_local_agent_file() -> Path | None:
    """
    Locate a local AgentID Names JSON, favoring common spellings.
    """
    candidates = [
        Path("AgentID Names.json"),
        Path("AgentID Names.JSON"),
        Path("agentid names.json"),
        Path("AgentID_Names.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_agent_metadata(uploaded_json) -> pd.DataFrame:
    """
    Load the AgentID Names JSON (uploaded or local) to map agent_id -> name/features.
    Expected fields: _id/agent_id/id, name, features (list with type keys).
    """
    local_agents = find_local_agent_file()
    source = uploaded_json if uploaded_json is not None else local_agents
    source_label = getattr(uploaded_json, "name", None) if uploaded_json is not None else (local_agents.name if local_agents else None)

    if source is None:
        return pd.DataFrame()

    try:
        if hasattr(source, "read"):  # uploaded file-like
            data = json.load(source)
        else:
            with open(source, "r", encoding="utf-8") as fh:
                data = json.load(fh)
    except Exception as exc:  # pragma: no cover - UI warning
        label = source_label or "AgentID Names.json"
        st.warning(f"Could not read {label}: {exc}")
        return pd.DataFrame()

    rows = []
    records = data if isinstance(data, list) else []
    for item in records:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("_id") or item.get("agent_id") or item.get("id") or ""
        agent_id = str(raw_id).strip()
        if agent_id == "":
            continue
        feature_list = item.get("features") or []
        feature_types = {
            str(f.get("type", "")).strip().lower()
            for f in feature_list
            if isinstance(f, dict) and "type" in f
        }
        rows.append(
            {
                "agent_id": agent_id.lower(),
                "agent_name": str(item.get("name", "")).strip(),
                "has_kb": "knowledge_base" in feature_types,
                "uses_context": "context" in feature_types,
                "uses_memory": "memory" in feature_types,
            }
        )

    if not rows:
        return pd.DataFrame()

    agents_df = pd.DataFrame(rows)
    agents_df["agent_id"] = agents_df["agent_id"].astype(str).str.strip().str.lower()
    for col in ("has_kb", "uses_context", "uses_memory"):
        agents_df[col] = agents_df[col].fillna(False)
    return agents_df.drop_duplicates("agent_id")


def load_usage_data(uploaded_file, chunksize: int = CSV_CHUNK_SIZE, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, int, bool]:
    """
    Load the main usage file with chunking to avoid memory blow-ups.
    Supports CSV (preferred) and Excel fallback. Returns (df, rows_read, truncated_flag).
    """
    if uploaded_file is None:
        return pd.DataFrame(), 0, False

    name = (uploaded_file.name or "").lower()
    truncated = False

    if name.endswith(".csv"):
        chunks = []
        rows = 0
        try:
            for chunk in pd.read_csv(uploaded_file, chunksize=chunksize):
                chunks.append(chunk)
                rows += len(chunk)
                if rows >= max_rows:
                    truncated = True
                    break
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return pd.DataFrame(), 0, False

        if not chunks:
            return pd.DataFrame(), 0, False

        df = pd.concat(chunks, ignore_index=True)
        if truncated:
            df = df.head(max_rows)
        return df, len(df), truncated

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(uploaded_file)
            return df, len(df), False
        except Exception as exc:
            st.error(f"Could not read Excel. Convert to CSV for better performance. Error: {exc}")
            return pd.DataFrame(), 0, False

    st.error("Unsupported file type. Please upload a CSV (preferred) or Excel file.")
    return pd.DataFrame(), 0, False


def model_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    grouped = (
        df.groupby("language_model", dropna=False)
          .agg(
              Calls=("language_model", "size"),
              Credits=("credits", "sum"),
          )
          .reset_index()
          .rename(columns={"language_model": "Model"})
    )
    grouped["Credits per Call"] = grouped.apply(
        lambda row: (row["Credits"] / row["Calls"]) if row["Calls"] > 0 else np.nan,
        axis=1,
    )
    total_credits = grouped["Credits"].sum()
    if total_credits > 0:
        grouped["Share (%)"] = (grouped["Credits"] / total_credits * 100).round(1)
    else:
        grouped["Share (%)"] = np.nan

    grouped["Credits per Call"] = grouped["Credits per Call"].round(2)
    grouped["Credits"] = grouped["Credits"].round(2)
    grouped = grouped.sort_values("Credits", ascending=False)
    return grouped


def user_breakdown(
    df: pd.DataFrame,
    burst_users: set[str] | None = None,
    roles_df: pd.DataFrame | None = None,
    show_roles: bool = False,
    previous_ranks: dict[str, int] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    burst_users = {u.strip().lower() for u in (burst_users or set())}
    roles_df = roles_df if roles_df is not None else pd.DataFrame()
    previous_ranks = {u.strip().lower(): rank for u, rank in (previous_ranks or {}).items()}

    # Exclude rows with "(no email)" from the ranking, but keep them in totals
    filtered = df[df["email"] != "(no email)"].copy()
    if filtered.empty:
        return pd.DataFrame()

    grouped = (
        filtered.groupby("email", dropna=False)
                .agg(
                    Calls=("email", "size"),
                    Credits=("credits", "sum"),
                    Agents=("agent_id", lambda s: s[s != "(unassigned)"].nunique()),
                )
                .reset_index()
                .rename(columns={"email": "User", "Agents": "Agents Engaged"})
    )
    grouped["Credits"] = grouped["Credits"].round(2)
    grouped["Credits per Call"] = grouped.apply(
        lambda row: (row["Credits"] / row["Calls"]) if row["Calls"] > 0 else np.nan,
        axis=1,
    )
    grouped["Credits per Call"] = grouped["Credits per Call"].round(2)
    grouped = grouped.sort_values("Credits", ascending=False)
    grouped.insert(0, "#", range(1, len(grouped) + 1))
    total_user_credits = grouped["Credits"].sum()
    grouped["Credit Share (%)"] = (
        (grouped["Credits"] / total_user_credits * 100).round(1) if total_user_credits > 0 else np.nan
    )
    grouped.insert(
        1,
        "Rank Change",
        grouped.apply(
            lambda row: format_rank_change(
                row["#"],
                previous_ranks.get(row["User"].strip().lower()) if previous_ranks else None,
            ),
            axis=1,
        ),
    )

    grouped["Over 12k Credits"] = grouped["Credits"] > MONTHLY_CREDIT_LIMIT
    grouped["Burst Detected"] = grouped["User"].str.lower().isin(burst_users)
    grouped["Flags"] = grouped.apply(
        lambda row: ", ".join(
            flag for flag in [
                "Over monthly allotment" if row["Over 12k Credits"] else "",
                "Burst activity" if row["Burst Detected"] else "",
            ] if flag
        ) or "OK",
        axis=1,
    )

    if show_roles and not roles_df.empty:
        roles_lookup = roles_df.drop_duplicates("email").set_index("email")
        grouped["Role"] = grouped["User"].str.lower().map(roles_lookup["role"]).fillna("")
        grouped["Discipline"] = grouped["User"].str.lower().map(roles_lookup["discipline"]).fillna("")
        grouped["Office"] = grouped["User"].str.lower().map(roles_lookup["office"]).fillna("")

        ordered_columns = ["#", "Rank Change", "User", "Role", "Discipline", "Office", "Calls", "Credits", "Credit Share (%)", "Credits per Call", "Agents Engaged", "Over 12k Credits", "Burst Detected", "Flags"]
    else:
        ordered_columns = ["#", "Rank Change", "User", "Calls", "Credits", "Credit Share (%)", "Credits per Call", "Agents Engaged", "Over 12k Credits", "Burst Detected", "Flags"]

    return grouped[ordered_columns]


def agent_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if "agent_name" not in df.columns:
        df["agent_name"] = ""

    grouped = (
        df.groupby(["agent_id", "agent_name"], dropna=False)
          .agg(
              Calls=("agent_id", "size"),
              Credits=("credits", "sum"),
          )
          .reset_index()
          .rename(columns={"agent_id": "Agent ID", "agent_name": "Agent Name"})
    )
    grouped["Credits"] = grouped["Credits"].round(2)
    grouped = grouped.sort_values("Credits", ascending=False)
    grouped["Agent Name"] = grouped["Agent Name"].fillna("").replace({"nan": ""})
    return grouped


def enrich_with_roles(df: pd.DataFrame, roles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach role/discipline/office to the usage frame for grouping and charts.
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    if roles_df is None or roles_df.empty:
        for col in ("role", "discipline", "office"):
            if col not in result.columns:
                result[col] = ""
        return result

    lookup = roles_df.drop_duplicates("email").set_index("email")
    result["role"] = result["email"].map(lookup["role"]).fillna("")
    result["discipline"] = result["email"].map(lookup["discipline"]).fillna("")
    result["office"] = result["email"].map(lookup["office"]).fillna("")
    return result


def enrich_with_agents(df: pd.DataFrame, agents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach agent metadata (name, feature flags) by agent_id.
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    for col in ("agent_name", "has_kb", "uses_context", "uses_memory"):
        if col not in result.columns:
            result[col] = "" if col == "agent_name" else False

    if agents_df is None or agents_df.empty:
        return result

    lookup = agents_df.drop_duplicates("agent_id").set_index("agent_id")
    result["agent_name"] = result["agent_id"].str.lower().map(lookup["agent_name"]).fillna("")
    result["has_kb"] = result["agent_id"].str.lower().map(lookup["has_kb"]).fillna(False)
    result["uses_context"] = result["agent_id"].str.lower().map(lookup["uses_context"]).fillna(False)
    result["uses_memory"] = result["agent_id"].str.lower().map(lookup["uses_memory"]).fillna(False)
    return result


def style_user_dataframe(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Highlight users exceeding the monthly allotment in red for quick scanning.
    """
    if df.empty:
        return df.style

    def highlight_user(row: pd.Series) -> list[str]:
        should_highlight = bool(row.get("Over 12k Credits", False))
        styles = []
        for col_name in row.index:
            if col_name == "User" and should_highlight:
                styles.append("color: red; font-weight: 600;")
            else:
                styles.append("")
        return styles

    return df.style.apply(highlight_user, axis=1)


def select_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """
    Return the first matching column from the candidate list (case/space insensitive).
    """
    if df is None or df.empty:
        return None

    normalized = {normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def parse_jsonish(value):
    """
    Try to JSON-decode a value; fall back to the original string/object on failure.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return text
    return value


def count_words(text: str | float | int | None) -> int:
    """
    Simple whitespace word count, ignoring empty/NaN values.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    return len(str(text).split())


def extract_user_text(payload) -> str:
    """
    Pull user-authored content from a chat payload shaped like:
    [{"role":"system","content":"..."},{"role":"user","content":"..."}]
    Falls back to the raw string for plain-text prompts.
    """
    parsed = parse_jsonish(payload)
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            if role == "user":
                content = item.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content if c)
                parts.append(str(content))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(parsed, dict):
        content = parsed.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content if c)
        return str(content).strip()
    if isinstance(parsed, str):
        return parsed.strip()
    return ""


def extract_output_details(payload) -> tuple[str, int]:
    """
    Return (assistant_text, tool_call_count) from a response payload that may include
    `content`, `tool_calls`, or `function_call`.
    """
    parsed = parse_jsonish(payload)
    output_text = ""
    tool_calls = 0

    if isinstance(parsed, dict):
        content = (
            parsed.get("content", "")
            or parsed.get("message", "")
            or parsed.get("text", "")
        )
        if isinstance(content, list):
            content = " ".join(str(c) for c in content if c)
        output_text = str(content).strip()

        tc = parsed.get("tool_calls")
        if isinstance(tc, list):
            tool_calls = len(tc)
        elif tc:
            tool_calls = 1

        fc = parsed.get("function_call")
        if isinstance(fc, dict):
            tool_calls = max(tool_calls, 1)
    elif isinstance(parsed, list):
        text_parts = []
        for item in parsed:
            if isinstance(item, dict):
                role = str(item.get("role", "")).lower()
                if role in ("assistant", "") and "content" in item:
                    text_parts.append(str(item.get("content", "")))
                if "tool_calls" in item and isinstance(item["tool_calls"], list):
                    tool_calls = max(tool_calls, len(item["tool_calls"]))
            else:
                text_parts.append(str(item))
        output_text = "\n".join(text_parts).strip()
    elif isinstance(parsed, str):
        output_text = parsed.strip()

    return output_text, tool_calls


def build_query_insights(df: pd.DataFrame) -> dict:
    """
    Derive query/message metrics (lengths, tool calls, chats per session) when
    input/output and session columns are present.
    """
    result = {
        "available": False,
        "reason": "",
        "rows": pd.DataFrame(),
        "per_user": pd.DataFrame(),
        "over_time": pd.DataFrame(),
        "tool_calls_distribution": pd.DataFrame(),
        "session_counts": pd.DataFrame(),
        "session_over_time": pd.DataFrame(),
        "latency_vs_input": pd.DataFrame(),
        "summary": {},
    }

    if df is None or df.empty:
        result["reason"] = "No rows available after filters."
        return result

    input_col = select_first_column(df, INPUT_MESSAGE_CANDIDATES)
    output_col = select_first_column(df, OUTPUT_MESSAGE_CANDIDATES)
    session_col = select_first_column(df, SESSION_ID_CANDIDATES)

    if not any([input_col, output_col, session_col]):
        result["reason"] = "Input/output/session columns not found in the data."
        return result

    working = pd.DataFrame(index=df.index)
    working["created_at"] = df["created_at"]
    working["email"] = df.get("email", "(no email)")
    working["latency_ms"] = pd.to_numeric(df.get("latency_ms", np.nan), errors="coerce")

    if input_col:
        working["input_text"] = df[input_col].apply(extract_user_text)
        working["input_words"] = working["input_text"].apply(count_words)
    else:
        working["input_text"] = ""
        working["input_words"] = np.nan

    if output_col:
        output_details = df[output_col].apply(extract_output_details)
        working["output_text"] = output_details.apply(lambda t: t[0])
        working["tool_calls"] = output_details.apply(lambda t: t[1])
        working["output_words"] = working["output_text"].apply(count_words)
    else:
        working["output_text"] = ""
        working["tool_calls"] = np.nan
        working["output_words"] = np.nan

    if session_col:
        working["session_id"] = (
            df[session_col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})
        )
    else:
        working["session_id"] = pd.NA

    working["created_date"] = working["created_at"].dt.date

    # Keep rows that have at least some message content
    valid_mask = (
        working["input_words"].fillna(0) > 0
    ) | (working["output_words"].fillna(0) > 0)
    query_rows = working[valid_mask].copy()

    if query_rows.empty:
        result["reason"] = "Message columns were found, but no usable text was detected."
        return result

    per_user = (
        query_rows[query_rows["email"] != "(no email)"]
        .groupby("email")
        .agg(
            avg_input_words=("input_words", "mean"),
            avg_output_words=("output_words", "mean"),
            avg_tool_calls=("tool_calls", "mean"),
            messages=("input_words", "size"),
        )
        .reset_index()
        .rename(columns={"email": "User"})
        .sort_values("messages", ascending=False)
    )

    over_time = (
        query_rows
        .groupby("created_date")
        .agg(
            avg_input_words=("input_words", "mean"),
            avg_output_words=("output_words", "mean"),
            avg_tool_calls=("tool_calls", "mean"),
            messages=("input_words", "size"),
        )
        .reset_index()
        .rename(columns={"created_date": "Date"})
        .sort_values("Date")
    )

    tool_calls_distribution = pd.DataFrame()
    if "tool_calls" in query_rows.columns:
        tool_calls_distribution = (
            query_rows.assign(tool_calls=lambda d: pd.to_numeric(d["tool_calls"], errors="coerce").fillna(0).astype(int))
            .groupby("tool_calls")
            .size()
            .reset_index(name="Count")
            .rename(columns={"tool_calls": "Tool Calls"})
            .sort_values("Tool Calls")
        )

    session_counts = pd.DataFrame()
    session_over_time = pd.DataFrame()
    sessions_df = query_rows.dropna(subset=["session_id"])
    if not sessions_df.empty:
        def primary_user(series: pd.Series) -> str:
            non_empty = series[series != "(no email)"]
            if non_empty.empty:
                return "(no email)"
            try:
                return non_empty.mode().iat[0]
            except Exception:
                return non_empty.iloc[0]

        session_counts = (
            sessions_df
            .groupby("session_id")
            .agg(
                Chats=("session_id", "size"),
                First_Seen=("created_at", "min"),
                User=("email", primary_user),
            )
            .reset_index()
            .rename(columns={"session_id": "Session ID"})
            .sort_values("Chats", ascending=False)
        )
        session_counts["First_Seen"] = pd.to_datetime(session_counts["First_Seen"])
        session_counts["Start Date"] = session_counts["First_Seen"].dt.date

        session_over_time = (
            session_counts
            .groupby("Start Date")
            .agg(
                avg_chats_per_session=("Chats", "mean"),
                sessions=("Session ID", "count"),
            )
            .reset_index()
            .rename(columns={"Start Date": "Date"})
            .sort_values("Date")
        )

    result["available"] = True
    result["rows"] = query_rows
    result["per_user"] = per_user
    result["over_time"] = over_time
    result["tool_calls_distribution"] = tool_calls_distribution
    result["session_counts"] = session_counts
    result["session_over_time"] = session_over_time
    result["latency_vs_input"] = (
        query_rows.dropna(subset=["latency_ms"])
        .groupby("created_date")
        .agg(
            avg_input_words=("input_words", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
        .rename(columns={"created_date": "Date"})
        .sort_values("Date")
    )
    result["summary"] = {
        "avg_input_words": query_rows["input_words"].mean(),
        "avg_output_words": query_rows["output_words"].mean(),
        "avg_tool_calls": pd.to_numeric(query_rows["tool_calls"], errors="coerce").mean(),
        "avg_chats_per_session": session_counts["Chats"].mean() if not session_counts.empty else np.nan,
    }

    return result


# ---------- UI ----------

st.markdown(
    """
    <style>
    .small-text { font-size: 0.8rem; color: #9CA3AF; }
    .metric-subtitle { font-size: 0.75rem; color: #9CA3AF; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💳 Credit Consumption Dashboard")

st.markdown(
    """
    Review credit consumption by **model**, **user (email)**, and **agent ID**  
    from your monthly export, with credits sourced from `actions`
    and only rows with a proper `log_id` included.
    """
)

with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload CSV export",
        type=["csv"],
        help="Uses columns: log_id, agent_id, call_type, language_model, email, actions (credits), created_at"
    )

    include_embeddings = st.checkbox(
        "Include embeddings (call_type = aembedding)",
        value=True
    )

    local_roles_path = find_local_roles_workbook()
    roles_file = st.file_uploader(
        "Upload Prophet roles.xlsx (optional)",
        type=["xlsx"],
        help="Adds role, discipline, and office columns by matching email.",
    )
    roles_toggle_default = roles_file is not None or local_roles_path is not None
    show_roles = (
        st.checkbox(
            "Show role/discipline/office (if available)",
            value=roles_toggle_default,
            help="Requires Prophet roles.xlsx (uploaded or in the app folder).",
        )
        if roles_toggle_default
        else False
    )
    if local_roles_path is not None and roles_file is None:
        st.caption(f"Using local roles workbook: {local_roles_path.name}")

    local_agent_file = find_local_agent_file()
    agent_file = st.file_uploader(
        "Upload AgentID Names.json (optional)",
        type=["json"],
        help="Maps agent_id to agent name and features (knowledge base, context, memory).",
    )
    if local_agent_file is not None and agent_file is None:
        st.caption(f"Using local agent JSON: {local_agent_file.name}")

    max_rows_setting = st.number_input(
        "Max rows to load (for large CSVs)",
        min_value=50_000,
        max_value=2_000_000,
        value=MAX_ROWS,
        step=50_000,
        help="Files larger than this limit will be truncated to protect memory. Lower this if uploads crash; raise cautiously if you have RAM.",
    )

    st.markdown(
        """
        <div class="small-text">
        Rows are dropped if: invalid/missing <code>log_id</code>, 
        <code>credits</code> (actions) ≤ 0, missing <code>call_type</code> /
        <code>language_model</code>, or invalid <code>created_at</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

# ---------- Load & clean ----------

with st.spinner("Loading file (chunked for large CSVs)..."):
    raw_df, total_rows, truncated = load_usage_data(uploaded_file, max_rows=int(max_rows_setting))

if raw_df.empty:
    st.error("Could not read any rows from the uploaded file.")
    st.stop()

clean_df = clean_dataframe(raw_df)

clean_rows = len(clean_df)
dropped_rows = total_rows - clean_rows
if truncated:
    st.warning(f"Large file truncated to first {total_rows:,} rows for performance (max {int(max_rows_setting):,}). Lower the limit if it still crashes, or convert to CSV if using Excel.")

if clean_df.empty:
    st.error("No valid rows after cleaning. Check that the CSV uses the expected columns.")
    st.stop()

roles_df = load_roles_mapping(roles_file)
roles_available = show_roles and not roles_df.empty
if show_roles and roles_df.empty:
    st.info("Prophet roles.xlsx not provided or unreadable. Role enrichment is skipped.")

agents_df = load_agent_metadata(agent_file)
agents_available = not agents_df.empty
if agent_file is not None and agents_df.empty:
    st.info("AgentID Names.json not provided or unreadable. Agent metadata is skipped.")
elif agents_available:
    st.caption(f"Loaded {len(agents_df):,} agent records; IDs normalized to lowercase for matching.")

# Monthly consumption pattern (for projections & charts)
monthly_base = clean_df.copy()
if not include_embeddings:
    monthly_base = monthly_base[monthly_base["call_type"] != "aembedding"].copy()

monthly_totals = monthly_base.groupby("month_key")["credits"].sum()
observed_months = len(monthly_totals)
avg_monthly_usage = monthly_totals.mean() if observed_months > 0 else np.nan
projected_annual_usage = avg_monthly_usage * 12 if observed_months > 0 else np.nan
estimated_remaining = (
    TOTAL_ANNUAL_CREDITS - projected_annual_usage if pd.notna(projected_annual_usage) else np.nan
)

# Monthly activity frame for charts
monthly_usage_df = (
    monthly_base.groupby("month_key")
    .agg(
        total_credits=("credits", "sum"),
        total_calls=("credits", "size"),
        active_users=("email", lambda s: s[s != "(no email)"].nunique()),
    )
    .reset_index()
    .sort_values("month_key")
)
monthly_usage_df["avg_calls_per_user"] = monthly_usage_df.apply(
    lambda row: row["total_calls"] / row["active_users"] if row["active_users"] > 0 else np.nan,
    axis=1,
)
monthly_usage_df["avg_credits_per_user"] = monthly_usage_df.apply(
    lambda row: row["total_credits"] / row["active_users"] if row["active_users"] > 0 else np.nan,
    axis=1,
)
monthly_usage_df["month_label"] = monthly_usage_df["month_key"].apply(
    lambda m: pd.Period(m).to_timestamp().strftime("%b %Y")
)
monthly_usage_df["cumulative_credits"] = monthly_usage_df["total_credits"].cumsum()

weekly_usage_df = (
    monthly_base.groupby("week_key")
    .agg(
        total_credits=("credits", "sum"),
        total_calls=("credits", "size"),
        active_users=("email", lambda s: s[s != "(no email)"].nunique()),
    )
    .reset_index()
    .sort_values("week_key")
)
weekly_usage_df["week_start"] = weekly_usage_df["week_key"].apply(lambda w: pd.Period(w, freq="W-SUN").start_time)
weekly_usage_df["week_end"] = weekly_usage_df["week_start"] + pd.Timedelta(days=6)
weekly_usage_df["week_label"] = weekly_usage_df.apply(
    lambda row: f"{row['week_start']:%b %d} – {row['week_end']:%b %d}",
    axis=1,
)

# Timeframe selection (month/week/day)
time_granularity = st.radio(
    "Timeframe",
    TIME_GRANULARITIES,
    index=0,
    horizontal=True,
)
period_column_map = {"Monthly": "month_key", "Weekly": "week_key", "Daily": "date_key"}
selected_column = period_column_map[time_granularity]

available_periods, period_labels = build_period_options(clean_df, time_granularity)
if not available_periods:
    st.error("No usable periods found after cleaning.")
    st.stop()

selected_period = st.selectbox(
    f"{time_granularity} period",
    options=available_periods,
    format_func=lambda p: period_labels.get(p, p),
    index=len(available_periods) - 1,
)
selected_period_label = period_labels.get(selected_period, selected_period)
period_slice = clean_df[clean_df[selected_column] == selected_period].copy()

previous_visible_df = pd.DataFrame()
selected_idx = available_periods.index(selected_period)
if selected_idx > 0:
    previous_slice = clean_df[clean_df[selected_column] == available_periods[selected_idx - 1]].copy()
    if include_embeddings:
        previous_visible_df = previous_slice.copy()
    else:
        previous_visible_df = previous_slice[previous_slice["call_type"] != "aembedding"].copy()

# Apply embeddings toggle
if include_embeddings:
    visible_df = period_slice.copy()
else:
    visible_df = period_slice[period_slice["call_type"] != "aembedding"].copy()

previous_ranks = compute_rank_map(previous_visible_df)
burst_users = detect_burst_users(visible_df, threshold=BURST_THRESHOLD)
visible_enriched_df = enrich_with_roles(visible_df, roles_df)
visible_enriched_df = enrich_with_agents(visible_enriched_df, agents_df)
query_insights = build_query_insights(visible_enriched_df)

# Embedding share is *always* computed on the full period
embeddings_df = period_slice[period_slice["call_type"] == "aembedding"]
total_period_credits = period_slice["credits"].sum()
embedding_credits = embeddings_df["credits"].sum()
embedding_share = (embedding_credits / total_period_credits) if total_period_credits > 0 else np.nan
period_label_short = {"Monthly": "Month", "Weekly": "Week", "Daily": "Day"}[time_granularity]

# ---------- Top metrics ----------

total_credits_visible = visible_df["credits"].sum()
total_calls_visible = len(visible_df)
active_users = visible_df[visible_df["email"] != "(no email)"]["email"].nunique()
mapped_agents = visible_enriched_df[(visible_enriched_df["agent_id"] != "(unassigned)") & (visible_enriched_df["agent_name"] != "")]
total_agents_present = visible_enriched_df[visible_enriched_df["agent_id"] != "(unassigned)"]["agent_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        f"Total Credits (Selected {period_label_short})",
        f"{total_credits_visible:,.2f}" if total_calls_visible > 0 else "—",
    )
    st.markdown(
        '<div class="metric-subtitle">From actions/credits, after filters</div>',
        unsafe_allow_html=True,
    )

with col2:
    st.metric(
        "Total Calls (Rows)",
        f"{total_calls_visible:,.0f}" if total_calls_visible > 0 else "—",
    )
    st.markdown(
        '<div class="metric-subtitle">Only rows with a proper log_id</div>',
        unsafe_allow_html=True,
    )

with col3:
    st.metric(
        "Active Users (Emails)",
        f"{active_users:,.0f}" if total_calls_visible > 0 else "—",
    )
    st.markdown(
        '<div class="metric-subtitle">Distinct non-empty email addresses</div>',
        unsafe_allow_html=True,
    )

with col4:
    st.metric(
        "Embedding Share (Credits)",
        f"{embedding_share * 100:,.1f} %" if pd.notna(embedding_share) else "—",
        help="Share of total credits in this period attributed to call_type = aembedding."
    )
    st.markdown(
        '<div class="metric-subtitle">Computed on full period, regardless of toggle</div>',
        unsafe_allow_html=True,
    )

if agents_available and total_agents_present > 0:
    st.caption(
        f"Agent mapping: {mapped_agents['agent_id'].nunique():,} of {total_agents_present:,} agent IDs matched to names."
    )

latency_samples = visible_df["latency_ms"].dropna() if "latency_ms" in visible_df.columns else pd.Series(dtype="float")
if not latency_samples.empty:
    lat_col1, lat_col2, lat_col3 = st.columns(3)
    with lat_col1:
        st.metric(
            "Avg Latency",
            f"{latency_samples.mean():,.0f} ms",
            help="Average latency where provided in the export.",
        )
    with lat_col2:
        st.metric(
            "P95 Latency",
            f"{latency_samples.quantile(0.95):,.0f} ms",
            help="95th percentile latency (ms).",
        )
    with lat_col3:
        st.metric(
            "Latency Samples",
            f"{len(latency_samples):,}",
            help="Rows with a usable latency value.",
        )
else:
    st.caption("Latency data not found in this export.")

projection_col1, projection_col2 = st.columns(2)

with projection_col1:
    st.metric(
        "Projected Annual Usage",
        f"{projected_annual_usage:,.0f}" if pd.notna(projected_annual_usage) else "—",
        help="Average monthly credits x 12 (based on available months).",
    )
    st.markdown(
        f'<div class="metric-subtitle">Observed months: {observed_months or "—"}</div>',
        unsafe_allow_html=True,
    )

with projection_col2:
    st.metric(
        "Estimated Credits Remaining",
        f"{estimated_remaining:,.0f}" if pd.notna(estimated_remaining) else "—",
        help=f"Total pool {TOTAL_ANNUAL_CREDITS:,.0f} minus projected annual usage.",
    )
    st.markdown(
        '<div class="metric-subtitle">Total annual credits: 250,000</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="small-text">
    {selected_period_label} · 
    {total_period_credits:,.2f} total credits · 
    {len(period_slice):,} clean rows
    &nbsp;·&nbsp;
    {dropped_rows:,} rows removed from original {total_rows:,} (invalid log_id / credits / fields)
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Breakdowns ----------

st.markdown("---")
st.subheader("Breakdowns")

if visible_df.empty:
    st.warning(
        "No rows remaining after filters for this period. "
        "Try toggling embeddings back on or picking a different period."
    )
    st.stop()

model_col, user_col, agent_col = st.columns([1.2, 1.4, 1.2])
model_df = pd.DataFrame()
user_df = pd.DataFrame()
agent_df = pd.DataFrame()

# By model
with model_col:
    st.markdown("##### By Model")
    st.caption("`language_model` – calls, credits/call, and credit share")
    model_df = model_breakdown(visible_df)

    if model_df.empty:
        st.info("No model data to display.")
    else:
        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
        )

        model_share_chart = (
            alt.Chart(model_df)
            .mark_arc()
            .encode(
                theta=alt.Theta("Share (%):Q", title="Credit share (%)"),
                color=alt.Color("Model:N", title="Model"),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    alt.Tooltip("Calls:Q", format=",", title="Calls"),
                    alt.Tooltip("Credits per Call:Q", format=",.2f", title="Credits per call"),
                    alt.Tooltip("Share (%):Q", title="Credit share (%)"),
                ],
            )
            .properties(title="Model credit share")
        )
        st.altair_chart(model_share_chart, use_container_width=True)

        credits_per_call_chart = (
            alt.Chart(model_df)
            .mark_bar()
            .encode(
                y=alt.Y("Model:N", sort="-x", title="Model"),
                x=alt.X("Credits per Call:Q", title="Credits per call"),
                tooltip=[
                    alt.Tooltip("Model:N", title="Model"),
                    alt.Tooltip("Credits per Call:Q", format=",.2f", title="Credits per call"),
                    alt.Tooltip("Calls:Q", format=",", title="Calls"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    alt.Tooltip("Share (%):Q", title="Credit share (%)"),
                ],
            )
            .properties(title="Credits per call by model", height=260)
        )
        st.altair_chart(credits_per_call_chart, use_container_width=True)

# By user (email)
with user_col:
    st.markdown("##### By User (Email)")
    st.caption("`email` – credits/call, ranked by spend, with overage (>12k), burst flags (>10 msgs/hr), and rank change vs previous period")
    user_df = user_breakdown(
        visible_df,
        burst_users=burst_users,
        roles_df=roles_df,
        show_roles=roles_available,
        previous_ranks=previous_ranks,
    )

    if user_df.empty:
        st.info("No user data to display (no valid emails).")
    else:
        styled_users = style_user_dataframe(user_df)
        st.dataframe(
            styled_users,
            use_container_width=True,
            hide_index=True,
        )

# By agent
with agent_col:
    st.markdown("##### By Agent ID")
    st.caption("`agent_id` – usage & credits per agent")
    agent_df = agent_breakdown(visible_df)

    if agent_df.empty:
        st.info("No agent data to display.")
    else:
        st.dataframe(
            agent_df,
            use_container_width=True,
            hide_index=True,
        )

# Per-user aggregations for charts (selected period + filters, excluding blank emails)
user_usage_df = (
    visible_df[visible_df["email"] != "(no email)"]
    .groupby("email")
    .agg(
        Calls=("email", "size"),
        Credits=("credits", "sum"),
    )
    .reset_index()
    .rename(columns={"email": "User"})
    .sort_values("Credits", ascending=False)
)
if not user_usage_df.empty:
    user_usage_df["Credits per Call"] = user_usage_df.apply(
        lambda row: (row["Credits"] / row["Calls"]) if row["Calls"] > 0 else np.nan,
        axis=1,
    )
    user_usage_df["Credits per Call"] = user_usage_df["Credits per Call"].round(2)
    user_usage_df["Credits"] = user_usage_df["Credits"].round(2)

daily_all_df = pd.DataFrame()
if not visible_df.empty:
    tmp = visible_df.copy()
    tmp["date"] = tmp["created_at"].dt.date
    daily_all_df = (
        tmp.groupby("date")["credits"]
        .sum()
        .reset_index()
        .rename(columns={"credits": "Credits"})
    )
    daily_all_df["date"] = pd.to_datetime(daily_all_df["date"])

daily_avg_agents_df = pd.DataFrame()
agents_base = monthly_base[
    (monthly_base["agent_id"] != "(unassigned)") & (monthly_base["email"] != "(no email)")
].copy()
if not agents_base.empty:
    agents_base["date"] = agents_base["created_at"].dt.date
    daily_user_agents = (
        agents_base.groupby(["date", "email"])["agent_id"]
        .nunique()
        .reset_index()
        .rename(columns={"agent_id": "agents"})
    )
    daily_avg_agents_df = (
        daily_user_agents
        .groupby("date")
        .agg(
            avg_agents_per_user=("agents", "mean"),
            users=("email", "nunique"),
        )
        .reset_index()
        .rename(columns={"date": "Date"})
        .sort_values("Date")
    )
    daily_avg_agents_df["Date"] = pd.to_datetime(daily_avg_agents_df["Date"])

latency_by_model_df = pd.DataFrame()
daily_latency_df = pd.DataFrame()
agent_latency_summary_df = pd.DataFrame()
kb_df = pd.DataFrame()
ctx_df = pd.DataFrame()
mem_df = pd.DataFrame()
agent_model_distribution_df = pd.DataFrame()
agent_engagement_df = pd.DataFrame()
daily_avg_agents_df = pd.DataFrame()

if agents_available and not visible_enriched_df.empty:
    mapped_agents = visible_enriched_df[
        (visible_enriched_df["agent_name"] != "") &
        (visible_enriched_df["agent_id"] != "(unassigned)")
    ].copy()
    if not mapped_agents.empty:
        agent_engagement_df = (
            mapped_agents
            .groupby(["agent_id", "agent_name"], dropna=False)
            .agg(
                Calls=("agent_id", "size"),
                Credits=("credits", "sum"),
                Unique_Users=("email", lambda s: s[s != "(no email)"].nunique()),
            )
            .reset_index()
            .rename(columns={"agent_id": "Agent ID", "agent_name": "Agent", "Unique_Users": "Unique Users"})
            .sort_values("Credits", ascending=False)
        )
        agent_engagement_df["Credits"] = agent_engagement_df["Credits"].round(2)
        total_visible_credits = total_credits_visible if "total_credits_visible" in locals() else visible_enriched_df["credits"].sum()
        agent_engagement_df["Credit Share (%)"] = agent_engagement_df.apply(
            lambda row: (row["Credits"] / total_visible_credits * 100) if total_visible_credits > 0 else np.nan,
            axis=1,
        ).round(1)

# ---------- Charts & visualizations ----------

st.markdown("---")
st.subheader("Visualizations")

tab_org, tab_activity, tab_allotment, tab_user_usage, tab_latency, tab_daily_users, tab_latency_deep, tab_agent_latency, tab_agent_engagement, tab_query, tab_user_detail = st.tabs(
    ["Org Usage", "User Activity", "Credit Allotment", "User Usage", "Latency", "Daily Credits", "Latency Deep Dive", "Agent Latency", "Agent Engagement", "Query Insights", "User Detail"]
)

with tab_org:
    st.caption("Credit consumption by office, discipline, and role (requires roles data).")

    if visible_enriched_df.empty:
        st.info("No data to display.")
    elif roles_df.empty:
        st.info("Upload Prophet roles.xlsx to view office/discipline/role charts.")
    else:
        def render_org_bar(data: pd.DataFrame, category: str, title: str):
            if data.empty:
                st.info(f"No {category} data to display.")
                return
            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    y=alt.Y(f"{category}:N", sort="-x", title=category.title()),
                    x=alt.X("Credits:Q", title="Credits"),
                    tooltip=[
                        alt.Tooltip(f"{category}:N", title=category.title()),
                        alt.Tooltip("Credits:Q", format=","),
                        alt.Tooltip("Calls:Q", format=","),
                    ],
                )
                .properties(title=title, height=220)
            )
            st.altair_chart(chart, use_container_width=True)

        office_df = (
            visible_enriched_df[visible_enriched_df["office"] != ""]
            .groupby("office")
            .agg(Credits=("credits", "sum"), Calls=("credits", "size"))
            .reset_index()
            .rename(columns={"office": "Office"})
            .sort_values("Credits", ascending=False)
        )
        discipline_df = (
            visible_enriched_df[visible_enriched_df["discipline"] != ""]
            .groupby("discipline")
            .agg(Credits=("credits", "sum"), Calls=("credits", "size"))
            .reset_index()
            .rename(columns={"discipline": "Discipline"})
            .sort_values("Credits", ascending=False)
        )
        role_df = (
            visible_enriched_df[visible_enriched_df["role"] != ""]
            .groupby("role")
            .agg(Credits=("credits", "sum"), Calls=("credits", "size"))
            .reset_index()
            .rename(columns={"role": "Role"})
            .sort_values("Credits", ascending=False)
        )

        org_col1, org_col2 = st.columns(2)
        with org_col1:
            render_org_bar(office_df, "Office", "Credits by Office")
        with org_col2:
            render_org_bar(discipline_df, "Discipline", "Credits by Discipline")
        render_org_bar(role_df, "Role", "Credits by Role")

with tab_activity:
    st.caption("Average user activity month over month (filters applied).")
    if monthly_usage_df.empty:
        st.info("No monthly data to display.")
    else:
        activity_long = monthly_usage_df.melt(
            id_vars=["month_key", "month_label"],
            value_vars=["avg_calls_per_user", "avg_credits_per_user"],
            var_name="Metric",
            value_name="Value",
        )
        metric_labels = {
            "avg_calls_per_user": "Avg calls per user",
            "avg_credits_per_user": "Avg credits per user",
        }
        activity_long["Metric"] = activity_long["Metric"].map(metric_labels)

        activity_chart = (
            alt.Chart(activity_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("month_label:N", sort=monthly_usage_df["month_label"].tolist(), title="Month"),
                y=alt.Y("Value:Q", title="Average"),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("month_label:N", title="Month"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Value:Q", format=".2f"),
                ],
            )
            .properties(height=300, title="Average user activity")
        )
        st.altair_chart(activity_chart, use_container_width=True)

    st.markdown("##### Avg agents engaged per user (daily)")
    if daily_avg_agents_df.empty:
        st.info("No agent engagement data available.")
    else:
        agents_chart = (
            alt.Chart(daily_avg_agents_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("avg_agents_per_user:Q", title="Avg agents per user"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("avg_agents_per_user:Q", format=",.2f", title="Avg agents"),
                    alt.Tooltip("users:Q", format=",", title="Users"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(agents_chart, use_container_width=True)

with tab_allotment:
    st.caption("Credits consumed vs monthly and annual allotments.")
    if monthly_usage_df.empty:
        st.info("No monthly data to display.")
    else:
        month_sort = monthly_usage_df["month_label"].tolist()
        monthly_bar = (
            alt.Chart(monthly_usage_df)
            .mark_bar(color="#2563eb")
            .encode(
                x=alt.X("month_label:N", sort=month_sort, title="Month"),
                y=alt.Y("total_credits:Q", title="Monthly Credits"),
                tooltip=[
                    alt.Tooltip("month_label:N", title="Month"),
                    alt.Tooltip("total_credits:Q", format=",", title="Credits"),
                ],
            )
        )
        monthly_rule = alt.Chart(pd.DataFrame({"limit": [MONTHLY_CREDIT_LIMIT]})).mark_rule(
            color="red", strokeDash=[6, 3]
        ).encode(y="limit:Q")
        st.altair_chart(
            (monthly_bar + monthly_rule).properties(title="Monthly credits vs 12,000 limit", height=320),
            use_container_width=True,
        )

        cumulative_line = (
            alt.Chart(monthly_usage_df)
            .mark_line(point=True, color="#10b981")
            .encode(
                x=alt.X("month_label:N", sort=month_sort, title="Month"),
                y=alt.Y("cumulative_credits:Q", title="Cumulative Credits"),
                tooltip=[
                    alt.Tooltip("month_label:N", title="Month"),
                    alt.Tooltip("cumulative_credits:Q", format=",", title="Cumulative credits"),
                ],
            )
        )
        annual_rule = alt.Chart(pd.DataFrame({"limit": [TOTAL_ANNUAL_CREDITS]})).mark_rule(
            color="orange", strokeDash=[6, 3]
        ).encode(y="limit:Q")
        st.altair_chart(
            (cumulative_line + annual_rule).properties(
                title="Cumulative credits vs 250,000 annual allotment", height=320
            ),
            use_container_width=True,
        )

        st.markdown("##### Weekly credits over time")
        if weekly_usage_df.empty:
            st.info("No weekly data to display.")
        else:
            weekly_chart = (
                alt.Chart(weekly_usage_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("week_start:T", title="Week starting"),
                    y=alt.Y("total_credits:Q", title="Weekly Credits"),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("total_credits:Q", format=",", title="Credits"),
                        alt.Tooltip("total_calls:Q", format=",", title="Calls"),
                        alt.Tooltip("active_users:Q", format=",", title="Active users"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(weekly_chart, use_container_width=True)

with tab_user_usage:
    st.caption("Per-user usage for the selected period (filters applied; blank emails excluded).")
    if user_usage_df.empty:
        st.info("No user data to display.")
    else:
        top_n_users = 25

        top_calls = user_usage_df.sort_values("Calls", ascending=False).head(top_n_users)
        top_credits = user_usage_df.sort_values("Credits", ascending=False).head(top_n_users)

        calls_chart = (
            alt.Chart(top_calls)
            .mark_bar(color="#6366f1")
            .encode(
                y=alt.Y("User:N", sort="-x", title="User"),
                x=alt.X("Calls:Q", title="Calls"),
                tooltip=[
                    alt.Tooltip("User:N", title="User"),
                    alt.Tooltip("Calls:Q", title="Calls"),
                    alt.Tooltip(field="Credits per Call", type="quantitative", title="Credits per Call"),
                ],
            )
            .properties(title=f"Calls per user (top {top_n_users})", height=340)
        )

        credits_chart = (
            alt.Chart(top_credits)
            .mark_bar(color="#22c55e")
            .encode(
                y=alt.Y("User:N", sort="-x", title="User"),
                x=alt.X("Credits:Q", title="Credits"),
                tooltip=[
                    alt.Tooltip("User:N", title="User"),
                    alt.Tooltip("Credits:Q", title="Credits", format=","),
                    alt.Tooltip(field="Credits per Call", type="quantitative", title="Credits per Call"),
                ],
            )
            .properties(title=f"Credits per user (top {top_n_users})", height=340)
        )

        user_col1, user_col2 = st.columns(2)
        with user_col1:
            st.altair_chart(calls_chart, use_container_width=True)
        with user_col2:
            st.altair_chart(credits_chart, use_container_width=True)

        mean_credits = user_usage_df["Credits"].mean()
        std_credits = user_usage_df["Credits"].std()

        dist_chart = (
            alt.Chart(user_usage_df)
            .mark_bar(opacity=0.75, color="#0ea5e9")
            .encode(
                x=alt.X("Credits:Q", bin=alt.Bin(maxbins=20), title="Credits per user"),
                y=alt.Y("count():Q", title="Number of users"),
                tooltip=[
                    alt.Tooltip("count()", title="Users"),
                ],
            )
        )

        overlays = []
        if pd.notna(mean_credits):
            overlays.append(
                alt.Chart(pd.DataFrame({"value": [mean_credits], "label": ["Mean"]}))
                .mark_rule(color="red", strokeDash=[6, 3])
                .encode(
                    x="value:Q",
                    tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=",")]
                )
            )
        if pd.notna(std_credits) and std_credits > 0:
            overlays.append(
                alt.Chart(
                    pd.DataFrame(
                        {
                            "value": [mean_credits - std_credits, mean_credits + std_credits],
                            "label": ["-1σ", "+1σ"],
                        }
                    )
                )
                .mark_rule(color="#f97316", strokeDash=[4, 4])
                .encode(
                    x="value:Q",
                    tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=",")]
                )
            )

        for overlay in overlays:
            dist_chart += overlay

        st.altair_chart(
            dist_chart.properties(
                title="Distribution of credits consumed per user (mean ± 1σ)",
                height=320,
            ),
            use_container_width=True,
        )

with tab_latency:
    st.caption("Latency trends for the selected period (when latency is present in the export).")
    latency_df = visible_enriched_df.dropna(subset=["latency_ms"]) if "latency_ms" in visible_enriched_df.columns else pd.DataFrame()

    if latency_df.empty:
        st.info("No latency data available for this period.")
    else:
        latency_df = latency_df.copy()
        latency_df["date"] = latency_df["created_at"].dt.date
        daily_latency = (
            latency_df
            .groupby("date")["latency_ms"]
            .agg(
                avg_latency_ms="mean",
                p95_latency_ms=lambda s: s.quantile(0.95),
            )
            .reset_index()
        )
        daily_latency_df = daily_latency.copy()
        daily_latency["date"] = pd.to_datetime(daily_latency["date"])
        latency_long = daily_latency.melt(
            id_vars=["date"],
            value_vars=["avg_latency_ms", "p95_latency_ms"],
            var_name="Metric",
            value_name="LatencyMs",
        )
        latency_labels = {"avg_latency_ms": "Average (ms)", "p95_latency_ms": "P95 (ms)"}
        latency_long["Metric"] = latency_long["Metric"].map(latency_labels)

        latency_chart = (
            alt.Chart(latency_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("LatencyMs:Q", title="Latency (ms)"),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("LatencyMs:Q", format=",.0f", title="Latency (ms)"),
                ],
            )
            .properties(title="Latency over time", height=320)
        )
        st.altair_chart(latency_chart, use_container_width=True)

        st.markdown("##### Latency by model")
        latency_by_model = (
            latency_df
            .groupby("language_model")
            .agg(
                avg_latency_ms=("latency_ms", "mean"),
                p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
                samples=("latency_ms", "count"),
            )
            .reset_index()
            .rename(columns={"language_model": "Model"})
            .sort_values("p95_latency_ms", ascending=False)
        )
        latency_by_model_df = latency_by_model.copy()

        if latency_by_model.empty:
            st.info("No latency data by model.")
        else:
            latency_by_model_display = latency_by_model.copy()
            latency_by_model_display["avg_latency_ms"] = latency_by_model_display["avg_latency_ms"].round(0)
            latency_by_model_display["p95_latency_ms"] = latency_by_model_display["p95_latency_ms"].round(0)
            st.dataframe(
                latency_by_model_display,
                use_container_width=True,
                hide_index=True,
            )

            latency_model_long = latency_by_model.melt(
                id_vars=["Model", "samples"],
                value_vars=["avg_latency_ms", "p95_latency_ms"],
                var_name="Metric",
                value_name="LatencyMs",
            )
            latency_model_long["Metric"] = latency_model_long["Metric"].map({
                "avg_latency_ms": "Average (ms)",
                "p95_latency_ms": "P95 (ms)",
            })

            model_latency_chart = (
                alt.Chart(latency_model_long)
                .mark_bar()
                .encode(
                    y=alt.Y("Model:N", sort="-x", title="Model"),
                    x=alt.X("LatencyMs:Q", title="Latency (ms)"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Model"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("LatencyMs:Q", format=",.0f", title="Latency (ms)"),
                        alt.Tooltip("samples:Q", format=",", title="Samples"),
                    ],
                )
                .properties(title="Latency by model", height=320)
            )
            st.altair_chart(model_latency_chart, use_container_width=True)

with tab_daily_users:
    st.caption("Layered daily credit consumption by top users (filters applied).")
    if daily_all_df.empty:
        st.info("No daily credit data available.")
    else:
        daily_total_chart = (
            alt.Chart(daily_all_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Credits:Q", title="Credits"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                ],
            )
            .properties(height=260, title="Daily credits (all users)")
        )
        st.altair_chart(daily_total_chart, use_container_width=True)
    daily_usage = visible_df[visible_df["email"] != "(no email)"].copy()

    if daily_usage.empty:
        st.info("No user data available for daily breakdown.")
    else:
        daily_usage["date"] = daily_usage["created_at"].dt.date
        top_n_users = 5
        top_users = (
            daily_usage.groupby("email")["credits"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n_users)
            .index.tolist()
        )

        layered_daily = (
            daily_usage[daily_usage["email"].isin(top_users)]
            .groupby(["date", "email"])["credits"]
            .sum()
            .reset_index()
            .rename(columns={"email": "User", "credits": "Credits"})
        )
        layered_daily["date"] = pd.to_datetime(layered_daily["date"])

        base = (
            alt.Chart(layered_daily)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Credits:Q", title="Credits per day"),
                color=alt.Color("User:N", title="User", sort=top_users),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("User:N", title="User"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                ],
            )
            .properties(title=f"Top {top_n_users} users by daily credits", height=340)
        )

        layered_chart = base.mark_area(opacity=0.15) + base.mark_line(point=True, strokeWidth=2)
        st.altair_chart(layered_chart, use_container_width=True)

with tab_latency_deep:
    st.caption("Latency deep dive: time-of-day heatmap, model trends by day/month/year, and office breakdown (requires roles).")
    latency_df = (
        visible_enriched_df.dropna(subset=["latency_ms"])
        if "latency_ms" in visible_enriched_df.columns
        else pd.DataFrame()
    )

    if latency_df.empty:
        st.info("No latency data available for this period.")
    else:
        latency_df = latency_df.copy()
        latency_df["date"] = latency_df["created_at"].dt.date
        latency_df["date_ts"] = pd.to_datetime(latency_df["date"])
        latency_df["hour"] = latency_df["created_at"].dt.hour
        latency_df["month_start"] = latency_df["created_at"].dt.to_period("M").apply(lambda p: p.start_time)
        latency_df["year"] = latency_df["created_at"].dt.year
        latency_df["Model"] = latency_df["language_model"]

        # Daily heatmap by hour
        st.markdown("##### Latency by day and hour")
        hourly = (
            latency_df
            .groupby(["date_ts", "hour"])["latency_ms"]
            .agg(
                avg_latency_ms="mean",
                p95_latency_ms=lambda s: s.quantile(0.95),
                samples="count",
            )
            .reset_index()
        )
        heatmap = (
            alt.Chart(hourly)
            .mark_rect()
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("date_ts:T", title="Date"),
                color=alt.Color("avg_latency_ms:Q", title="Avg latency (ms)", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("date_ts:T", title="Date"),
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                    alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                    alt.Tooltip("samples:Q", format=",", title="Samples"),
                ],
            )
            .properties(height=320, title="Avg latency heatmap")
        )
        st.altair_chart(heatmap, use_container_width=True)

        # Helper for model/time aggregation
        def model_timeframe(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
            return (
                df
                .groupby([period_col, "Model"])
                .agg(
                    avg_latency_ms=("latency_ms", "mean"),
                    p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
                    samples=("latency_ms", "count"),
                )
                .reset_index()
            )

        def model_metric_chart(frame: pd.DataFrame, x_field: str, x_title: str, metric: str, metric_title: str, x_type: str):
            if frame.empty:
                st.info(f"No latency by model for {x_title.lower()}.")
                return None
            return (
                alt.Chart(frame)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{x_field}:{x_type}", title=x_title),
                    y=alt.Y(f"{metric}:Q", title=metric_title),
                    color=alt.Color("Model:N", title="Model"),
                    tooltip=[
                        alt.Tooltip(x_field + (":T" if x_type == "T" else ":O"), title=x_title),
                        alt.Tooltip("Model:N", title="Model"),
                        alt.Tooltip(f"{metric}:Q", format=",.0f", title=metric_title),
                        alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                        alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                        alt.Tooltip("samples:Q", format=",", title="Samples"),
                    ],
                )
                .properties(height=260)
            )

        st.markdown("##### Latency by model over time")
        daily_model = model_timeframe(latency_df, "date_ts")
        monthly_model = model_timeframe(latency_df, "month_start")
        yearly_model = model_timeframe(latency_df, "year")

        day_p95_chart = model_metric_chart(daily_model, "date_ts", "Date", "p95_latency_ms", "P95 latency (ms)", "T")
        month_p95_chart = model_metric_chart(monthly_model, "month_start", "Month", "p95_latency_ms", "P95 latency (ms)", "T")
        year_p95_chart = model_metric_chart(yearly_model, "year", "Year", "p95_latency_ms", "P95 latency (ms)", "O")

        day_avg_chart = model_metric_chart(daily_model, "date_ts", "Date", "avg_latency_ms", "Avg latency (ms)", "T")
        month_avg_chart = model_metric_chart(monthly_model, "month_start", "Month", "avg_latency_ms", "Avg latency (ms)", "T")
        year_avg_chart = model_metric_chart(yearly_model, "year", "Year", "avg_latency_ms", "Avg latency (ms)", "O")

        model_col1, model_col2 = st.columns(2)
        with model_col1:
            if day_p95_chart is not None:
                st.altair_chart(day_p95_chart.properties(title="P95 latency by model (daily)"), use_container_width=True)
            if day_avg_chart is not None:
                st.altair_chart(day_avg_chart.properties(title="Average latency by model (daily)"), use_container_width=True)
        with model_col2:
            if month_p95_chart is not None:
                st.altair_chart(month_p95_chart.properties(title="P95 latency by model (monthly)"), use_container_width=True)
            if month_avg_chart is not None:
                st.altair_chart(month_avg_chart.properties(title="Average latency by model (monthly)"), use_container_width=True)
            if year_p95_chart is not None:
                st.altair_chart(year_p95_chart.properties(title="P95 latency by model (yearly)"), use_container_width=True)
            if year_avg_chart is not None:
                st.altair_chart(year_avg_chart.properties(title="Average latency by model (yearly)"), use_container_width=True)

        # Latency by office (requires roles)
        st.markdown("##### Latency by office (requires roles)")
        office_latency = latency_df[latency_df["office"] != ""]
        if office_latency.empty:
            st.info("Upload Prophet roles.xlsx to see office-level latency.")
        else:
            office_summary = (
                office_latency
                .groupby("office")["latency_ms"]
                .agg(
                    avg_latency_ms="mean",
                    p95_latency_ms=lambda s: s.quantile(0.95),
                    samples="count",
                )
                .reset_index()
                .rename(columns={"office": "Office"})
                .sort_values("p95_latency_ms", ascending=False)
            )
            office_chart = (
                alt.Chart(office_summary)
                .mark_bar()
                .encode(
                    y=alt.Y("Office:N", sort="-x", title="Office"),
                    x=alt.X("p95_latency_ms:Q", title="P95 latency (ms)"),
                    tooltip=[
                        alt.Tooltip("Office:N", title="Office"),
                        alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                        alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                        alt.Tooltip("samples:Q", format=",", title="Samples"),
                    ],
                )
                .properties(height=320, title="Latency by office (P95)")
            )
            st.altair_chart(office_chart, use_container_width=True)

with tab_agent_latency:
    st.caption("Latency by agent features (only agents mapped from AgentID Names.json).")
    if not agents_available:
        st.info("Upload AgentID Names.json to view agent-level latency.")
    else:
        if "latency_ms" not in visible_enriched_df.columns:
            st.info("Latency data not found in this export.")
        else:
            agent_latency = visible_enriched_df.dropna(subset=["latency_ms"]).copy()
            agent_latency = agent_latency[agent_latency["agent_name"] != ""]

            if agent_latency.empty:
                st.info("No latency rows for agents present in AgentID Names.json.")
            else:
                agent_latency["latency_ms"] = pd.to_numeric(agent_latency["latency_ms"], errors="coerce")
                agent_latency = agent_latency.dropna(subset=["latency_ms"])

                agent_summary = (
                    agent_latency
                    .groupby(["agent_name", "agent_id", "has_kb", "uses_context", "uses_memory"])["latency_ms"]
                    .agg(
                        avg_latency_ms="mean",
                        p95_latency_ms=lambda s: s.quantile(0.95),
                        samples="count",
                    )
                    .reset_index()
                    .sort_values("p95_latency_ms", ascending=False)
                )
                agent_latency_summary_df = agent_summary.copy()
                agent_summary_display = agent_summary.copy()
                for col in ("avg_latency_ms", "p95_latency_ms"):
                    agent_summary_display[col] = agent_summary_display[col].round(0)

                st.markdown("##### Latency by mapped agent")
                st.dataframe(
                    agent_summary_display.rename(columns={"agent_name": "Agent", "agent_id": "Agent ID"}),
                    use_container_width=True,
                    hide_index=True,
                )

                # Agent latency chart
                agent_chart_data = agent_summary_display.rename(columns={"agent_name": "Agent"})
                agent_latency_chart = (
                    alt.Chart(agent_chart_data)
                    .mark_bar()
                    .encode(
                        y=alt.Y("Agent:N", sort="-x", title="Agent"),
                        x=alt.X("p95_latency_ms:Q", title="P95 latency (ms)"),
                        tooltip=[
                            alt.Tooltip("Agent:N", title="Agent"),
                            alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                            alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                            alt.Tooltip("samples:Q", format=",", title="Samples"),
                        ],
                    )
                    .properties(height=320, title="P95 latency by agent")
                )
                st.altair_chart(agent_latency_chart, use_container_width=True)

                def summarize_flag(df: pd.DataFrame, flag_col: str, label: str) -> pd.DataFrame:
                    grouped = (
                        df.groupby(flag_col)["latency_ms"]
                        .agg(
                            avg_latency_ms="mean",
                            p95_latency_ms=lambda s: s.quantile(0.95),
                            samples="count",
                        )
                        .reset_index()
                    )
                    grouped[flag_col] = grouped[flag_col].map({True: f"{label}: Yes", False: f"{label}: No"})
                    for col in ("avg_latency_ms", "p95_latency_ms"):
                        grouped[col] = grouped[col].round(0)
                    return grouped

                def render_flag_chart(df: pd.DataFrame, x_label: str):
                    if df.empty:
                        st.info(f"No latency data for {x_label.lower()}.")
                        return
                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(
                            y=alt.Y(df.columns[0] + ":N", sort="-x", title=x_label),
                            x=alt.X("p95_latency_ms:Q", title="P95 latency (ms)"),
                            tooltip=[
                                alt.Tooltip(df.columns[0] + ":N", title=x_label),
                                alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                                alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                                alt.Tooltip("samples:Q", format=",", title="Samples"),
                            ],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)

                flag_col1, flag_col2, flag_col3 = st.columns(3)
                kb_df = summarize_flag(agent_latency, "has_kb", "Knowledge Base")
                ctx_df = summarize_flag(agent_latency, "uses_context", "Context")
                mem_df = summarize_flag(agent_latency, "uses_memory", "Memory")

                with flag_col1:
                    st.markdown("###### Knowledge base")
                    render_flag_chart(kb_df, "Knowledge base")
                with flag_col2:
                    st.markdown("###### Context")
                    render_flag_chart(ctx_df, "Context")
                with flag_col3:
                    st.markdown("###### Memory")
                    render_flag_chart(mem_df, "Memory")

                # Time-of-day latency (hourly)
                st.markdown("##### Hour-of-day latency (all mapped agents)")
                agent_latency["hour"] = agent_latency["created_at"].dt.hour
                hourly_latency = (
                    agent_latency.groupby("hour")["latency_ms"]
                    .agg(
                        avg_latency_ms="mean",
                        p95_latency_ms=lambda s: s.quantile(0.95),
                        samples="count",
                    )
                    .reset_index()
                    .sort_values("hour")
                )
                hourly_chart = (
                    alt.Chart(hourly_latency)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("hour:O", title="Hour of day"),
                        y=alt.Y("p95_latency_ms:Q", title="P95 latency (ms)"),
                        tooltip=[
                            alt.Tooltip("hour:O", title="Hour"),
                            alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                            alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                            alt.Tooltip("samples:Q", format=",", title="Samples"),
                        ],
                    )
                    .properties(height=240, title="P95 latency by hour")
                )
                st.altair_chart(hourly_chart, use_container_width=True)

                # Minute-level granularity (rounded to minute; capped for chart size)
                st.markdown("##### Minute-level latency (rounded to minute)")
                agent_latency["minute_key"] = agent_latency["created_at"].dt.floor("T")
                minute_latency = (
                    agent_latency.groupby("minute_key")["latency_ms"]
                    .agg(
                        avg_latency_ms="mean",
                        p95_latency_ms=lambda s: s.quantile(0.95),
                        samples="count",
                    )
                    .reset_index()
                    .sort_values("minute_key")
                )
                minute_latency_cap = minute_latency.tail(500)  # avoid over-plotting
                minute_chart = (
                    alt.Chart(minute_latency_cap)
                    .mark_line()
                    .encode(
                        x=alt.X("minute_key:T", title="Time (minute)"),
                        y=alt.Y("p95_latency_ms:Q", title="P95 latency (ms)"),
                        tooltip=[
                            alt.Tooltip("minute_key:T", title="Minute"),
                            alt.Tooltip("p95_latency_ms:Q", format=",.0f", title="P95 latency (ms)"),
                            alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                            alt.Tooltip("samples:Q", format=",", title="Samples"),
                        ],
                    )
                    .properties(height=240, title="Recent 500 minutes (P95)")
                )
                st.altair_chart(minute_chart, use_container_width=True)

                st.markdown("##### Model distribution across mapped agents")
                model_agents = visible_enriched_df[visible_enriched_df["agent_name"] != ""].copy()
                model_agents = model_agents[model_agents["language_model"].notna()]
                if model_agents.empty:
                    st.info("No mapped agents with model data in this period.")
                else:
                    unique_agent_model = model_agents[["agent_id", "agent_name", "language_model"]].drop_duplicates()
                    total_mapped_agents = unique_agent_model["agent_id"].nunique()
                    model_distribution = (
                        unique_agent_model
                        .groupby("language_model")["agent_id"]
                        .nunique()
                        .reset_index()
                        .rename(columns={"language_model": "Model", "agent_id": "Agents"})
                        .sort_values("Agents", ascending=False)
                    )
                    model_distribution["Agent Share (%)"] = (
                        model_distribution["Agents"] / total_mapped_agents * 100
                    ).round(1)
                    agent_model_distribution_df = model_distribution.copy()

                    st.dataframe(model_distribution, use_container_width=True, hide_index=True)

                    dist_chart = (
                        alt.Chart(model_distribution)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Model:N", sort="-x", title="Model"),
                            x=alt.X("Agent Share (%):Q", title="Share of mapped agents (%)"),
                            tooltip=[
                                alt.Tooltip("Model:N", title="Model"),
                                alt.Tooltip("Agents:Q", format=",", title="Agents using model"),
                                alt.Tooltip("Agent Share (%):Q", title="Share (%)"),
                            ],
                        )
                        .properties(height=300, title="Mapped agents by model")
                    )
                    st.altair_chart(dist_chart, use_container_width=True)

with tab_agent_engagement:
    st.caption("Mapped agents (from AgentID Names.json): calls, credits, and unique users.")
    if not agents_available:
        st.info("Upload AgentID Names.json to view mapped agent engagement.")
    elif agent_engagement_df.empty:
        st.info("No mapped agents found in this period.")
    else:
        st.dataframe(
            agent_engagement_df,
            use_container_width=True,
            hide_index=True,
        )

        engagement_chart = (
            alt.Chart(agent_engagement_df.head(30))
            .mark_bar()
            .encode(
                y=alt.Y("Agent:N", sort="-x", title="Agent"),
                x=alt.X("Credits:Q", title="Credits"),
                tooltip=[
                    alt.Tooltip("Agent:N", title="Agent"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    alt.Tooltip("Calls:Q", format=",", title="Calls"),
                    alt.Tooltip("Unique Users:Q", format=",", title="Unique users"),
                    alt.Tooltip("Credit Share (%):Q", title="Credit share (%)"),
                ],
            )
            .properties(height=320, title="Credits by mapped agent (top 30)")
        )
        st.altair_chart(engagement_chart, use_container_width=True)

        users_chart = (
            alt.Chart(agent_engagement_df.head(30))
            .mark_bar(color="#f59e0b")
            .encode(
                y=alt.Y("Agent:N", sort="-x", title="Agent"),
                x=alt.X("Unique Users:Q", title="Unique users"),
                tooltip=[
                    alt.Tooltip("Agent:N", title="Agent"),
                    alt.Tooltip("Unique Users:Q", format=",", title="Unique users"),
                    alt.Tooltip("Calls:Q", format=",", title="Calls"),
                    alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    alt.Tooltip("Credit Share (%):Q", title="Credit share (%)"),
                ],
            )
            .properties(height=320, title="Unique users by mapped agent (top 30)")
        )
        st.altair_chart(users_chart, use_container_width=True)

with tab_query:
    st.caption(
        "Insights from input/output messages (e.g., `input_messages`, `output_message`) and chat sessions (`session_id`)."
    )
    if not query_insights.get("available"):
        st.info(query_insights.get("reason") or "Message and session columns not found in this export.")
    else:
        summary = query_insights.get("summary", {})

        def fmt_avg(val):
            return f"{val:,.1f}" if pd.notna(val) else "—"

        qc1, qc2, qc3, qc4 = st.columns(4)
        with qc1:
            st.metric("Avg input length (words)", fmt_avg(summary.get("avg_input_words")))
        with qc2:
            st.metric("Avg output length (words)", fmt_avg(summary.get("avg_output_words")))
        with qc3:
            st.metric("Avg tool calls per response", fmt_avg(summary.get("avg_tool_calls")))
        with qc4:
            st.metric("Avg chats per session", fmt_avg(summary.get("avg_chats_per_session")))

        st.markdown("##### Query length over time")
        over_time_df = query_insights.get("over_time", pd.DataFrame())
        if over_time_df.empty:
            st.info("No time-based query data available.")
        else:
            over_time_long = over_time_df.melt(
                id_vars=["Date"],
                value_vars=["avg_input_words", "avg_output_words"],
                var_name="Metric",
                value_name="Value",
            )
            metric_labels = {
                "avg_input_words": "Avg input length",
                "avg_output_words": "Avg output length",
            }
            over_time_long["Metric"] = over_time_long["Metric"].map(metric_labels)
            time_chart = (
                alt.Chart(over_time_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Value:Q", title="Words"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Value:Q", format=",.1f", title="Words"),
                    ],
                )
                .properties(height=280, title="Average query length (words)")
            )
            st.altair_chart(time_chart, use_container_width=True)

        st.markdown("##### Average query length by user")
        per_user_df = query_insights.get("per_user", pd.DataFrame())
        if per_user_df.empty:
            st.info("No user-level query data available.")
        else:
            display_user = per_user_df.copy()
            for col in ("avg_input_words", "avg_output_words", "avg_tool_calls"):
                display_user[col] = display_user[col].round(1)
            st.dataframe(
                display_user.head(50).rename(columns={
                    "avg_input_words": "Avg Input (words)",
                    "avg_output_words": "Avg Output (words)",
                    "avg_tool_calls": "Avg Tool Calls",
                    "messages": "Messages",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("##### Tool calls per response")
        tool_dist = query_insights.get("tool_calls_distribution", pd.DataFrame())
        if tool_dist.empty:
            st.info("No tool call data available in message payloads.")
        else:
            tool_chart = (
                alt.Chart(tool_dist)
                .mark_bar()
                .encode(
                    x=alt.X("Tool Calls:O", title="Tool calls per assistant message"),
                    y=alt.Y("Count:Q", title="Responses"),
                    tooltip=[
                        alt.Tooltip("Tool Calls:O", title="Tool calls"),
                        alt.Tooltip("Count:Q", format=",", title="Responses"),
                    ],
                )
                .properties(height=260, title="Distribution of tool calls")
            )
            st.altair_chart(tool_chart, use_container_width=True)

        st.markdown("##### Avg latency vs avg input length")
        latency_input_df = query_insights.get("latency_vs_input", pd.DataFrame())
        if latency_input_df.empty:
            st.info("No latency/input overlap found to chart.")
        else:
            scatter = (
                alt.Chart(latency_input_df)
                .mark_circle(size=80)
                .encode(
                    x=alt.X("avg_input_words:Q", title="Avg input length (words)"),
                    y=alt.Y("avg_latency_ms:Q", title="Avg latency (ms)"),
                    color=alt.Color("Date:T", title="Date"),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("avg_input_words:Q", format=",.1f", title="Avg input (words)"),
                        alt.Tooltip("avg_latency_ms:Q", format=",.0f", title="Avg latency (ms)"),
                    ],
                )
                .properties(height=280, title="Avg latency vs avg input length (by day)")
            )
            st.altair_chart(scatter, use_container_width=True)

        st.markdown("##### Chats per session")
        session_counts = query_insights.get("session_counts", pd.DataFrame())
        session_over_time = query_insights.get("session_over_time", pd.DataFrame())

        if session_counts.empty:
            st.info("No session_id column found or no sessions detected.")
        else:
            session_display = session_counts.copy()
            session_display["Chats"] = session_display["Chats"].astype(int)
            st.dataframe(
                session_display[["Session ID", "User", "Chats", "Start Date"]].head(50),
                use_container_width=True,
                hide_index=True,
            )

            if session_over_time.empty:
                st.info("No session timeline available.")
            else:
                session_chart = (
                    alt.Chart(session_over_time)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Date:T", title="Session start date"),
                        y=alt.Y("avg_chats_per_session:Q", title="Avg chats per session"),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("avg_chats_per_session:Q", format=",.1f", title="Avg chats"),
                            alt.Tooltip("sessions:Q", format=",", title="Sessions"),
                        ],
                    )
                    .properties(height=280, title="Chats per session over time")
                )
                st.altair_chart(session_chart, use_container_width=True)

with tab_user_detail:
    st.caption("Drill into a single user: models used, daily credits, hour-of-day patterns, and agent mix.")
    if user_usage_df.empty:
        st.info("No user data available for this period.")
    else:
        user_options = user_usage_df["User"].tolist()
        selected_user = st.selectbox(
            "Select user",
            options=user_options,
            index=0,
        )

        user_df = visible_enriched_df[
            visible_enriched_df["email"].str.lower() == selected_user.strip().lower()
        ].copy()

        if user_df.empty:
            st.info("No rows found for this user in the selected period.")
        else:
            total_credits_user = user_df["credits"].sum()
            total_calls_user = len(user_df)
            credits_per_call_user = (total_credits_user / total_calls_user) if total_calls_user > 0 else np.nan
            agent_count_user = user_df[user_df["agent_id"] != "(unassigned)"]["agent_id"].nunique()

            user_latency = user_df["latency_ms"].dropna() if "latency_ms" in user_df.columns else pd.Series(dtype="float")
            col_u1, col_u2, col_u3, col_u4, col_u5 = st.columns(5)
            with col_u1:
                st.metric("Credits", f"{total_credits_user:,.2f}")
            with col_u2:
                st.metric("Calls", f"{total_calls_user:,}")
            with col_u3:
                st.metric("Credits per Call", f"{credits_per_call_user:,.2f}" if pd.notna(credits_per_call_user) else "—")
            with col_u4:
                st.metric("Agents Engaged", f"{agent_count_user:,}")
            with col_u5:
                if user_latency.empty:
                    st.metric("P95 Latency", "—")
                else:
                    st.metric("P95 Latency", f"{user_latency.quantile(0.95):,.0f} ms")

            # Models used
            model_usage = (
                user_df.groupby("language_model")
                .agg(
                    Calls=("language_model", "size"),
                    Credits=("credits", "sum"),
                    Avg_Latency_ms=("latency_ms", "mean"),
                )
                .reset_index()
                .rename(columns={"language_model": "Model"})
                .sort_values("Credits", ascending=False)
            )
            model_usage["Credits"] = model_usage["Credits"].round(2)
            if "Avg_Latency_ms" in model_usage.columns:
                model_usage["Avg_Latency_ms"] = model_usage["Avg_Latency_ms"].round(0)

            st.markdown("##### Models used")
            st.dataframe(model_usage, use_container_width=True, hide_index=True)

            # Daily credits
            st.markdown("##### Daily credits")
            user_df["date"] = user_df["created_at"].dt.date
            daily_user = (
                user_df.groupby("date")["credits"]
                .sum()
                .reset_index()
                .rename(columns={"credits": "Credits"})
            )
            daily_user["date"] = pd.to_datetime(daily_user["date"])
            daily_chart = (
                alt.Chart(daily_user)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Credits:Q", title="Credits"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    ],
                )
                .properties(height=260, title="Daily credits")
            )
            st.altair_chart(daily_chart, use_container_width=True)

            # Hour-of-day habits
            st.markdown("##### Hour-of-day pattern")
            user_df["hour"] = user_df["created_at"].dt.hour
            hour_usage = (
                user_df.groupby("hour")["credits"]
                .sum()
                .reset_index()
                .rename(columns={"credits": "Credits"})
            )
            hour_chart = (
                alt.Chart(hour_usage)
                .mark_bar()
                .encode(
                    x=alt.X("hour:O", title="Hour of day"),
                    y=alt.Y("Credits:Q", title="Credits"),
                    tooltip=[
                        alt.Tooltip("hour:O", title="Hour"),
                        alt.Tooltip("Credits:Q", format=",", title="Credits"),
                    ],
                )
                .properties(height=200, title="Credits by hour")
            )
            st.altair_chart(hour_chart, use_container_width=True)

            # Call type / embeddings share
            st.markdown("##### Call type mix")
            call_mix = (
                user_df.groupby("call_type")["credits"]
                .sum()
                .reset_index()
                .rename(columns={"call_type": "Call Type", "credits": "Credits"})
                .sort_values("Credits", ascending=False)
            )
            call_mix["Credits"] = call_mix["Credits"].round(2)
            st.dataframe(call_mix, use_container_width=True, hide_index=True)

            # Agent usage when available
            if "agent_id" in user_df.columns:
                agent_usage = (
                    user_df.groupby(["agent_id", "agent_name"])
                    .agg(
                        Calls=("agent_id", "size"),
                        Credits=("credits", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"agent_id": "Agent ID", "agent_name": "Agent"})
                    .sort_values("Credits", ascending=False)
                )
                agent_usage["Credits"] = agent_usage["Credits"].round(2)
                st.markdown("##### Agent usage (if mapped)")
                st.dataframe(agent_usage, use_container_width=True, hide_index=True)

export_sheets = {
    "model_breakdown": model_df,
    "user_breakdown": user_df,
    "agent_breakdown": agent_df,
    "daily_credits": daily_all_df,
    "monthly_usage": monthly_usage_df,
    "weekly_usage": weekly_usage_df,
    "avg_agents_per_user_daily": daily_avg_agents_df,
    "latency_daily": daily_latency_df,
    "latency_by_model": latency_by_model_df,
    "agent_latency": agent_latency_summary_df,
    "agent_latency_kb": kb_df,
    "agent_latency_context": ctx_df,
    "agent_latency_memory": mem_df,
    "agent_model_distribution": agent_model_distribution_df,
    "agent_engagement": agent_engagement_df,
    "user_usage": user_usage_df,
    "query_rows": query_insights.get("rows", pd.DataFrame()).head(RAW_EXPORT_LIMIT),
    "query_per_user": query_insights.get("per_user", pd.DataFrame()),
    "query_over_time": query_insights.get("over_time", pd.DataFrame()),
    "query_tool_calls": query_insights.get("tool_calls_distribution", pd.DataFrame()),
    "query_latency_vs_input": query_insights.get("latency_vs_input", pd.DataFrame()),
    "session_chat_counts": query_insights.get("session_counts", pd.DataFrame()),
    "session_over_time": query_insights.get("session_over_time", pd.DataFrame()),
}
if not visible_df.empty:
    export_sheets["filtered_rows"] = visible_df.head(RAW_EXPORT_LIMIT)

export_data_available = any(df is not None and not df.empty for df in export_sheets.values())
if export_data_available:
    excel_bytes = export_to_excel(export_sheets)
    st.download_button(
        "📥 Download Excel (all views)",
        data=excel_bytes,
        file_name=f"credit_dashboard_{selected_period}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Exports key tables for this period. Raw rows capped at 50k to keep file size reasonable.",
    )
else:
    st.caption("Nothing to export yet; load data first.")

st.markdown("---")
st.markdown(
    """
    <div class="small-text">
    Cleaning rules: rows are excluded if <code>log_id</code> is missing/short,
    <code>actions</code> is non-numeric or ≤ 0, 
    <code>call_type</code> or <code>language_model</code> are blank,
    or <code>created_at</code> is invalid.
    </div>
    """,
    unsafe_allow_html=True,
)
