from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TOTAL_ANNUAL_CREDITS = 250_000
MONTHLY_CREDIT_LIMIT = 12_000
BURST_THRESHOLD = 10

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

    # 6) Email normalization (avoid NaN showing up)
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    bad_email_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    df.loc[df["email"].str.lower().isin(bad_email_values), "email"] = "(no email)"

    # 7) Agent ID normalization
    df["agent_id"] = df["agent_id"].astype(str).str.strip()
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
) -> pd.DataFrame:
    if df.empty:
        return df

    burst_users = {u.strip().lower() for u in (burst_users or set())}
    roles_df = roles_df if roles_df is not None else pd.DataFrame()

    # Exclude rows with "(no email)" from the ranking, but keep them in totals
    filtered = df[df["email"] != "(no email)"].copy()
    if filtered.empty:
        return pd.DataFrame()

    grouped = (
        filtered.groupby("email", dropna=False)
                .agg(
                    Calls=("email", "size"),
                    Credits=("credits", "sum"),
                )
                .reset_index()
                .rename(columns={"email": "User"})
    )
    grouped["Credits"] = grouped["Credits"].round(2)
    grouped["Credits per Call"] = grouped.apply(
        lambda row: (row["Credits"] / row["Calls"]) if row["Calls"] > 0 else np.nan,
        axis=1,
    )
    grouped["Credits per Call"] = grouped["Credits per Call"].round(2)
    grouped = grouped.sort_values("Credits", ascending=False)
    grouped.insert(0, "#", range(1, len(grouped) + 1))

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

        ordered_columns = ["#", "User", "Role", "Discipline", "Office", "Calls", "Credits", "Credits per Call", "Over 12k Credits", "Burst Detected", "Flags"]
    else:
        ordered_columns = ["#", "User", "Calls", "Credits", "Credits per Call", "Over 12k Credits", "Burst Detected", "Flags"]

    return grouped[ordered_columns]


def agent_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    grouped = (
        df.groupby("agent_id", dropna=False)
          .agg(
              Calls=("agent_id", "size"),
              Credits=("credits", "sum"),
          )
          .reset_index()
          .rename(columns={"agent_id": "Agent ID"})
    )
    grouped["Credits"] = grouped["Credits"].round(2)
    grouped = grouped.sort_values("Credits", ascending=False)
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

raw_df = pd.read_csv(uploaded_file)
clean_df = clean_dataframe(raw_df)

total_rows = len(raw_df)
clean_rows = len(clean_df)
dropped_rows = total_rows - clean_rows

if clean_df.empty:
    st.error("No valid rows after cleaning. Check that the CSV uses the expected columns.")
    st.stop()

roles_df = load_roles_mapping(roles_file)
roles_available = show_roles and not roles_df.empty
if show_roles and roles_df.empty:
    st.info("Prophet roles.xlsx not provided or unreadable. Role enrichment is skipped.")

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

# Month choices
available_months = sorted(clean_df["month_key"].dropna().unique())
month_labels = {
    month: pd.Period(month).to_timestamp().strftime("%B %Y") for month in available_months
}

selected_month = st.selectbox(
    "Month",
    options=available_months,
    format_func=lambda m: month_labels.get(m, m),
    index=len(available_months) - 1 if available_months else 0,
)

month_slice = clean_df[clean_df["month_key"] == selected_month].copy()

# Apply embeddings toggle
if include_embeddings:
    visible_df = month_slice.copy()
else:
    visible_df = month_slice[month_slice["call_type"] != "aembedding"].copy()

burst_users = detect_burst_users(visible_df, threshold=BURST_THRESHOLD)
visible_enriched_df = enrich_with_roles(visible_df, roles_df)

# Embedding share is *always* computed on the full month
embeddings_df = month_slice[month_slice["call_type"] == "aembedding"]
total_month_credits = month_slice["credits"].sum()
embedding_credits = embeddings_df["credits"].sum()
embedding_share = (embedding_credits / total_month_credits) if total_month_credits > 0 else np.nan

# ---------- Top metrics ----------

total_credits_visible = visible_df["credits"].sum()
total_calls_visible = len(visible_df)
active_users = visible_df[visible_df["email"] != "(no email)"]["email"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Credits (Selected Month)",
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
        help="Share of total monthly credits attributed to call_type = aembedding."
    )
    st.markdown(
        '<div class="metric-subtitle">Computed on full month, regardless of toggle</div>',
        unsafe_allow_html=True,
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
    {month_labels.get(selected_month, selected_month)} · 
    {total_month_credits:,.2f} total credits · 
    {len(month_slice):,} clean rows
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
        "No rows remaining after filters for this month. "
        "Try toggling embeddings back on or picking a different month."
    )
    st.stop()

model_col, user_col, agent_col = st.columns([1.2, 1.4, 1.2])

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

# By user (email)
with user_col:
    st.markdown("##### By User (Email)")
    st.caption("`email` – credits/call, ranked by spend, with overage (>12k) & burst flags (>10 msgs/hr)")
    user_df = user_breakdown(
        visible_df,
        burst_users=burst_users,
        roles_df=roles_df,
        show_roles=roles_available,
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

# Per-user aggregations for charts (selected month + filters, excluding blank emails)
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

# ---------- Charts & visualizations ----------

st.markdown("---")
st.subheader("Visualizations")

tab_org, tab_activity, tab_allotment, tab_user_usage, tab_latency, tab_daily_users = st.tabs(
    ["Org Usage", "User Activity", "Credit Allotment", "User Usage", "Latency", "Daily Credits"]
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

with tab_user_usage:
    st.caption("Per-user usage for the selected month (filters applied; blank emails excluded).")
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
    st.caption("Latency trends for the selected month (when latency is present in the export).")
    latency_df = visible_df.dropna(subset=["latency_ms"]) if "latency_ms" in visible_df.columns else pd.DataFrame()

    if latency_df.empty:
        st.info("No latency data available for this month.")
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
