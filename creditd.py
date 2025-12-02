from pathlib import Path

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
    df["email"] = df["email"].astype(str).str.strip()
    bad_email_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    df.loc[df["email"].str.lower().isin(bad_email_values), "email"] = "(no email)"

    # 7) Agent ID normalization
    df["agent_id"] = df["agent_id"].astype(str).str.strip()
    bad_agent_values = {"", "nan", "NaN", "none", "None", "null", "Null"}
    df.loc[df["agent_id"].str.lower().isin(bad_agent_values), "agent_id"] = "(unassigned)"

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
    roles_path = Path("prophet roles.xlsx")
    source = uploaded_roles if uploaded_roles is not None else (roles_path if roles_path.exists() else None)
    if source is None:
        return pd.DataFrame()

    try:
        roles_df = pd.read_excel(source)
    except Exception as exc:  # pragma: no cover - UI warning
        st.warning(f"Could not read Prophet roles workbook: {exc}")
        return pd.DataFrame()

    roles_df = roles_df.rename(columns={c: c.strip().lower() for c in roles_df.columns})
    if "email" not in roles_df.columns:
        st.warning("Prophet roles workbook is missing an 'email' column; skipping enrichment.")
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
    total_credits = grouped["Credits"].sum()
    if total_credits > 0:
        grouped["Share (%)"] = (grouped["Credits"] / total_credits * 100).round(1)
    else:
        grouped["Share (%)"] = np.nan

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

        ordered_columns = ["#", "User", "Role", "Discipline", "Office", "Calls", "Credits", "Over 12k Credits", "Burst Detected", "Flags"]
    else:
        ordered_columns = ["#", "User", "Calls", "Credits", "Over 12k Credits", "Burst Detected", "Flags"]

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

    roles_file = st.file_uploader(
        "Upload Prophet roles.xlsx (optional)",
        type=["xlsx"],
        help="Adds role, discipline, and office columns by matching email.",
    )
    roles_toggle_default = roles_file is not None or Path("prophet roles.xlsx").exists()
    show_roles = (
        st.checkbox(
            "Show role/discipline/office (if available)",
            value=roles_toggle_default,
            help="Requires Prophet roles.xlsx (uploaded or in the app folder).",
        )
        if roles_toggle_default
        else False
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

# Monthly consumption pattern (for projections)
monthly_totals = clean_df.groupby("month_key")["credits"].sum()
observed_months = len(monthly_totals)
avg_monthly_usage = monthly_totals.mean() if observed_months > 0 else np.nan
projected_annual_usage = avg_monthly_usage * 12 if observed_months > 0 else np.nan
estimated_remaining = (
    TOTAL_ANNUAL_CREDITS - projected_annual_usage if pd.notna(projected_annual_usage) else np.nan
)

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
    st.caption("`language_model` – volume & credit share")
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
    st.caption("`email` – ranked by total credits, with overage (>12k) & burst flags (>10 msgs/hr)")
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
