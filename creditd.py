import streamlit as st
import pandas as pd
import numpy as np

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


def user_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

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
    return grouped


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
    from your monthly export, with credits sourced from `cost_in_credits`
    and only rows with a proper `Id` included.
    """
)

with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload CSV export",
        type=["csv"],
        help="Uses columns: Id, agent_id, call_type, language_model, email, cost_in_credits, created_at"
    )

    include_embeddings = st.checkbox(
        "Include embeddings (call_type = aembedding)",
        value=True
    )

    st.markdown(
        """
        <div class="small-text">
        Rows are dropped if: invalid/missing <code>Id</code>, 
        <code>credits</code> ≤ 0, missing <code>call_type</code> /
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
        '<div class="metric-subtitle">From cost_in_credits, after filters</div>',
        unsafe_allow_html=True,
    )

with col2:
    st.metric(
        "Total Calls (Rows)",
        f"{total_calls_visible:,.0f}" if total_calls_visible > 0 else "—",
    )
    st.markdown(
        '<div class="metric-subtitle">Only rows with a proper Id</div>',
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

st.markdown(
    f"""
    <div class="small-text">
    {month_labels.get(selected_month, selected_month)} · 
    {total_month_credits:,.2f} total credits · 
    {len(month_slice):,} clean rows
    &nbsp;·&nbsp;
    {dropped_rows:,} rows removed from original {total_rows:,} (invalid Id / credits / fields)
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
    st.caption("`email` – ranked by total credits")
    user_df = user_breakdown(visible_df)

    if user_df.empty:
        st.info("No user data to display (no valid emails).")
    else:
        st.dataframe(
            user_df,
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
    Cleaning rules: rows are excluded if <code>Id</code> is missing/short,
    <code>cost_in_credits</code> is non-numeric or ≤ 0, 
    <code>call_type</code> or <code>language_model</code> are blank,
    or <code>created_at</code> is invalid.
    </div>
    """,
    unsafe_allow_html=True,
)
