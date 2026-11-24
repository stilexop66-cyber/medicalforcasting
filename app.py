import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inventory import compute_inventory_plan
from src.preprocess import DRUG_COLUMNS, build_features, load_raw_sales
from src.train import evaluate_model, forecast_future

st.set_page_config(
    page_title="Medical Supply Demand Forecasting",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Medical Supply Demand Forecasting & Procurement Dashboard")
st.caption("Data-driven demand forecasting and reorder planning from daily pharmacy sales history")


@st.cache_data
def get_data(csv_path: str = "data/salesdaily.csv") -> pd.DataFrame:
    return load_raw_sales(csv_path)


@st.cache_resource
def run_pipeline(drug_col: str, test_days: int, horizon_days: int):
    raw_df = get_data()
    feat_df = build_features(raw_df, drug_col)
    model, validation_df, metrics = evaluate_model(feat_df, test_days=test_days, model_name="XGBoost")
    forecast_df = forecast_future(feat_df, model, horizon_days=horizon_days)
    return raw_df, feat_df, validation_df, forecast_df, metrics


# --- Sidebar Controls ---
st.sidebar.header("Forecast & Inventory Controls")
selected_drug = st.sidebar.selectbox("ATC Drug Category", DRUG_COLUMNS, index=3)  # Default: N02BE
horizon_days = st.sidebar.selectbox("Forecast Horizon (Days)", [7, 14, 30, 60], index=1)
test_days = st.sidebar.selectbox("Validation Window (Days)", [30, 60, 90], index=1)

st.sidebar.divider()
st.sidebar.subheader("Inventory Parameters")
current_stock = st.sidebar.number_input("Current Stock on Hand (Units)", min_value=0.0, value=300.0, step=10.0)
lead_time_days = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1, max_value=21, value=5)
service_level = st.sidebar.selectbox("Target Service Level (%)", [90, 95, 98, 99], index=1)

try:
    raw_df, feat_df, validation_df, forecast_df, metrics = run_pipeline(
        selected_drug, test_days, horizon_days
    )
except FileNotFoundError:
    st.error("Could not find `data/salesdaily.csv`. Make sure the file is placed inside `miniproj/data/`.")
    st.stop()

plan = compute_inventory_plan(
    forecast_df=forecast_df,
    historical_series=raw_df[selected_drug],
    current_stock=current_stock,
    lead_time_days=lead_time_days,
    service_level=service_level,
)

# --- Alert Banner ---
if plan["status"] == "Reorder Now":
    st.warning(f"⚠️ **Reorder Alert ({selected_drug}):** {plan['alert_message']}")
else:
    st.success(f"✅ **Stock Healthy ({selected_drug}):** {plan['alert_message']}")

# --- Top KPI Row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{horizon_days}-Day Projected Demand", f"{plan['projected_horizon_demand']:.1f}")
c2.metric("Reorder Point (ROP)", f"{plan['reorder_point']:.1f}")
c3.metric("Recommended Order Qty", f"{plan['recommended_order_qty']} units")
c4.metric("Validation MAE / RMSE", f"{metrics['MAE']} / {metrics['RMSE']}")
mape_str = f"{metrics['MAPE (%)']}%" if metrics["MAPE (%)"] is not None else "N/A"
c5.metric("Validation MAPE", mape_str)

# --- Forecast & Validation Chart ---
recent_history = feat_df.tail(90)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent_history["date"],
    y=recent_history["y"],
    mode="lines",
    name="Historical Daily Demand (Last 90d)",
    line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=validation_df["date"],
    y=validation_df["predicted"],
    mode="lines",
    name="Test Validation Predictions",
    line=dict(color="#ff7f0e", width=2, dash="dot"),
))
fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["forecast_demand"],
    mode="lines+markers",
    name=f"Forecast ({horizon_days}d)",
    line=dict(color="#2ca02c", width=2.5),
))
fig.add_hline(
    y=plan["reorder_point"] / max(lead_time_days, 1),
    line_dash="dash",
    line_color="#d62728",
    annotation_text="Approx. Daily ROP Threshold",
    annotation_position="top left",
)

fig.update_layout(
    title=f"{selected_drug} — Historical Demand, Validation & {horizon_days}-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Daily Demand (Units)",
    hovermode="x unified",
    height=440,
)
st.plotly_chart(fig, use_container_width=True)

# --- Forecast Table & CSV Export ---
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader(f"Next {horizon_days} Days Forecast Schedule ({selected_drug})")
    display_fc = forecast_df.copy()
    display_fc["date"] = pd.to_datetime(display_fc["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    csv_bytes = display_fc.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_bytes,
        file_name=f"{selected_drug}_{horizon_days}d_forecast.csv",
        mime="text/csv",
    )

with right_col:
    st.subheader("Inventory Summary")
    st.write(f"- **Average Daily Forecast:** {plan['avg_daily_forecast']} units/day")
    st.write(f"- **Safety Stock ({service_level}% SL):** {plan['safety_stock']} units")
    st.write(f"- **Reorder Point:** {plan['reorder_point']} units")
    st.write(f"- **Recommended Order:** {plan['recommended_order_qty']} units")
    cover_text = f"{plan['days_of_cover']} days" if plan["days_of_cover"] is not None else "N/A"
    st.write(f"- **Estimated Days of Cover:** {cover_text}")