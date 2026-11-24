import math
import numpy as np
import pandas as pd

SERVICE_LEVEL_Z = {
    90: 1.28,
    95: 1.65,
    98: 2.05,
    99: 2.33,
}


def compute_inventory_plan(
    forecast_df: pd.DataFrame,
    historical_series: pd.Series,
    current_stock: float,
    lead_time_days: int = 5,
    service_level: int = 95,
) -> dict:
    """Compute safety stock, reorder point, recommended order qty, and alert status."""
    if forecast_df.empty:
        raise ValueError("forecast_df cannot be empty.")

    avg_daily_forecast = float(forecast_df["forecast_demand"].mean())
    projected_horizon_demand = float(forecast_df["forecast_demand"].sum())

    daily_std = float(pd.Series(historical_series).astype(float).std(ddof=1))
    if np.isnan(daily_std) or daily_std < 0:
        daily_std = 0.0

    z_score = SERVICE_LEVEL_Z.get(service_level, 1.65)
    safety_stock = z_score * daily_std * math.sqrt(max(lead_time_days, 0))
    reorder_point = (avg_daily_forecast * max(lead_time_days, 0)) + safety_stock

    recommended_order_qty = max(
        0.0,
        projected_horizon_demand + safety_stock - float(current_stock),
    )

    days_of_cover = (
        float(current_stock) / avg_daily_forecast
        if avg_daily_forecast > 1e-6
        else float("inf")
    )

    if float(current_stock) <= reorder_point:
        status = "Reorder Now"
        alert_message = (
            f"Stock ({current_stock:.1f}) is at or below Reorder Point ({reorder_point:.1f}). "
            f"Recommended order: {math.ceil(recommended_order_qty)} units."
        )
    else:
        status = "Healthy Stock"
        alert_message = (
            f"Stock ({current_stock:.1f}) is above Reorder Point ({reorder_point:.1f}). "
            f"Estimated cover: {days_of_cover:.1f} days."
        )

    return {
        "avg_daily_forecast": round(avg_daily_forecast, 2),
        "projected_horizon_demand": round(projected_horizon_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": int(math.ceil(recommended_order_qty)),
        "days_of_cover": round(days_of_cover, 1) if math.isfinite(days_of_cover) else None,
        "status": status,
        "alert_message": alert_message,
    }