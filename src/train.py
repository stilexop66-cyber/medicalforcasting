from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS = [
    "month",
    "dayofweek",
    "dayofyear",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_mean_30",
    "roll_std_7",
]


def _make_model(model_name: str = "XGBoost"):
    if model_name == "XGBoost" and HAS_XGB:
        return XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    non_zero = y_true > 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
        if np.any(non_zero)
        else np.nan
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2) if not np.isnan(mape) else None,
    }


def evaluate_model(feat_df: pd.DataFrame, test_days: int = 60, model_name: str = "XGBoost"):
    if len(feat_df) <= test_days + 30:
        raise ValueError("Not enough rows in feature table for the requested test_days split.")

    train_df = feat_df.iloc[:-test_days].copy()
    test_df = feat_df.iloc[-test_days:].copy()

    model = _make_model(model_name)
    model.fit(train_df[FEATURE_COLS], train_df["y"])
    test_preds = np.clip(model.predict(test_df[FEATURE_COLS]), 0.0, None)

    metrics = compute_metrics(test_df["y"], test_preds)
    validation_df = pd.DataFrame({
        "date": test_df["date"].values,
        "actual": test_df["y"].values,
        "predicted": np.round(test_preds, 2),
    })

    # Fit on full history for future forecasting
    full_model = _make_model(model_name)
    full_model.fit(feat_df[FEATURE_COLS], feat_df["y"])
    return full_model, validation_df, metrics


def forecast_future(
    feat_df: pd.DataFrame,
    model,
    horizon_days: int = 14,
) -> pd.DataFrame:
    """Recursively forecast horizon_days ahead from the last date in feat_df."""
    history = feat_df[["date", "y"]].copy().reset_index(drop=True)
    future_rows = []

    for _ in range(horizon_days):
        next_date = history["date"].iloc[-1] + timedelta(days=1)
        vals = history["y"].to_numpy(dtype=float)

        lag_1 = float(vals[-1])
        lag_7 = float(vals[-7]) if len(vals) >= 7 else lag_1
        lag_14 = float(vals[-14]) if len(vals) >= 14 else lag_7
        lag_28 = float(vals[-28]) if len(vals) >= 28 else lag_14

        roll_mean_7 = float(np.mean(vals[-7:]))
        roll_mean_30 = float(np.mean(vals[-30:]))
        roll_std_7 = float(np.std(vals[-7:], ddof=1)) if len(vals) >= 2 else 0.0

        row_features = pd.DataFrame([{
            "month": next_date.month,
            "dayofweek": next_date.dayofweek,
            "dayofyear": next_date.dayofyear,
            "is_weekend": int(next_date.dayofweek >= 5),
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_28": lag_28,
            "roll_mean_7": roll_mean_7,
            "roll_mean_30": roll_mean_30,
            "roll_std_7": roll_std_7,
        }])

        pred = float(np.clip(model.predict(row_features)[0], 0.0, None))
        future_rows.append({"date": next_date, "forecast_demand": round(pred, 2)})
        history = pd.concat(
            [history, pd.DataFrame([{"date": next_date, "y": pred}])],
            ignore_index=True,
        )

    return pd.DataFrame(future_rows)