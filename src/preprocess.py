import pandas as pd

DRUG_COLUMNS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]


def load_raw_sales(csv_path: str = "data/salesdaily.csv") -> pd.DataFrame:
    """Load salesdaily.csv, convert datum to datetime, and sort chronologically."""
    df = pd.read_csv(csv_path)
    df["datum"] = pd.to_datetime(df["datum"])
    df = df.sort_values("datum").drop_duplicates(subset=["datum"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame, drug_col: str) -> pd.DataFrame:
    """Build calendar, lag, and rolling features without future data leakage."""
    if drug_col not in DRUG_COLUMNS:
        raise ValueError(f"Choose one of {DRUG_COLUMNS}, got '{drug_col}'")

    series = df[drug_col].astype(float)
    past_series = series.shift(1)

    out = pd.DataFrame({
        "date": df["datum"],
        "y": series,
        "month": df["datum"].dt.month,
        "dayofweek": df["datum"].dt.dayofweek,
        "dayofyear": df["datum"].dt.dayofyear,
        "is_weekend": (df["datum"].dt.dayofweek >= 5).astype(int),
        "lag_1": series.shift(1),
        "lag_7": series.shift(7),
        "lag_14": series.shift(14),
        "lag_28": series.shift(28),
        "roll_mean_7": past_series.rolling(7).mean(),
        "roll_mean_30": past_series.rolling(30).mean(),
        "roll_std_7": past_series.rolling(7).std(),
    })

    return out.dropna().reset_index(drop=True)