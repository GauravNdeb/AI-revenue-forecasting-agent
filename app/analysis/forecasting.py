import pandas as pd


def calculate_forecast(
    df: pd.DataFrame,
    stage_win_rates: dict
) -> dict:

    df = df.copy()

    # Assign probability based on stage
    df["win_probability"] = (
        df["stage"].map(stage_win_rates).fillna(0)
    )

    # Weighted revenue
    df["weighted_revenue"] = (
        df["amount_usd"] * df["win_probability"]
    )

    total_pipeline = df["amount_usd"].sum()

    weighted_forecast = df["weighted_revenue"].sum()

    # Simple confidence range for hackathon MVP
    lower_bound = weighted_forecast * 0.85
    upper_bound = weighted_forecast * 1.15

    confidence = "Medium"

    if len(df) >= 100:
        confidence = "High"
    elif len(df) >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "total_pipeline": round(float(total_pipeline), 2),
        "weighted_forecast": round(float(weighted_forecast), 2),
        "forecast_lower_bound": round(float(lower_bound), 2),
        "forecast_upper_bound": round(float(upper_bound), 2),
        "confidence": confidence,
    }
