import pandas as pd


def load_pipeline_data(file_path: str) -> pd.DataFrame:
    """
    Load CRM pipeline data from CSV and validate required columns.
    """

    df = pd.read_csv(file_path)

    required_columns = [
        "deal_id",
        "customer_name",
        "sales_rep",
        "amount_usd",
        "stage",
        "created_date",
        "expected_close_date",
        "last_activity_date",
        "days_in_stage",
        "num_emails",
        "num_calls",
        "num_meetings",
        "stakeholder_count",
        "economic_buyer_engaged",
        "proposal_sent",
        "close_date_changes",
        "customer_response_days",
        "engagement_trend",
        "notes",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert dates
    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    df["expected_close_date"] = pd.to_datetime(
        df["expected_close_date"], errors="coerce"
    )

    df["last_activity_date"] = pd.to_datetime(
        df["last_activity_date"], errors="coerce"
    )

    # Calculate days since last activity
    # Fixed analysis date makes the demo reproducible.
    analysis_date = pd.Timestamp("2026-09-11")

    df["days_since_activity"] = (
        analysis_date - df["last_activity_date"]
    ).dt.days

    # Prevent negative values
    df["days_since_activity"] = (
        df["days_since_activity"].clip(lower=0)
    )

    return df