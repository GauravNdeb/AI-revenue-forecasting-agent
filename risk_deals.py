import pandas as pd


def calculate_risk_score(row):
    score = 0
    reasons = []

    # 1. Inactivity
    if row["days_since_activity"] > 14:
        score += 25
        reasons.append(
            f"No customer activity for {int(row['days_since_activity'])} days"
        )

    # 2. Stalled stage
    if row["days_in_stage"] > 30:
        score += 20
        reasons.append(
            f"Deal has been in {row['stage']} for {int(row['days_in_stage'])} days"
        )

    # 3. Economic buyer
    if not bool(row["economic_buyer_engaged"]):
        score += 20
        reasons.append("Economic buyer is not engaged")

    # 4. Engagement
    if str(row["engagement_trend"]).lower() == "declining":
        score += 15
        reasons.append("Customer engagement is declining")

    # 5. Close date movement
    if row["close_date_changes"] >= 2:
        score += 20
        reasons.append(
            f"Close date has been changed {int(row['close_date_changes'])} times"
        )

    # 6. Slow response
    if row["customer_response_days"] > 7:
        score += 10
        reasons.append(
            f"Customer response time is {int(row['customer_response_days'])} days"
        )

    # 7. Stakeholder coverage
    if row["stakeholder_count"] < 2:
        score += 10
        reasons.append("Too few stakeholders are engaged")

    score = min(score, 100)

    if score >= 70:
        risk_level = "High"
    elif score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return score, risk_level, reasons


def calculate_opportunity_score(row, stage_win_rates):
    """
    Score healthy/high-potential deals.
    This is NOT simply the inverse of risk.
    """

    score = 0

    stage_probability = stage_win_rates.get(row["stage"], 0)

    # Stage probability
    score += stage_probability * 50

    # Positive engagement
    trend = str(row["engagement_trend"]).lower()

    if trend == "increasing":
        score += 20
    elif trend == "stable":
        score += 10

    # Economic buyer
    if bool(row["economic_buyer_engaged"]):
        score += 15

    # Recent activity
    if row["days_since_activity"] <= 7:
        score += 10

    # Stakeholder coverage
    if row["stakeholder_count"] >= 3:
        score += 5

    return round(min(score, 100), 1)


def analyze_risks(df: pd.DataFrame) -> pd.DataFrame:

    stage_win_rates = {
        "Discovery": 0.15,
        "Demo": 0.30,
        "Proposal": 0.55,
        "Negotiation": 0.75,
    }

    results = []

    for _, row in df.iterrows():

        risk_score, risk_level, reasons = calculate_risk_score(row)

        opportunity_score = calculate_opportunity_score(
            row,
            stage_win_rates
        )

        results.append({
            "deal_id": row["deal_id"],
            "customer_name": row["customer_name"],
            "sales_rep": row["sales_rep"],
            "amount_usd": float(row["amount_usd"]),
            "stage": row["stage"],

            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_reasons": reasons,

            "opportunity_score": opportunity_score,

            "days_since_activity": int(row["days_since_activity"]),
            "days_in_stage": int(row["days_in_stage"]),

            "economic_buyer_engaged": bool(
                row["economic_buyer_engaged"]
            ),

            "stakeholder_count": int(
                row["stakeholder_count"]
            ),

            "engagement_trend": row["engagement_trend"],

            "close_date_changes": int(
                row["close_date_changes"]
            ),

            "customer_response_days": int(
                row["customer_response_days"]
            ),
        })

    return pd.DataFrame(results)