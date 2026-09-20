from app.llm.model import ask_ai


def analyze_deal_with_ai(deal: dict) -> dict:

    prompt = f"""
You are an expert B2B sales manager.

Analyze this CRM deal using only the information provided.

Deal ID: {deal.get("deal_id")}
Customer: {deal.get("customer_name")}
Sales Rep: {deal.get("sales_rep")}
Amount: ${deal.get("amount_usd")}
Stage: {deal.get("stage")}

Days Since Activity:
{deal.get("days_since_activity")}

Days In Stage:
{deal.get("days_in_stage")}

Stakeholders:
{deal.get("stakeholder_count")}

Economic Buyer Engaged:
{deal.get("economic_buyer_engaged")}

Proposal Sent:
{deal.get("proposal_sent")}

Close Date Changes:
{deal.get("close_date_changes")}

Customer Response Days:
{deal.get("customer_response_days")}

Engagement Trend:
{deal.get("engagement_trend")}

Risk Score:
{deal.get("risk_score")}

Risk Level:
{deal.get("risk_level")}

Risk Reasons:
{deal.get("risk_reasons")}

Sales Notes:
{deal.get("notes")}

Give:

1. Why this deal is at risk.
2. Key risk signals.
3. ONE specific next-best-action.
4. Why that action matters.

Do not invent information.

Keep it concise.

Format:

RISK EXPLANATION:
...

KEY RISKS:
- ...
- ...

NEXT BEST ACTION:
...

WHY THIS ACTION:
...
"""

    response = ask_ai(
        prompt
    )

    return {
        "deal_id": deal.get(
            "deal_id"
        ),
        "customer_name": deal.get(
            "customer_name"
        ),
        "ai_analysis": response
    }