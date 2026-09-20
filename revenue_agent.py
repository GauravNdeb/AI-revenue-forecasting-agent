import json
import pandas as pd

from app.llm.model import get_llm


STAGE_WIN_RATES = {
    "Discovery": 0.15,
    "Demo": 0.30,
    "Proposal": 0.55,
    "Negotiation": 0.75,
}


def forecast_pipeline(df: pd.DataFrame):
    working = df.copy()

    working["win_probability"] = (
        working["stage"]
        .map(STAGE_WIN_RATES)
        .fillna(0)
    )

    working["weighted_revenue"] = (
        working["amount_usd"]
        * working["win_probability"]
    )

    total_pipeline = working["amount_usd"].sum()
    weighted_forecast = working["weighted_revenue"].sum()

    return {
        "total_pipeline": float(total_pipeline),
        "weighted_forecast": float(weighted_forecast),
        "lower_bound": float(weighted_forecast * 0.85),
        "upper_bound": float(weighted_forecast * 1.15),
    }


def find_risky_deals(df: pd.DataFrame):

    from app.analysis.risk_deals import analyze_risks

    risk_df = analyze_risks(df)

    risky = risk_df[
        risk_df["risk_level"].isin(["High", "Medium"])
    ].copy()

    risky = risky.sort_values(
        by="risk_score",
        ascending=False
    )

    return risky.head(15)


def find_best_deals(df: pd.DataFrame):

    from app.analysis.risk_deals import analyze_risks

    analysis_df = analyze_risks(df)

    best = analysis_df[
        analysis_df["risk_level"] == "Low"
    ].copy()

    best = best.sort_values(
        by=[
            "opportunity_score",
            "amount_usd"
        ],
        ascending=False
    )

    return best.head(15)


def inspect_deal(df: pd.DataFrame, deal_id: str):

    from app.analysis.risk_deals import analyze_risks

    analysis_df = analyze_risks(df)

    matching = analysis_df[
        analysis_df["deal_id"].astype(str).str.lower()
        == str(deal_id).lower()
    ]

    if matching.empty:
        return None

    return matching.iloc[0].to_dict()


def build_pipeline_summary(df: pd.DataFrame):

    forecast = forecast_pipeline(df)

    stage_summary = (
        df.groupby("stage")
        .agg(
            deals=("deal_id", "count"),
            pipeline=("amount_usd", "sum")
        )
        .reset_index()
    )

    rep_summary = (
        df.groupby("sales_rep")
        .agg(
            deals=("deal_id", "count"),
            pipeline=("amount_usd", "sum")
        )
        .reset_index()
    )

    return {
        "deal_count": len(df),
        "forecast": forecast,
        "stage_summary": stage_summary.to_dict(
            orient="records"
        ),
        "rep_summary": rep_summary.to_dict(
            orient="records"
        ),
    }


# =========================================================
# AGENT DECISION
# =========================================================

def agent_decide(df: pd.DataFrame):

    llm = get_llm()

    summary = build_pipeline_summary(df)

    prompt = f"""
You are an autonomous B2B Revenue Operations Agent.

Your job is to decide which analysis tools are needed
to understand the current CRM pipeline.

AVAILABLE TOOLS:

1. forecast_pipeline
Calculates weighted revenue forecast.

2. find_risky_deals
Finds stalled, inactive, declining-engagement,
missing-buyer and slipping deals.

3. find_best_deals
Finds healthy high-potential opportunities.

4. inspect_deal
Investigates a specific priority deal.

CURRENT PIPELINE:

{json.dumps(summary, indent=2, default=str)}

Decide which tools should be executed.

Return ONLY valid JSON:

{{
    "tools_to_run": [
        "forecast_pipeline",
        "find_risky_deals",
        "find_best_deals"
    ],
    "reason": "short explanation",
    "priority": "risk|opportunity|balanced"
}}

Rules:

- forecast_pipeline should normally be selected.
- find_risky_deals should normally be selected.
- find_best_deals should normally be selected.
- Only use the tool names listed above.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content.strip()

        # Sometimes LLM returns ```json ... ```
        if content.startswith("```"):
            content = content.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        decision = json.loads(content)

    except Exception:

        # Safe fallback
        decision = {
            "tools_to_run": [
                "forecast_pipeline",
                "find_risky_deals",
                "find_best_deals",
            ],
            "reason": "Default pipeline intelligence analysis",
            "priority": "balanced",
        }

    allowed_tools = {
        "forecast_pipeline",
        "find_risky_deals",
        "find_best_deals",
    }

    decision["tools_to_run"] = [
        tool
        for tool in decision.get(
            "tools_to_run",
            []
        )
        if tool in allowed_tools
    ]

    if not decision["tools_to_run"]:

        decision["tools_to_run"] = [
            "forecast_pipeline",
            "find_risky_deals",
            "find_best_deals",
        ]

    return decision


# =========================================================
# TOOL EXECUTION
# =========================================================

def execute_selected_tools(
    df: pd.DataFrame,
    decision: dict
):

    results = {}

    for tool in decision.get(
        "tools_to_run",
        []
    ):

        if tool == "forecast_pipeline":

            results["forecast"] = forecast_pipeline(
                df
            )

        elif tool == "find_risky_deals":

            results["risky_deals"] = (
                find_risky_deals(df)
                .to_dict(orient="records")
            )

        elif tool == "find_best_deals":

            results["best_deals"] = (
                find_best_deals(df)
                .to_dict(orient="records")
            )

    return results


# =========================================================
# NEXT BEST ACTION
# =========================================================

def generate_next_best_actions(df, tool_results):
    import json

    llm = get_llm()

    risky_deals = tool_results.get(
        "risky_deals",
        []
    )

    # ONLY HIGH-RISK DEALS
    high_risk_deals = [
        deal
        for deal in risky_deals
        if str(
            deal.get("risk_level", "")
        ).lower() == "high"
    ]

    high_risk_deals = high_risk_deals[:10]

    if not high_risk_deals:
        return {
            "actions": []
        }

    prompt = f"""
You are a senior B2B Revenue Recovery Agent.

Generate NEXT BEST ACTIONS ONLY for HIGH-RISK deals.

HIGH-RISK CRM DEALS:

{json.dumps(
    high_risk_deals,
    indent=2,
    default=str
)}

For every deal:

1. Identify the main risk.
2. Give ONE concrete action.
3. Explain why the action should happen now.

Do NOT give generic advice.

Bad:
"Follow up with customer."

Good:
"Schedule a 20-minute meeting with the economic buyer
this week to confirm budget approval and decision timeline."

Return ONLY valid JSON:

{{
    "actions": [
        {{
            "deal_id": "DEAL-001",
            "customer_name": "Customer",
            "category": "risk",
            "priority": "high",
            "risk_score": 85,
            "main_risk": "Economic buyer is not engaged",
            "action": "Schedule a meeting with the economic buyer this week.",
            "reason": "The deal is already delayed and lacks executive sponsorship."
        }}
    ]
}}
"""

    response = llm.invoke(prompt)

    try:

        return json.loads(
            response.content.strip()
        )

    except Exception:

        return {
            "actions": []
        }

def generate_high_risk_actions(df: pd.DataFrame):
    """
    Generate next-best-actions only for HIGH risk deals.
    """

    from app.analysis.risk_deals import analyze_risks

    risk_df = analyze_risks(df)

    high_risk = risk_df[
        risk_df["risk_level"] == "High"
    ].copy()

    if high_risk.empty:
        return []

    high_risk = high_risk.sort_values(
        by="risk_score",
        ascending=False
    ).head(15)

    llm = get_llm()

    deals = high_risk.to_dict(
        orient="records"
    )

    prompt = f"""
You are a senior B2B revenue operations manager.

These are HIGH-RISK CRM deals:

{json.dumps(deals, indent=2, default=str)}

For EACH deal, generate exactly ONE concrete
NEXT BEST ACTION.

The action must directly address the actual
risk signals.

Examples:

- Missing economic buyer
  -> Schedule an economic buyer discovery meeting.

- No recent activity
  -> Re-engage the customer with a specific
     business-value follow-up.

- Deal stuck in stage
  -> Schedule a stage-exit meeting and identify
     the blocking requirement.

- Multiple close-date changes
  -> Run a close-date validation call and
     confirm the customer's actual buying timeline.

Do NOT give generic advice.

Return ONLY valid JSON:

{{
    "actions": [
        {{
            "deal_id": "DEAL-001",
            "customer_name": "Customer",
            "risk_score": 85,
            "action": "Specific action",
            "reason": "Why this action addresses the risk",
            "urgency": "Immediate"
        }}
    ]
}}

Rules:

- Include every high-risk deal provided.
- One action per deal.
- Use actual CRM signals.
- Keep action concise.
"""

    response = llm.invoke(prompt)

    try:
        result = json.loads(
            response.content.strip()
        )

        return result.get("actions", [])

    except Exception:
        return []
def generate_high_risk_actions(df, tool_results):
    import json

    llm = get_llm()

    risky_deals = tool_results.get("risky_deals", [])

    high_risk = [
        deal
        for deal in risky_deals
        if str(deal.get("risk_level", "")).lower() == "high"
    ]

    # If risk level is not available, use score
    if not high_risk:
        high_risk = [
            deal
            for deal in risky_deals
            if float(deal.get("risk_score", 0)) >= 70
        ]

    high_risk = high_risk[:10]

    if not high_risk:
        return {
            "actions": []
        }

    prompt = f"""
You are an expert B2B revenue recovery agent.

Your ONLY job is to create NEXT BEST ACTIONS
for HIGH-RISK CRM deals.

Here are the high-risk deals:

{json.dumps(high_risk, indent=2, default=str)}

For every high-risk deal:

1. Identify the main risk.
2. Recommend ONE specific action the sales rep
   should take next.
3. Explain why that action matters now.
4. Make the action operational and specific.

Examples:

Bad:
"Follow up with customer."

Good:
"Schedule a 20-minute meeting with the economic buyer
this week to confirm budget approval and decision timeline."

Return ONLY valid JSON:

{{
    "actions": [
        {{
            "deal_id": "DEAL-001",
            "customer_name": "Customer",
            "risk_score": 85,
            "risk_level": "High",
            "main_risk": "Economic buyer is not engaged",
            "next_best_action": "Schedule a meeting with the economic buyer...",
            "reason": "Without economic buyer engagement..."
        }}
    ]
}}

Do not generate actions for low-risk or medium-risk deals.
"""

    response = llm.invoke(prompt)

    try:
        return json.loads(response.content.strip())
    except Exception:
        return {
            "actions": []
        }
def ask_revenue_agent(question, df, tool_results=None):

    import json

    llm = get_llm()

    if tool_results is None:
        tool_results = {}

    forecast = tool_results.get("forecast", {})
    risky = tool_results.get("risky_deals", [])
    best = tool_results.get("best_deals", [])

    context = {
        "forecast": forecast,
        "high_risk_deals": [
            d for d in risky
            if float(d.get("risk_score", 0)) >= 70
        ][:10],
        "best_deals": best[:10],
    }

    prompt = f"""
You are an AI Revenue Operations Agent.

Answer the sales leader's question using the CRM
pipeline data below.

CRM CONTEXT:

{json.dumps(context, indent=2, default=str)}

USER QUESTION:

{question}

Rules:

- Use actual CRM data.
- Mention deal IDs when relevant.
- Mention numbers when relevant.
- If the user asks where to focus, prioritize
  high-risk revenue-impacting deals.
- If the user asks about a specific deal, explain
  its risk signals.
- If asked for an action, give a concrete next action.
- Do not invent CRM facts.

Give a concise executive-level answer.
"""

    response = llm.invoke(prompt)

    return response.content