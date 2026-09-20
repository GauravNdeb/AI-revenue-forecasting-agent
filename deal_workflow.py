from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from app.utils.data_loader import load_pipeline_data
from app.analysis.risk_deals import analyze_risks
from app.agents.sales_agent import analyze_deal_with_ai


DATA_PATH = "data/crm_pipeline.csv"


class DealState(TypedDict, total=False):
    deal_id: str
    deal: dict
    risk_analysis: dict
    ai_analysis: dict


def load_deal(state: DealState) -> DealState:
    df = load_pipeline_data(DATA_PATH)

    deal_id = str(state["deal_id"]).strip()

    matching = df[
        df["deal_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        == deal_id.lower()
    ]

    if matching.empty:
        raise ValueError(f"Deal '{deal_id}' not found.")

    deal = matching.iloc[0].to_dict()

    return {
        **state,
        "deal": deal,
    }


def calculate_deal_risk(state: DealState) -> DealState:
    df = load_pipeline_data(DATA_PATH)

    risk_df = analyze_risks(df)

    deal_id = str(state["deal_id"]).strip()

    matching = risk_df[
        risk_df["deal_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        == deal_id.lower()
    ]

    if matching.empty:
        raise ValueError(
            f"Risk information not found for deal {deal_id}"
        )

    risk = matching.iloc[0].to_dict()

    return {
        **state,
        "risk_analysis": risk,
    }


def generate_ai_analysis(state: DealState) -> DealState:

    deal = state["deal"].copy()
    risk = state["risk_analysis"]

    deal["risk_score"] = risk["risk_score"]
    deal["risk_level"] = risk["risk_level"]
    deal["risk_reasons"] = risk["risk_reasons"]

    ai_result = analyze_deal_with_ai(deal)

    return {
        **state,
        "ai_analysis": ai_result,
    }


def build_deal_workflow():

    workflow = StateGraph(DealState)

    workflow.add_node("load_deal", load_deal)
    workflow.add_node("calculate_risk", calculate_deal_risk)
    workflow.add_node("ai_analysis", generate_ai_analysis)

    workflow.add_edge(
        START,
        "load_deal"
    )

    workflow.add_edge(
        "load_deal",
        "calculate_risk"
    )

    workflow.add_edge(
        "calculate_risk",
        "ai_analysis"
    )

    workflow.add_edge(
        "ai_analysis",
        END
    )

    return workflow.compile()


deal_workflow = build_deal_workflow()


def run_deal_analysis(deal_id: str) -> dict:

    result = deal_workflow.invoke({
        "deal_id": str(deal_id)
    })

    return result