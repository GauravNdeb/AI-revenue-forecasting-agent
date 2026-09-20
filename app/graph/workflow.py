from typing import TypedDict, Any

import pandas as pd
from langgraph.graph import StateGraph, START, END

from app.agents.revenue_agent import (
    agent_decide,
    execute_selected_tools,
    generate_high_risk_actions,
)


class RevenueState(TypedDict, total=False):
    df: Any
    decision: dict
    tool_results: dict
    next_best_actions: dict


def observe_pipeline(state: RevenueState) -> RevenueState:
    df = state["df"]

    if df is None or df.empty:
        raise ValueError("Pipeline dataset is empty.")

    return {
        **state,
        "df": df,
    }


def decision_node(state: RevenueState) -> RevenueState:
    df = state["df"]

    decision = agent_decide(df)

    return {
        **state,
        "decision": decision,
    }


def tool_execution_node(state: RevenueState) -> RevenueState:
    df = state["df"]
    decision = state["decision"]

    results = execute_selected_tools(
        df,
        decision,
    )

    return {
        **state,
        "tool_results": results,
    }


def next_best_action_node(state: RevenueState) -> RevenueState:

    df = state["df"]
    results = state["tool_results"]

    actions = generate_high_risk_actions(
        df,
        results
    )

    return {
        **state,
        "next_best_actions": actions
    }


def build_revenue_workflow():

    workflow = StateGraph(RevenueState)

    workflow.add_node(
        "observe_pipeline",
        observe_pipeline,
    )

    workflow.add_node(
        "decision_agent",
        decision_node,
    )

    workflow.add_node(
        "execute_tools",
        tool_execution_node,
    )

    workflow.add_node(
        "next_best_action_agent",
        next_best_action_node,
    )

    workflow.add_edge(
        START,
        "observe_pipeline",
    )

    workflow.add_edge(
        "observe_pipeline",
        "decision_agent",
    )

    workflow.add_edge(
        "decision_agent",
        "execute_tools",
    )

    workflow.add_edge(
        "execute_tools",
        "next_best_action_agent",
    )

    workflow.add_edge(
        "next_best_action_agent",
        END,
    )

    return workflow.compile()

revenue_workflow = build_revenue_workflow()

def run_revenue_agent(df: pd.DataFrame) -> dict:

    result = revenue_workflow.invoke({
        "df": df,
    })

    return result
