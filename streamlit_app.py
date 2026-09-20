import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from app.graph.workflow import (
    run_revenue_agent,
    revenue_workflow,
)

from app.llm.model import get_llm


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Revenue Forecasting Agent",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "pipeline_df" not in st.session_state:
    st.session_state.pipeline_df = None

if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

if "draft_actions" not in st.session_state:
    st.session_state.draft_actions = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def inr(value):

    try:
        value = float(value)
    except Exception:
        value = 0

    crore = value / 10_000_000

    if crore >= 1:
        return f"₹{crore:.2f} Cr"

    lakh = value / 100_000

    return f"₹{lakh:.2f} L"


def prepare_pipeline(df):

    working_df = df.copy()

    date_columns = [
        "created_date",
        "expected_close_date",
        "last_activity_date",
    ]

    for column in date_columns:

        if column in working_df.columns:

            working_df[column] = pd.to_datetime(
                working_df[column],
                errors="coerce",
            )

    if "last_activity_date" in working_df.columns:

        analysis_date = pd.Timestamp.today().normalize()

        working_df["days_since_activity"] = (
            analysis_date
            - working_df["last_activity_date"]
        ).dt.days.clip(lower=0)

    return working_df


def ask_ai_agent(question, df, agent_result):

    """
    Ask the Revenue AI Agent a question using
    the already analyzed pipeline context.
    """

    llm = get_llm()

    tool_results = agent_result.get(
        "tool_results",
        {},
    )

    forecast = tool_results.get(
        "forecast",
        {},
    )

    risky_deals = tool_results.get(
        "risky_deals",
        [],
    )

    best_deals = tool_results.get(
        "best_deals",
        [],
    )

    high_risk_deals = [
        deal
        for deal in risky_deals
        if str(
            deal.get("risk_level", "")
        ).lower() == "high"
    ]

    pipeline_context = {

        "total_deals": len(df),

        "forecast": forecast,

        "high_risk_deals": high_risk_deals[:15],

        "best_deals": best_deals[:15],

    }

    prompt = f"""
You are an AI Revenue Operations Agent.

You are helping a sales leader understand their CRM pipeline.

CURRENT PIPELINE DATA:

{pipeline_context}

USER QUESTION:

{question}

Instructions:

- Answer using the CRM data provided.
- Do not invent CRM facts.
- Mention specific deal IDs when useful.
- Mention revenue amounts when useful.
- If the question asks where to focus, prioritize
  high-risk deals with meaningful revenue impact.
- If the question asks about a specific deal,
  explain its risk signals.
- If the question asks for an action,
  give one concrete next action.
- Keep the answer concise and executive-friendly.

Return a useful natural-language answer.
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "📊 AI Revenue Forecasting Agent"
)

st.caption(
    "Agentic CRM pipeline intelligence for revenue leadership"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header(
    "📁 Pipeline Data"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CRM CSV",
    type=["csv"],
)


if uploaded_file:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        st.session_state.pipeline_df = df

        st.sidebar.success(
            f"✅ {len(df)} deals loaded"
        )

    except Exception as e:

        st.sidebar.error(
            f"Could not load CSV: {e}"
        )


# --------------------------------------------------
# ZERO STATE
# --------------------------------------------------

if st.session_state.pipeline_df is None:

    st.info(
        "👋 Upload your CRM pipeline CSV from the sidebar to start analysis."
    )

    st.markdown(
        """
        ## What the Revenue Agent does

        ### 📈 Revenue Forecast
        - Weighted pipeline forecast
        - Forecast range
        - Revenue confidence

        ### 🚨 Risk Intelligence
        - Stalled deals
        - Inactive customers
        - Missing economic buyers
        - Declining engagement
        - Close-date slippage

        ### ⭐ Opportunity Intelligence
        - Healthy high-potential deals
        - Strong engagement
        - Revenue acceleration opportunities

        ### 🤖 Next Best Action
        - High-risk deal intervention
        - AI-generated action
        - Reason for action
        - Revenue leadership priority

        ### 💬 Ask the Revenue Agent
        Ask questions like:

        > Where should we focus this week?

        > Which high-risk deal needs immediate attention?

        > Why is DEAL-023 risky?

        > What is causing forecast uncertainty?
        """
    )

    st.stop()


# --------------------------------------------------
# DATA
# --------------------------------------------------

df = st.session_state.pipeline_df

working_df = prepare_pipeline(
    df
)


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🚀 Analyze Pipeline",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "🤖 Revenue Agent is analyzing your pipeline..."
    ):

        try:

            result = run_revenue_agent(
                working_df
            )

            st.session_state.agent_result = result

            st.session_state.chat_messages = []

            st.success(
                "✅ Agent analysis completed."
            )

        except Exception as e:

            st.error(
                f"❌ Agent analysis failed: {e}"
            )

            st.exception(e)


# --------------------------------------------------
# WAIT FOR ANALYSIS
# --------------------------------------------------

if st.session_state.agent_result is None:

    st.warning(
        "Click **Analyze Pipeline** to run the Revenue Agent."
    )

    st.stop()


# --------------------------------------------------
# AGENT RESULT
# --------------------------------------------------

result = st.session_state.agent_result

tool_results = result.get(
    "tool_results",
    {},
)

decision = result.get(
    "decision",
    {},
)

actions_data = result.get(
    "next_best_actions",
    {},
)


# --------------------------------------------------
# TOOL RESULTS
# --------------------------------------------------

forecast = tool_results.get(
    "forecast",
    {},
)

risky_deals = tool_results.get(
    "risky_deals",
    [],
)

best_deals = tool_results.get(
    "best_deals",
    [],
)


# --------------------------------------------------
# HIGH RISK
# --------------------------------------------------

high_risk_deals = [

    deal

    for deal in risky_deals

    if str(
        deal.get("risk_level", "")
    ).lower() == "high"

]


# --------------------------------------------------
# AT RISK REVENUE
# --------------------------------------------------

at_risk_revenue = sum(

    float(
        deal.get(
            "amount_usd",
            0,
        )
    )

    for deal in high_risk_deals

)


# --------------------------------------------------
# FORECAST
# --------------------------------------------------

total_pipeline = forecast.get(
    "total_pipeline",
    0,
)

weighted_forecast = forecast.get(
    "weighted_forecast",
    0,
)

lower_bound = forecast.get(
    "lower_bound",
    forecast.get(
        "forecast_lower_bound",
        0,
    ),
)

upper_bound = forecast.get(
    "upper_bound",
    forecast.get(
        "forecast_upper_bound",
        0,
    ),
)


# ==================================================
# LEADERSHIP OVERVIEW
# ==================================================

st.subheader(
    "📊 Leadership Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Pipeline",
        inr(total_pipeline),
    )


with col2:

    st.metric(
        "Weighted Forecast",
        inr(weighted_forecast),
    )


with col3:

    st.metric(
        "High Risk Deals",
        len(high_risk_deals),
    )


with col4:

    st.metric(
        "At-Risk Revenue",
        inr(at_risk_revenue),
    )


# --------------------------------------------------
# FORECAST RANGE
# --------------------------------------------------

st.caption(
    f"Forecast range: {inr(lower_bound)} – {inr(upper_bound)}"
)


# ==================================================
# AGENT DECISION
# ==================================================

st.subheader(
    "🤖 Revenue Agent Decision"
)

decision_reason = decision.get(
    "reason",
    "Pipeline analyzed.",
)

st.write(
    decision_reason
)

selected_tools = decision.get(
    "tools_to_run",
    [],
)

if selected_tools:

    st.caption(
        "Tools selected by agent: "
        + ", ".join(selected_tools)
    )


# ==================================================
# WORKFLOW GRAPH
# ==================================================
from app.analysis.risk_deals import analyze_risks

health_df = analyze_risks(working_df.copy())

# INR conversion
USD_TO_INR = 83

health_df["deal_value_inr"] = (
    health_df["amount_usd"] * USD_TO_INR
)

# -----------------------------
# Business metrics
# -----------------------------

total_pipeline = health_df["deal_value_inr"].sum()

stage_rates = {
    "Discovery": 0.15,
    "Demo": 0.30,
    "Proposal": 0.55,
    "Negotiation": 0.75,
}

health_df["win_probability"] = (
    health_df["stage"]
    .map(stage_rates)
    .fillna(0)
)

weighted_forecast = (
    health_df["amount_usd"]
    * health_df["win_probability"]
).sum() * USD_TO_INR

risk_revenue = health_df.loc[
    health_df["risk_level"] == "High",
    "deal_value_inr"
].sum()

high_engagement = health_df[
    (
        health_df["engagement_trend"]
        .astype(str)
        .str.lower()
        == "increasing"
    )
    & (health_df["risk_score"] < 40)
]

# -----------------------------
# KPI row
# -----------------------------


st.markdown("### 🎯 Deal Health Matrix")

# -----------------------------
# Risk vs Opportunity graph
# -----------------------------

plot_df = health_df.copy()

plot_df["deal_value_cr"] = (
    plot_df["deal_value_inr"] / 1e7
)

plot_df["health"] = plot_df.apply(
    lambda row:
        "🔴 High Risk"
        if row["risk_score"] >= 70
        else (
            "🟠 Medium Risk"
            if row["risk_score"] >= 40
            else "🟢 Healthy"
        ),
    axis=1
)

fig = px.scatter(
    plot_df,
    x="opportunity_score",
    y="risk_score",
    size="deal_value_cr",
    color="health",
    hover_name="customer_name",
    hover_data=[
        "deal_id",
        "sales_rep",
        "stage",
        "deal_value_cr",
        "days_since_activity",
        "stakeholder_count",
        "engagement_trend",
    ],
    labels={
        "opportunity_score": "Opportunity Score",
        "risk_score": "Risk Score",
        "deal_value_cr": "Deal Value (₹ Cr)",
        "health": "Deal Health",
    },
    height=600,
)

fig.add_hline(
    y=70,
    line_dash="dash",
    annotation_text="High Risk"
)

fig.add_hline(
    y=40,
    line_dash="dash",
    annotation_text="Medium Risk"
)

fig.add_vline(
    x=60,
    line_dash="dash",
    annotation_text="High Opportunity"
)

fig.update_layout(
    xaxis=dict(range=[0, 100]),
    yaxis=dict(range=[0, 100]),
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🚨 At Risk",
        "⭐ Best Deals",
        "🎯 Next Best Actions",
        "💬 Ask Revenue Agent",
        "📋 All Deals",
    ]
)


# ==================================================
# TAB 1 — AT RISK
# ==================================================

with tab1:

    st.subheader(
        "🚨 At-Risk Deals"
    )

    st.caption(
        "Deals with medium or high risk signals."
    )

    if not risky_deals:

        st.success(
            "No medium/high-risk deals detected."
        )

    else:

        for deal in risky_deals:

            risk_level = deal.get(
                "risk_level",
                "Medium",
            )

            risk_icon = (
                "🔴"
                if risk_level == "High"
                else "🟠"
            )

            with st.expander(
                f"{risk_icon} "
                f"{deal['deal_id']} — "
                f"{deal['customer_name']} — "
                f"{inr(deal['amount_usd'])} — "
                f"Risk {deal['risk_score']}"
            ):

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Risk Score",
                    deal["risk_score"],
                )

                c2.metric(
                    "Risk Level",
                    risk_level,
                )

                c3.metric(
                    "Stage",
                    deal["stage"],
                )

                c4.metric(
                    "Days Inactive",
                    deal["days_since_activity"],
                )

                st.markdown(
                    "**Why is this deal at risk?**"
                )

                for reason in deal.get(
                    "risk_reasons",
                    [],
                ):

                    st.write(
                        f"• {reason}"
                    )

                st.write(
                    f"**Sales Rep:** "
                    f"{deal.get('sales_rep', '-')}"
                )

                st.write(
                    f"**Stakeholders:** "
                    f"{deal.get('stakeholder_count', '-')}"
                )

                buyer_status = (
                    "Engaged"
                    if deal.get(
                        "economic_buyer_engaged",
                        False,
                    )
                    else "Not engaged"
                )

                st.write(
                    f"**Economic Buyer:** "
                    f"{buyer_status}"
                )


# ==================================================
# TAB 2 — BEST DEALS
# ==================================================

with tab2:

    st.subheader(
        "⭐ Best Revenue Opportunities"
    )

    st.caption(
        "Healthy deals with strong opportunity signals."
    )

    if not best_deals:

        st.info(
            "No opportunity data available."
        )

    else:

        for deal in best_deals:

            with st.expander(
                f"⭐ {deal['deal_id']} — "
                f"{deal['customer_name']} — "
                f"{inr(deal['amount_usd'])}"
            ):

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Opportunity Score",
                    deal.get(
                        "opportunity_score",
                        "-",
                    ),
                )

                c2.metric(
                    "Stage",
                    deal.get(
                        "stage",
                        "-",
                    ),
                )

                c3.metric(
                    "Risk",
                    deal.get(
                        "risk_score",
                        "-",
                    ),
                )

                st.write(
                    f"**Sales Rep:** "
                    f"{deal.get('sales_rep', '-')}"
                )

                st.write(
                    f"**Engagement:** "
                    f"{deal.get('engagement_trend', '-')}"
                )

                buyer = (
                    "Engaged"
                    if deal.get(
                        "economic_buyer_engaged",
                        False,
                    )
                    else "Not engaged"
                )

                st.write(
                    f"**Economic Buyer:** {buyer}"
                )


# ==================================================
# TAB 3 — NEXT BEST ACTIONS
# ==================================================

with tab3:

    st.subheader(
        "🎯 High-Risk Deals — Next Best Actions"
    )

    st.caption(
        "AI-generated intervention actions for high-risk deals."
    )

    all_actions = actions_data.get(
        "actions",
        [],
    )

    # ----------------------------------------------
    # HIGH-RISK ACTIONS
    # ----------------------------------------------

    risk_actions = []

    for action in all_actions:

        category = str(
            action.get(
                "category",
                "risk",
            )
        ).lower()

        deal_id = str(
            action.get(
                "deal_id",
                ""
            )
        )

        matching_risk = [

            d for d in high_risk_deals

            if str(
                d.get("deal_id")
            ) == deal_id

        ]

        if (
            category == "risk"
            or matching_risk
        ):

            risk_actions.append(
                action
            )


    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    if not risk_actions:

        st.warning(
            "No AI next-best-action was generated for the current high-risk deals."
        )

        st.info(
            "The risk engine has identified the deals, "
            "but the AI action layer did not return an action."
        )

    else:

        for action in risk_actions:

            deal_id = action.get(
                "deal_id",
                "-",
            )

            customer = action.get(
                "customer_name",
                "-",
            )

            # Support both action schemas
            next_action = action.get(
                "next_best_action",
                action.get(
                    "action",
                    "No action generated.",
                ),
            )

            reason = action.get(
                "reason",
                action.get(
                    "action_reason",
                    "",
                ),
            )

            main_risk = action.get(
                "main_risk",
                "",
            )

            risk_score = action.get(
                "risk_score",
                None,
            )

            st.markdown(
                f"### 🚨 {deal_id} — {customer}"
            )

            if risk_score is not None:

                st.metric(
                    "Risk Score",
                    risk_score,
                )

            if main_risk:

                st.markdown(
                    f"**Main Risk:** {main_risk}"
                )

            st.warning(
                f"🎯 **Next Best Action:** {next_action}"
            )

            if reason:

                st.info(
                    f"💡 **Why now:** {reason}"
                )

            if st.button(
                "📌 Save Draft Action",
                key=f"risk_action_{deal_id}",
            ):

                draft = {

                    "deal_id": deal_id,

                    "customer_name": customer,

                    "action": next_action,

                    "reason": reason,

                }

                st.session_state.draft_actions.append(
                    draft
                )

                st.success(
                    "Action saved to Draft Actions."
                )

            st.divider()


# ==================================================
# TAB 4 — ASK REVENUE AGENT
# ==================================================

with tab4:

    st.subheader(
        "💬 Ask Revenue Agent"
    )

    st.caption(
        "Ask questions about your analyzed CRM pipeline."
    )

    # ----------------------------------------------
    # EXAMPLE QUESTIONS
    # ----------------------------------------------

    st.markdown(
        """
        **Try asking:**

        - Where should we focus this week?
        - Which high-risk deal needs immediate attention?
        - Why is DEAL-023 risky?
        - Which deals are threatening the forecast?
        - What is causing forecast uncertainty?
        - Which sales rep needs attention?
        """
    )

    # ----------------------------------------------
    # PREVIOUS CHAT
    # ----------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # ----------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------

    question = st.chat_input(
        "Ask your Revenue Agent..."
    )

    if question:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Revenue Agent is thinking..."
            ):

                try:

                    answer = ask_ai_agent(
                        question,
                        working_df,
                        result,
                    )

                    st.write(
                        answer
                    )

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.error(
                        f"AI Agent error: {e}"
                    )


# ==================================================
# TAB 5 — ALL DEALS
# ==================================================

with tab5:

    st.subheader(
        "📋 Complete Pipeline"
    )

    st.caption(
        f"{len(df)} / {len(df)} deals"
    )

    display_df = df.copy()

    # INR conversion
    if "amount_usd" in display_df.columns:

        display_df["amount_inr"] = (
            display_df["amount_usd"] * 83
        )

    columns_to_show = [
        "deal_id",
        "customer_name",
        "sales_rep",
        "stage",
        "amount_inr",
        "days_in_stage",
        "stakeholder_count",
        "economic_buyer_engaged",
        "engagement_trend",
    ]

    available_columns = [
        col
        for col in columns_to_show
        if col in display_df.columns
    ]

    display_df = display_df[
        available_columns
    ]

    display_df = display_df.rename(
        columns={
            "deal_id": "Deal",
            "customer_name": "Customer",
            "sales_rep": "Sales Rep",
            "stage": "Stage",
            "amount_inr": "Amount (INR)",
            "days_in_stage": "Days in Stage",
            "stakeholder_count": "Stakeholders",
            "economic_buyer_engaged": "Buyer Engaged",
            "engagement_trend": "Engagement",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# SAVED DRAFT ACTIONS
# ==================================================

st.divider()

st.subheader(
    "📌 Saved Draft Actions"
)

if not st.session_state.draft_actions:

    st.info(
        "No draft actions saved yet."
    )

else:

    for i, draft in enumerate(
        st.session_state.draft_actions,
        start=1,
    ):

        st.markdown(
            f"""
            **{i}. {draft.get('deal_id')} — {draft.get('customer_name')}**

            **Action:** {draft.get('action')}

            **Reason:** {draft.get('reason')}
            """
        )
def render_business_pipeline_health(df, agent_result):
    """
    Business-manager-friendly pipeline health dashboard.

    Shows:
    - Pipeline value
    - Weighted forecast
    - Risk revenue
    - High-engagement opportunities
    - Risk vs opportunity scatter
    - Stage-wise pipeline
    """

    from app.analysis.risk_deals import analyze_risks

    # -----------------------------------------
    # Analyze deals
    # -----------------------------------------
    analysis_df = analyze_risks(df.copy())

    # INR conversion
    USD_TO_INR = 83

    analysis_df["amount_inr"] = (
        analysis_df["amount_usd"] * USD_TO_INR
    )

    # -----------------------------------------
    # Business metrics
    # -----------------------------------------
    total_pipeline = analysis_df["amount_inr"].sum()

    weighted_forecast = (
        analysis_df["amount_usd"]
        * analysis_df["stage"].map({
            "Discovery": 0.15,
            "Demo": 0.30,
            "Proposal": 0.55,
            "Negotiation": 0.75,
        }).fillna(0)
    ).sum() * USD_TO_INR

    high_risk = analysis_df[
        analysis_df["risk_level"] == "High"
    ]

    medium_risk = analysis_df[
        analysis_df["risk_level"] == "Medium"
    ]

    risk_revenue = high_risk["amount_inr"].sum()

    high_engagement = analysis_df[
        (
            analysis_df["engagement_trend"]
            .astype(str)
            .str.lower()
            == "increasing"
        )
        & (analysis_df["risk_score"] < 40)
    ]

    # -----------------------------------------
    # KPI cards
    # -----------------------------------------
    st.subheader("📊 Pipeline Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Pipeline",
            f"₹{total_pipeline / 1e7:.2f} Cr"
        )

    with col2:
        st.metric(
            "Weighted Forecast",
            f"₹{weighted_forecast / 1e7:.2f} Cr"
        )

    with col3:
        st.metric(
            "At-Risk Revenue",
            f"₹{risk_revenue / 1e7:.2f} Cr",
            delta=f"{len(high_risk)} high-risk deals"
        )

    with col4:
        st.metric(
            "High Engagement",
            f"{len(high_engagement)} Deals",
            delta=f"{len(medium_risk)} medium-risk"
        )

    st.divider()

    # -----------------------------------------
    # Risk / Opportunity Matrix
    # -----------------------------------------
    st.markdown("### 🎯 Deal Health Matrix")

    st.caption(
        "High-value deals with high risk need immediate attention. "
        "Healthy high-opportunity deals represent stronger pipeline momentum."
    )

    plot_df = analysis_df.copy()

    plot_df["deal_value_cr"] = (
        plot_df["amount_inr"] / 1e7
    )

    plot_df["health"] = plot_df.apply(
        lambda row:
            "🔴 High Risk"
            if row["risk_score"] >= 70
            else (
                "🟠 Medium Risk"
                if row["risk_score"] >= 40
                else "🟢 Healthy"
            ),
        axis=1
    )

    fig = px.scatter(
        plot_df,
        x="opportunity_score",
        y="risk_score",
        size="deal_value_cr",
        color="health",
        hover_name="customer_name",
        hover_data={
            "deal_id": True,
            "sales_rep": True,
            "stage": True,
            "deal_value_cr": ":.2f",
            "risk_score": True,
            "opportunity_score": True,
            "days_since_activity": True,
            "stakeholder_count": True,
            "health": True,
        },
        labels={
            "opportunity_score": "Opportunity Score",
            "risk_score": "Risk Score",
            "deal_value_cr": "Deal Value (₹ Cr)",
            "health": "Deal Health",
        },
        height=600,
    )

    # Risk zones
    fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="High Risk"
    )

    fig.add_hline(
        y=40,
        line_dash="dash",
        annotation_text="Medium Risk"
    )

    fig.add_vline(
        x=60,
        line_dash="dash",
        annotation_text="High Opportunity"
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        legend_title="Deal Health",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------
    # Management interpretation
    # -----------------------------------------
    st.markdown("### 🧠 Management View")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔴 Immediate Attention")
        st.write(
            f"**{len(high_risk)} deals** are currently high-risk, "
            f"representing approximately "
            f"**₹{risk_revenue / 1e7:.2f} Cr** in pipeline."
        )

    with col2:
        healthy_revenue = analysis_df[
            analysis_df["risk_level"] == "Low"
        ]["amount_inr"].sum()

        st.markdown("#### 🟢 Healthy Pipeline")
        st.write(
            f"**{len(high_engagement)} deals** show increasing "
            f"engagement with low risk. Healthy pipeline value is "
            f"approximately **₹{healthy_revenue / 1e7:.2f} Cr**."
        )

    with col3:
        stalled = analysis_df[
            analysis_df["days_since_activity"] > 14
        ]

        st.markdown("#### 🟠 Stalled Activity")
        st.write(
            f"**{len(stalled)} deals** have had no recent activity "
            f"for more than 14 days."
        )

    st.divider()

    # -----------------------------------------
    # Stage-wise pipeline
    # -----------------------------------------
    st.markdown("### 📈 Pipeline by Sales Stage")

    stage_df = (
        analysis_df
        .groupby("stage")
        .agg(
            deals=("deal_id", "count"),
            pipeline=("amount_inr", "sum"),
        )
        .reset_index()
    )

    stage_df["pipeline_cr"] = (
        stage_df["pipeline"] / 1e7
    )

    fig_stage = px.bar(
        stage_df,
        x="stage",
        y="pipeline_cr",
        text="deals",
        labels={
            "stage": "Sales Stage",
            "pipeline_cr": "Pipeline Value (₹ Cr)",
            "deals": "Deals",
        },
        title="Pipeline Value by Stage",
        height=400,
    )

    fig_stage.update_traces(
        texttemplate="%{text} deals",
        textposition="outside"
    )

    st.plotly_chart(
        fig_stage,
        use_container_width=True
    )

    # -----------------------------------------
    # Risk summary table
    # -----------------------------------------
    st.markdown("### 🚨 Highest Risk Deals")

    risk_table = analysis_df[
        analysis_df["risk_score"] >= 40
    ].sort_values(
        by="risk_score",
        ascending=False
    ).head(10).copy()

    if not risk_table.empty:

        risk_table["Deal Value"] = (
            risk_table["amount_inr"] / 1e7
        ).map(lambda x: f"₹{x:.2f} Cr")

        display_risk = risk_table[
            [
                "deal_id",
                "customer_name",
                "sales_rep",
                "stage",
                "Deal Value",
                "risk_score",
                "risk_level",
                "days_since_activity",
                "engagement_trend",
            ]
        ].rename(
            columns={
                "deal_id": "Deal ID",
                "customer_name": "Customer",
                "sales_rep": "Sales Rep",
                "stage": "Stage",
                "risk_score": "Risk Score",
                "risk_level": "Risk",
                "days_since_activity": "Days Since Activity",
                "engagement_trend": "Engagement",
            }
        )

        st.dataframe(
            display_risk,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No medium/high-risk deals detected.")