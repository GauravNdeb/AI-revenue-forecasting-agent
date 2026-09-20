from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.utils.data_loader import load_pipeline_data
from app.analysis.statistics import calculate_stage_win_rates
from app.analysis.forecasting import calculate_forecast
from app.analysis.risk_deals import analyze_risks
from app.graph.deal_workflow import run_deal_analysis
from app.llm.model import ask_ai


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Pipeline Revenue Forecasting & Deal-Risk Agent",
    version="1.0.0",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "crm_pipeline.csv"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Revenue Forecasting Agent is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# UPLOAD
# ============================================================

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        content = await file.read()

        temp_df = pd.read_csv(
            pd.io.common.BytesIO(content)
        )

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

        missing = [
            c for c in required_columns
            if c not in temp_df.columns
        ]

        if missing:
            return {
                "success": False,
                "error": "Required columns are missing.",
                "missing_columns": missing
            }

        DATA_PATH.write_bytes(content)

        return {
            "success": True,
            "filename": file.filename,
            "rows": len(temp_df),
            "message": "Dataset uploaded successfully."
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# FORECAST
# ============================================================

@app.get("/forecast")
def forecast():

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded."
        )

    try:

        df = load_pipeline_data(
            str(DATA_PATH)
        )

        win_rates = calculate_stage_win_rates(df)

        result = calculate_forecast(
            df,
            win_rates
        )

        return {
            "success": True,
            **result,
            "deal_count": len(df)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ALL DEALS
# ============================================================

@app.get("/deals")
def deals():

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded."
        )

    df = load_pipeline_data(
        str(DATA_PATH)
    )

    return {
        "success": True,
        "count": len(df),
        "deals": df.fillna("").to_dict(
            orient="records"
        )
    }


# ============================================================
# RISK DEALS
# ============================================================

@app.get("/risk-deals")
def risk_deals():

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded."
        )

    df = load_pipeline_data(
        str(DATA_PATH)
    )

    risk_df = analyze_risks(df)

    risk_df = risk_df[
        risk_df["risk_level"].isin(
            ["High", "Medium"]
        )
    ]

    risk_df = risk_df.sort_values(
        "risk_score",
        ascending=False
    )

    return {
        "success": True,
        "count": len(risk_df),
        "deals": risk_df.to_dict(
            orient="records"
        )
    }


# ============================================================
# AI DEAL ANALYSIS
# ============================================================

@app.get("/analyze-deal/{deal_id}")
def analyze_deal(deal_id: str):

    try:

        result = run_deal_analysis(
            str(deal_id)
        )

        return {
            "success": True,
            "deal_id": deal_id,
            "deal": result.get("deal"),
            "risk_analysis": result.get(
                "risk_analysis"
            ),
            "ai_analysis": result.get(
                "ai_analysis"
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# CHAT
# ============================================================

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):

    if not DATA_PATH.exists():
        return {
            "success": False,
            "error": "Please upload a CSV first."
        }

    try:
        question = request.question.strip()

        if not question:
            return {
                "success": False,
                "error": "Please enter a question."
            }

        # =====================================================
        # LOAD ACTUAL DATA
        # =====================================================

        df = load_pipeline_data(
            str(DATA_PATH)
        )

        win_rates = calculate_stage_win_rates(df)

        forecast_result = calculate_forecast(
            df,
            win_rates
        )

        risk_df = analyze_risks(df)

        # =====================================================
        # BASIC METRICS
        # =====================================================

        total_pipeline = float(
            df["amount_usd"].sum()
        )

        weighted_forecast = float(
            forecast_result["weighted_forecast"]
        )

        total_deals = len(df)

        high_risk_df = risk_df[
            risk_df["risk_level"] == "High"
        ]

        medium_risk_df = risk_df[
            risk_df["risk_level"] == "Medium"
        ]

        # =====================================================
        # STAGE ANALYSIS
        # =====================================================

        stage_summary = (
            df.groupby("stage")
            .agg(
                deals=("deal_id", "count"),
                pipeline=("amount_usd", "sum")
            )
            .reset_index()
        )

        # =====================================================
        # SALES REP ANALYSIS
        # =====================================================

        rep_summary = (
            df.groupby("sales_rep")
            .agg(
                deals=("deal_id", "count"),
                pipeline=("amount_usd", "sum")
            )
            .reset_index()
            .sort_values(
                "pipeline",
                ascending=False
            )
        )

        # =====================================================
        # DEAL VALUE ANALYSIS
        # =====================================================

        top_deals = (
            df.sort_values(
                "amount_usd",
                ascending=False
            )
            .head(10)
        )

        # =====================================================
        # RISK ANALYSIS
        # =====================================================

        top_risk = (
            risk_df.sort_values(
                "risk_score",
                ascending=False
            )
            .head(10)
        )

        # =====================================================
        # QUESTION NORMALIZATION
        # =====================================================

        q = question.lower()

        # =====================================================
        # 1. DEAL COUNT
        # =====================================================

        if (
            "how many deals" in q
            or "number of deals" in q
            or "deal count" in q
            or "total deals" in q
        ):

            answer = (
                f"The uploaded CRM dataset contains "
                f"**{total_deals} deals**."
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 2. FORECAST
        # =====================================================

        if (
            "forecast" in q
            or "predict" in q
            or "expected revenue" in q
            or "expected sales" in q
        ):

            answer = (
                f"### Revenue Forecast\n\n"
                f"**Total Pipeline:** "
                f"${total_pipeline:,.0f}\n\n"
                f"**Weighted Forecast:** "
                f"${weighted_forecast:,.0f}\n\n"
                f"**Forecast Range:** "
                f"${forecast_result['forecast_lower_bound']:,.0f}"
                f" – "
                f"${forecast_result['forecast_upper_bound']:,.0f}\n\n"
                f"**Confidence:** "
                f"{forecast_result['confidence']}"
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 3. TOTAL PIPELINE / REVENUE
        # =====================================================

        if (
            "total pipeline" in q
            or "pipeline value" in q
            or "pipeline worth" in q
            or "total revenue" in q
        ):

            answer = (
                f"The current total pipeline is "
                f"**${total_pipeline:,.0f}** "
                f"across **{total_deals} deals**."
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 4. HIGHEST VALUE / BEST DEAL
        # =====================================================

        if (
            "best deal" in q
            or "biggest deal" in q
            or "largest deal" in q
            or "highest value deal" in q
            or "highest-value deal" in q
        ):

            rows = top_deals.head(5)

            lines = []

            for _, row in rows.iterrows():

                risk_match = risk_df[
                    risk_df["deal_id"].astype(str)
                    == str(row["deal_id"])
                ]

                if not risk_match.empty:

                    risk_score = int(
                        risk_match.iloc[0][
                            "risk_score"
                        ]
                    )

                    risk_level = risk_match.iloc[0][
                        "risk_level"
                    ]

                else:

                    risk_score = 0
                    risk_level = "Unknown"

                lines.append(
                    f"• **{row['deal_id']}** — "
                    f"{row['customer_name']} — "
                    f"${row['amount_usd']:,.0f} — "
                    f"{row['stage']} — "
                    f"Risk {risk_score}/100 ({risk_level})"
                )

            answer = (
                "### Highest-Value Deals\n\n"
                + "\n".join(lines)
                + "\n\n"
                "These are ranked by deal value, not by "
                "probability of closing."
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 5. HIGHEST RISK
        # =====================================================

        if (
            "highest risk" in q
            or "most risky" in q
            or "riskiest" in q
            or "at highest risk" in q
        ):

            rows = top_risk.head(5)

            if rows.empty:

                answer = (
                    "No medium or high-risk deals "
                    "were detected."
                )

            else:

                lines = []

                for _, row in rows.iterrows():

                    reasons = row[
                        "risk_reasons"
                    ]

                    if isinstance(
                        reasons,
                        list
                    ):

                        reason_text = "; ".join(
                            reasons[:2]
                        )

                    else:

                        reason_text = str(
                            reasons
                        )

                    lines.append(
                        f"• **{row['deal_id']}** — "
                        f"{row['customer_name']} — "
                        f"Risk **{row['risk_score']}/100** — "
                        f"{reason_text}"
                    )

                answer = (
                    "### Highest-Risk Deals\n\n"
                    + "\n".join(lines)
                )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 6. HIGH RISK DEALS
        # =====================================================

        if (
            "high risk deals" in q
            or "high-risk deals" in q
            or "high risk deal" in q
        ):

            rows = high_risk_df.sort_values(
                "risk_score",
                ascending=False
            )

            if rows.empty:

                answer = (
                    "There are currently no "
                    "high-risk deals."
                )

            else:

                lines = []

                for _, row in rows.head(10).iterrows():

                    lines.append(
                        f"• **{row['deal_id']}** — "
                        f"{row['customer_name']} — "
                        f"${row['amount_usd']:,.0f} — "
                        f"Risk {row['risk_score']}/100"
                    )

                answer = (
                    f"There are **{len(rows)} high-risk "
                    f"deals**.\n\n"
                    + "\n".join(lines)
                )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 7. MEDIUM RISK
        # =====================================================

        if (
            "medium risk" in q
            or "medium-risk" in q
        ):

            rows = medium_risk_df.sort_values(
                "risk_score",
                ascending=False
            )

            if rows.empty:

                answer = (
                    "There are currently no "
                    "medium-risk deals."
                )

            else:

                lines = []

                for _, row in rows.head(10).iterrows():

                    lines.append(
                        f"• **{row['deal_id']}** — "
                        f"{row['customer_name']} — "
                        f"Risk {row['risk_score']}/100"
                    )

                answer = (
                    f"There are **{len(rows)} medium-risk "
                    f"deals**.\n\n"
                    + "\n".join(lines)
                )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 8. STALLED DEALS
        # =====================================================

        if (
            "stalled" in q
            or "inactive deals" in q
            or "no activity" in q
            or "inactive" in q
        ):

            stalled = df[
                df["days_since_activity"] > 14
            ].sort_values(
                "days_since_activity",
                ascending=False
            )

            if stalled.empty:

                answer = (
                    "No deals have been inactive "
                    "for more than 14 days."
                )

            else:

                lines = []

                for _, row in stalled.head(10).iterrows():

                    lines.append(
                        f"• **{row['deal_id']}** — "
                        f"{row['customer_name']} — "
                        f"{int(row['days_since_activity'])} "
                        f"days since last activity"
                    )

                answer = (
                    f"Found **{len(stalled)} stalled/inactive "
                    f"deals**.\n\n"
                    + "\n".join(lines)
                )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 9. STAGE ANALYSIS
        # =====================================================

        if (
            "stage" in q
            or "pipeline by stage" in q
            or "which stage" in q
        ):

            rows = stage_summary.sort_values(
                "pipeline",
                ascending=False
            )

            lines = []

            for _, row in rows.iterrows():

                lines.append(
                    f"• **{row['stage']}** — "
                    f"{int(row['deals'])} deals — "
                    f"${row['pipeline']:,.0f}"
                )

            answer = (
                "### Pipeline by Stage\n\n"
                + "\n".join(lines)
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 10. SALES REP
        # =====================================================

        if (
            "sales rep" in q
            or "salesperson" in q
            or "representative" in q
            or "rep pipeline" in q
            or "which rep" in q
        ):

            rows = rep_summary.head(10)

            lines = []

            for _, row in rows.iterrows():

                lines.append(
                    f"• **{row['sales_rep']}** — "
                    f"{int(row['deals'])} deals — "
                    f"${row['pipeline']:,.0f} pipeline"
                )

            answer = (
                "### Sales Rep Pipeline\n\n"
                + "\n".join(lines)
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 11. DEAL-SPECIFIC QUESTION
        # =====================================================

        found_deal = None

        for deal_id in df[
            "deal_id"
        ].astype(str):

            if deal_id.lower() in q:

                found_deal = deal_id
                break

        if found_deal:

            deal_row = df[
                df["deal_id"].astype(str)
                == found_deal
            ].iloc[0]

            risk_row = risk_df[
                risk_df["deal_id"].astype(str)
                == found_deal
            ]

            if not risk_row.empty:

                risk_row = risk_row.iloc[0]

                reasons = risk_row[
                    "risk_reasons"
                ]

                if isinstance(
                    reasons,
                    list
                ):

                    reason_text = (
                        "\n".join(
                            [
                                f"• {r}"
                                for r in reasons
                            ]
                        )
                    )

                else:

                    reason_text = str(
                        reasons
                    )

                answer = (
                    f"### {deal_row['deal_id']}\n\n"
                    f"**Customer:** "
                    f"{deal_row['customer_name']}\n\n"
                    f"**Value:** "
                    f"${deal_row['amount_usd']:,.0f}\n\n"
                    f"**Stage:** "
                    f"{deal_row['stage']}\n\n"
                    f"**Risk:** "
                    f"{risk_row['risk_score']}/100 "
                    f"({risk_row['risk_level']})\n\n"
                    f"**Risk signals:**\n"
                    f"{reason_text}"
                )

            else:

                answer = (
                    f"**{deal_row['deal_id']}** — "
                    f"{deal_row['customer_name']} — "
                    f"${deal_row['amount_usd']:,.0f} — "
                    f"{deal_row['stage']}"
                )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

        # =====================================================
        # 12. AI FOR OTHER QUESTIONS
        # =====================================================

        context = f"""
CRM DATASET SUMMARY

Total deals: {total_deals}

Total pipeline:
${total_pipeline:,.0f}

Weighted forecast:
${weighted_forecast:,.0f}

Forecast range:
${forecast_result['forecast_lower_bound']:,.0f}
to
${forecast_result['forecast_upper_bound']:,.0f}

Confidence:
{forecast_result['confidence']}

High risk deals:
{len(high_risk_df)}

Medium risk deals:
{len(medium_risk_df)}

PIPELINE BY STAGE:
{stage_summary.to_string(index=False)}

SALES REP PIPELINE:
{rep_summary.head(10).to_string(index=False)}

TOP DEALS:
{top_deals.head(10).to_string(index=False)}

TOP RISK DEALS:
{top_risk.head(10).to_string(index=False)}
"""

        prompt = f"""
You are SalesAI, a B2B revenue forecasting assistant.

Answer the user's question using the CRM data below.

IMPORTANT:
- Use actual dataset numbers.
- Do not invent customers or deals.
- If the question asks for a comparison, explain the metric used.
- Keep the answer concise and useful.
- If the question is not answerable from this dataset,
  clearly say what information is missing.

CRM DATA:

{context}

USER QUESTION:

{question}
"""

        try:

            ai_answer = ask_ai(
                prompt
            )

            return {
                "success": True,
                "answer": ai_answer,
                "source": "AI"
            }

        except Exception:

            answer = (
                "I couldn't use the AI model for this "
                "question, but the uploaded CRM data is "
                "available for questions about forecast, "
                "pipeline, stages, sales reps and deal risk."
            )

            return {
                "success": True,
                "answer": answer,
                "source": "CRM analytics"
            }

    except Exception as e:

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }
