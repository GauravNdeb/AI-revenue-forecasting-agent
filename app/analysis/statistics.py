import pandas as pd


def calculate_stage_win_rates(df: pd.DataFrame) -> dict:
    """
    Return historical/synthetic win probabilities by sales stage.

    For the hackathon synthetic dataset, these are predefined
    historical probabilities.
    """

    stage_rates = {
        "Discovery": 0.15,
        "Demo": 0.30,
        "Proposal": 0.55,
        "Negotiation": 0.75,
    }

    return stage_rates
