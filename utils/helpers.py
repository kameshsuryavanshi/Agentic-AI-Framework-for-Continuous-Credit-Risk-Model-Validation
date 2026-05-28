import json
import pandas as pd
import numpy as np
from datetime import datetime

from utils.logger import get_logger

logger = get_logger("helpers")


# =========================================================
# SAVE DATAFRAME
# =========================================================

def save_dataframe(
    df,
    path,
    index=False
):

    logger.info(
        f"Saving dataframe -> {path}"
    )

    df.to_csv(
        path,
        index=index
    )


# =========================================================
# LOAD DATAFRAME
# =========================================================

def load_dataframe(path):

    logger.info(
        f"Loading dataframe -> {path}"
    )

    return pd.read_csv(path)


# =========================================================
# JSON SERIALIZER
# =========================================================

def convert_to_serializable(obj):

    if isinstance(
        obj,
        (np.integer,)
    ):

        return int(obj)

    elif isinstance(
        obj,
        (np.floating,)
    ):

        return float(obj)

    elif isinstance(
        obj,
        (np.ndarray,)
    ):

        return obj.tolist()

    return obj


# =========================================================
# SAVE JSON
# =========================================================

def save_json(
    data,
    path
):

    logger.info(
        f"Saving JSON -> {path}"
    )

    with open(path, "w") as f:

        json.dump(

            data,

            f,

            indent=4,

            default=convert_to_serializable

        )


# =========================================================
# LOAD JSON
# =========================================================

def load_json(path):

    logger.info(
        f"Loading JSON -> {path}"
    )

    with open(path, "r") as f:

        return json.load(f)


# =========================================================
# CURRENT TIMESTAMP
# =========================================================

def current_timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# GENERATE PORTFOLIO SUMMARY
# =========================================================

def generate_portfolio_summary(df):

    summary = {

        "total_loans":
            len(df),

        "default_rate":
            round(
                df["default_flag"].mean(),
                4
            ),

        "average_pd":
            round(
                df["pd_score"].mean(),
                4
            ),

        "total_ecl":
            round(
                df["ecl"].sum(),
                2
            ),

        "high_risk_loans":

            int(
                (df["pd_score"] > 0.25)
                .sum()
            ),

        "stage_1":

            int(
                (df["stage"] == 1)
                .sum()
            ),

        "stage_2":

            int(
                (df["stage"] == 2)
                .sum()
            ),

        "stage_3":

            int(
                (df["stage"] == 3)
                .sum()
            )

    }

    return summary


# =========================================================
# TOP RISK FEATURES
# =========================================================

def extract_top_risk_features(
    importance_df,
    top_n=10
):

    top_features = (

        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .head(top_n)

    )

    return top_features.to_dict(
        orient="records"
    )


# =========================================================
# DRIFT SUMMARY
# =========================================================

def summarize_drift(drift_metrics):

    unstable = []

    for feature, metrics in (
        drift_metrics.items()
    ):

        if metrics["drift_detected"]:

            unstable.append({

                "feature": feature,

                "psi": metrics["psi"]

            })

    return unstable


# =========================================================
# GENERATE EXECUTIVE SUMMARY
# =========================================================

def generate_executive_summary(
    portfolio_summary,
    drift_summary
):

    summary = f"""

    Executive Summary
    =================

    Total Loans:
    {portfolio_summary['total_loans']}

    Default Rate:
    {portfolio_summary['default_rate']}

    Average PD:
    {portfolio_summary['average_pd']}

    Total ECL:
    {portfolio_summary['total_ecl']}

    High Risk Loans:
    {portfolio_summary['high_risk_loans']}

    Drifted Variables:
    {len(drift_summary)}

    """

    return summary
