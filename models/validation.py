import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from config import (
    PSI_THRESHOLD,
    HIGH_RISK_THRESHOLD
)

from utils.logger import get_logger

logger = get_logger("validation")


class ValidationEngine:

    def __init__(self):

        pass

    # =====================================================
    # POPULATION STABILITY INDEX
    # =====================================================

    def calculate_psi(
        self,
        expected,
        actual,
        buckets=10
    ):

        expected = np.array(expected)
        actual = np.array(actual)

        breakpoints = np.arange(
            0,
            buckets + 1
        ) / buckets

        breakpoints = np.percentile(
            expected,
            breakpoints * 100
        )

        expected_counts = np.histogram(
            expected,
            bins=breakpoints
        )[0]

        actual_counts = np.histogram(
            actual,
            bins=breakpoints
        )[0]

        expected_percents = (
            expected_counts / len(expected)
        )

        actual_percents = (
            actual_counts / len(actual)
        )

        psi_values = []

        for e, a in zip(
            expected_percents,
            actual_percents
        ):

            if e == 0:
                e = 0.0001

            if a == 0:
                a = 0.0001

            psi = (
                (e - a) *
                np.log(e / a)
            )

            psi_values.append(psi)

        return np.sum(psi_values)

    # =====================================================
    # DRIFT ANALYSIS
    # =====================================================

    def drift_analysis(self, df):

        logger.info("Running drift analysis")

        current = df[df["stage"] == 1]

        shifted = df[df["stage"] >= 2]

        drift_metrics = {}

        monitored_cols = [

            "annual_inc",
            "loan_amnt",
            "dti",
            "revol_util",
            "pd_score"

        ]

        for col in monitored_cols:

            if col not in df.columns:
                continue

            psi = self.calculate_psi(
                current[col],
                shifted[col]
            )

            ks_stat = ks_2samp(
                current[col],
                shifted[col]
            )[0]

            drift_metrics[col] = {

                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "drift_detected": (
                    psi > PSI_THRESHOLD
                )

            }

        return drift_metrics

    # =====================================================
    # BIAS ANALYSIS
    # =====================================================

    def bias_analysis(self, df):

        logger.info("Running bias analysis")

        bias_metrics = {}

        if "home_ownership" in df.columns:

            groups = (
                df.groupby("home_ownership")
                ["default_flag"]
                .mean()
            )

            bias_metrics["home_ownership"] = (
                groups.to_dict()
            )

        if "grade" in df.columns:

            groups = (
                df.groupby("grade")
                ["default_flag"]
                .mean()
            )

            bias_metrics["grade"] = (
                groups.to_dict()
            )

        return bias_metrics

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    def risk_distribution(self, df):

        logger.info("Generating risk distribution")

        high_risk = (
            df["pd_score"] >= HIGH_RISK_THRESHOLD
        ).sum()

        medium_risk = (

            (df["pd_score"] >= 0.10) &
            (df["pd_score"] < HIGH_RISK_THRESHOLD)

        ).sum()

        low_risk = (
            df["pd_score"] < 0.10
        ).sum()

        return {

            "high_risk": int(high_risk),
            "medium_risk": int(medium_risk),
            "low_risk": int(low_risk)

        }

    # =====================================================
    # PORTFOLIO SUMMARY
    # =====================================================

    def portfolio_summary(self, df):

        logger.info("Generating portfolio summary")

        summary = {

            "total_loans": len(df),

            "default_rate": round(
                df["default_flag"].mean(),
                4
            ),

            "avg_pd": round(
                df["pd_score"].mean(),
                4
            ),

            "total_ecl": round(
                df["ecl"].sum(),
                2
            ),

            "stage_distribution": (

                df["stage"]
                .value_counts()
                .to_dict()

            )

        }

        return summary

    # =====================================================
    # TOP RISK SEGMENTS
    # =====================================================

    def top_risk_segments(self, df):

        logger.info("Analyzing top risk segments")

        segments = {}

        if "grade" in df.columns:

            grade_risk = (

                df.groupby("grade")
                ["pd_score"]
                .mean()
                .sort_values(
                    ascending=False
                )

            )

            segments["grade"] = (
                grade_risk.head(5).to_dict()
            )

        return segments

    # =====================================================
    # MAIN VALIDATION RUNNER
    # =====================================================

    def run_validation(self, df):

        logger.info("Starting validation engine")

        results = {

            "portfolio_summary":
                self.portfolio_summary(df),

            "drift":
                self.drift_analysis(df),

            "bias":
                self.bias_analysis(df),

            "risk_distribution":
                self.risk_distribution(df),

            "top_risk_segments":
                self.top_risk_segments(df)

        }

        logger.info("Validation completed")

        return results
