import numpy as np
import pandas as pd

# from config import (
#     STAGE_2_DPD,
#     STAGE_3_DPD,
#     BASE_SCENARIO_WEIGHT,
#     ADVERSE_SCENARIO_WEIGHT,
#     SEVERE_SCENARIO_WEIGHT
# )
from config import settings

from utils.logger import get_logger

logger = get_logger("ecl_engine")


class IFRS9Engine:

    def __init__(self):

        pass

    # =====================================================
    # STAGE CLASSIFICATION
    # =====================================================

    def assign_stage(self, row):

        if row["default_flag"] == 1:
            return 3

        if row.get("delinq_2yrs", 0) >= settings.stage_3_dpd:
            return 3

        if row.get("delinq_2yrs", 0) >= settings.stage_2_dpd:
            return 2

        if row["pd_score"] >= 0.25:
            return 2

        return 1

    # =====================================================
    # CALCULATE LGD
    # =====================================================

    def calculate_lgd(self, row):

        base_lgd = 0.45

        if row["high_utilization_flag"] == 1:
            base_lgd += 0.10

        if row["high_dti_flag"] == 1:
            base_lgd += 0.05

        return min(base_lgd, 0.95)

    # =====================================================
    # CALCULATE EAD
    # =====================================================

    def calculate_ead(self, row):

        return row["loan_amnt"]

    # =====================================================
    # SCENARIO ADJUSTMENT
    # =====================================================

    def calculate_scenario_pd(self, pd_value):

        base = pd_value

        adverse = pd_value * 1.25

        severe = pd_value * 1.60

        weighted_pd = (

            (base * settings.base_scenario_weight) +

            (adverse * settings.adverse_scenario_weight) +

            (severe * settings.severe_scenario_weight)

        )

        return weighted_pd

    # =====================================================
    # CALCULATE ECL
    # =====================================================

    def calculate_ecl(self, row):

        pd_value = self.calculate_scenario_pd(
            row["pd_score"]
        )

        lgd = row["lgd"]

        ead = row["ead"]

        if row["stage"] == 1:

            return pd_value * lgd * ead

        elif row["stage"] == 2:

            lifetime_multiplier = 2.5

            return (
                pd_value *
                lgd *
                ead *
                lifetime_multiplier
            )

        else:

            impaired_multiplier = 4

            return (
                pd_value *
                lgd *
                ead *
                impaired_multiplier
            )

    # =====================================================
    # MAIN RUNNER
    # =====================================================

    def run(self, df):

        logger.info("Running IFRS9 engine")

        # ------------------------------------------
        # STAGE
        # ------------------------------------------

        df["stage"] = df.apply(
            self.assign_stage,
            axis=1
        )

        # ------------------------------------------
        # LGD
        # ------------------------------------------

        df["lgd"] = df.apply(
            self.calculate_lgd,
            axis=1
        )

        # ------------------------------------------
        # EAD
        # ------------------------------------------

        df["ead"] = df.apply(
            self.calculate_ead,
            axis=1
        )

        # ------------------------------------------
        # ECL
        # ------------------------------------------

        df["ecl"] = df.apply(
            self.calculate_ecl,
            axis=1
        )

        logger.info("IFRS9 calculation completed")

        return df
