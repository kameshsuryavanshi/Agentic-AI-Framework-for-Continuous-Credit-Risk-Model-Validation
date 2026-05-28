import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.logger import get_logger

logger = get_logger("preprocessing")


class LendingClubPreprocessor:

    def __init__(self):

        self.label_encoders = {}

        self.target_col = "default_flag"

    # =====================================================
    # LOAD DATA
    # =====================================================

    def load_data(self, path):

        logger.info("Loading raw dataset")

        df = pd.read_csv(
            path,
            low_memory=False
        )

        logger.info(f"Dataset Shape: {df.shape}")

        return df

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    def remove_duplicates(self, df):

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        logger.info(
            f"Removed {before - after} duplicate rows"
        )

        return df

    # =====================================================
    # REMOVE LEAKAGE FEATURES
    # =====================================================

    def remove_leakage_columns(self, df):

        leakage_cols = [

            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_amnt",
            "last_pymnt_d",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "out_prncp",
            "out_prncp_inv",
            "hardship_flag",
            "debt_settlement_flag",
            "settlement_status",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term"

        ]

        existing_cols = [
            c for c in leakage_cols
            if c in df.columns
        ]

        df = df.drop(columns=existing_cols)

        logger.info(
            f"Removed {len(existing_cols)} leakage columns"
        )

        return df

    # =====================================================
    # CREATE TARGET
    # =====================================================

    def create_target(self, df):

        logger.info("Creating default target")

        bad_status = [

            "Charged Off",
            "Default",
            "Late (31-120 days)",
            "Late (16-30 days)"

        ]

        df[self.target_col] = df["loan_status"].apply(
            lambda x: 1 if x in bad_status else 0
        )

        return df

    # =====================================================
    # PROCESS DATE COLUMNS
    # =====================================================

    def process_dates(self, df):

        logger.info(
            "Processing date columns"
        )

        date_cols = [

            "issue_d",
            "earliest_cr_line",
            "last_credit_pull_d",
            "last_pymnt_d"

        ]

        for col in date_cols:

            if col not in df.columns:
                continue

            # ---------------------------------------------
            # Convert to string
            # ---------------------------------------------

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

            # ---------------------------------------------
            # Convert to datetime
            # LendingClub format:
            # Jan-2015
            # Feb-2018
            # ---------------------------------------------

            df[col] = pd.to_datetime(

                df[col],

                format="%b-%Y",

                errors="coerce"

            )

            # ---------------------------------------------
            # Create age feature
            # ---------------------------------------------

            df[f"{col}_months"] = (

                (
                    pd.Timestamp.today() -
                    df[col]
                ).dt.days / 30

            )

            # ---------------------------------------------
            # Fill missing
            # ---------------------------------------------

            df[f"{col}_months"] = (

                df[f"{col}_months"]

                .fillna(

                    df[f"{col}_months"]
                    .median()

                )

            )

        return df

    # =====================================================
    # EMPLOYMENT LENGTH
    # =====================================================

    def process_emp_length(self, df):

        if "emp_length" not in df.columns:
            return df

        logger.info("Processing employment length")

        df["emp_length"] = (

            df["emp_length"]
            .astype(str)
            .str.extract(r"(\\d+)")

        )

        df["emp_length"] = pd.to_numeric(
            df["emp_length"],
            errors="coerce"
        )

        return df

    # =====================================================
    # PERCENTAGE COLUMNS
    # =====================================================

    def process_percentage_columns(self, df):

        logger.info("Processing percentage columns")

        pct_cols = [

            "int_rate",
            "revol_util"

        ]

        for col in pct_cols:

            if col not in df.columns:
                continue

            df[col] = (

                df[col]
                .astype(str)
                .str.replace("%", "")

            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    # =====================================================
    # NUMERIC CLEANING
    # =====================================================

    def clean_numeric_columns(self, df):

        numeric_cols = [

            "annual_inc",
            "loan_amnt",
            "installment",
            "revol_bal",
            "dti"

        ]

        for col in numeric_cols:

            if col not in df.columns:
                continue

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        return df

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    def create_risk_features(self, df):

        logger.info("Creating risk features")

        # -----------------------------------------
        # LOAN TO INCOME
        # -----------------------------------------

        df["loan_income_ratio"] = (

            df["loan_amnt"] /
            (df["annual_inc"] + 1)

        )

        # -----------------------------------------
        # INSTALLMENT BURDEN
        # -----------------------------------------

        df["installment_income_ratio"] = (

            df["installment"] /
            ((df["annual_inc"] / 12) + 1)

        )

        # -----------------------------------------
        # REVOLVING UTILIZATION
        # -----------------------------------------

        df["revol_balance_ratio"] = (

            df["revol_bal"] /
            (df["annual_inc"] + 1)

        )

        # -----------------------------------------
        # CREDIT LINE STABILITY
        # -----------------------------------------

        if (
            "open_acc" in df.columns and
            "total_acc" in df.columns
        ):

            df["credit_line_ratio"] = (

                df["open_acc"] /
                (df["total_acc"] + 1)

            )

        # -----------------------------------------
        # HIGH UTILIZATION FLAG
        # -----------------------------------------

        if "revol_util" in df.columns:

            df["high_utilization_flag"] = (
                df["revol_util"] > 80
            ).astype(int)

        # -----------------------------------------
        # HIGH DTI FLAG
        # -----------------------------------------

        if "dti" in df.columns:

            df["high_dti_flag"] = (
                df["dti"] > 35
            ).astype(int)

        return df

    # =====================================================
    # DROP HIGH NULL COLUMNS
    # =====================================================

    def drop_high_null_columns(
        self,
        df,
        threshold=0.7
    ):

        logger.info("Dropping high null columns")

        null_ratio = df.isnull().mean()

        cols_to_drop = (
            null_ratio[null_ratio > threshold]
            .index
            .tolist()
        )

        df = df.drop(columns=cols_to_drop)

        logger.info(
            f"Dropped {len(cols_to_drop)} high-null columns"
        )

        return df

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    def handle_missing_values(self, df):

        logger.info("Handling missing values")

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        categorical_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in numeric_cols:

            median_value = df[col].median()

            df[col] = df[col].fillna(
                median_value
            )

        for col in categorical_cols:

            df[col] = df[col].fillna(
                "UNKNOWN"
            )

        return df

    # =====================================================
    # ENCODE CATEGORICALS
    # =====================================================

    def encode_categorical_columns(self, df):

        logger.info("Encoding categorical columns")

        cat_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in cat_cols:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col].astype(str)
            )

            self.label_encoders[col] = encoder

        return df

    # =====================================================
    # FEATURE SELECTION
    # =====================================================

    def select_features(self, df):

        logger.info("Selecting final features")

        important_features = [

            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "fico_range_low",
            "fico_range_high",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "mort_acc",
            "pub_rec_bankruptcies",
            "credit_history_years",
            "loan_age_months",
            "loan_income_ratio",
            "installment_income_ratio",
            "revol_balance_ratio",
            "credit_line_ratio",
            "high_utilization_flag",
            "high_dti_flag",
            self.target_col

        ]

        existing_cols = [

            c for c in important_features
            if c in df.columns

        ]

        df = df[existing_cols]

        return df

    # =====================================================
    # MEMORY OPTIMIZATION
    # =====================================================

    def optimize_memory(self, df):

        logger.info("Optimizing memory")

        for col in df.select_dtypes(
            include=["float64"]
        ).columns:

            df[col] = pd.to_numeric(
                df[col],
                downcast="float"
            )

        for col in df.select_dtypes(
            include=["int64"]
        ).columns:

            df[col] = pd.to_numeric(
                df[col],
                downcast="integer"
            )

        return df

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def run_pipeline(self, dataset_path):

        logger.info("Starting preprocessing pipeline")

        df = self.load_data(dataset_path)

        df = self.remove_duplicates(df)

        df = self.remove_leakage_columns(df)

        df = self.create_target(df)

        df = self.process_dates(df)

        df = self.process_emp_length(df)

        df = self.process_percentage_columns(df)

        df = self.clean_numeric_columns(df)

        df = self.create_risk_features(df)

        df = self.drop_high_null_columns(df)

        df = self.handle_missing_values(df)

        df = self.encode_categorical_columns(df)

        df = self.select_features(df)

        df = self.optimize_memory(df)

        logger.info("Preprocessing completed")

        logger.info(f"Final Shape: {df.shape}")

        return df
