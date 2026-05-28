import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import PD_MODEL_FILE
from utils.logger import get_logger

logger = get_logger("risk_model")


class CreditRiskModel:

    def __init__(self):

        self.model = XGBClassifier(

            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1

        )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    def prepare_data(self, df):

        logger.info("Preparing train/test datasets")

        X = df.drop(columns=["default_flag"])

        y = df["default_flag"]

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=42

            )
        )

        return X_train, X_test, y_train, y_test

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    def train(self, X_train, y_train):

        logger.info("Training XGBoost model")

        self.model.fit(
            X_train,
            y_train
        )

        logger.info("Training completed")

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, X):

        return self.model.predict(X)

    # =====================================================
    # PREDICT PROBABILITY
    # =====================================================

    def predict_probability(self, X):

        probs = self.model.predict_proba(X)

        return probs[:, 1]

    # =====================================================
    # EVALUATE MODEL
    # =====================================================

    def evaluate(self, X_test, y_test):

        logger.info("Evaluating model")

        preds = self.predict(X_test)

        probs = self.predict_probability(X_test)

        metrics = {

            "auc": roc_auc_score(
                y_test,
                probs
            ),

            "accuracy": accuracy_score(
                y_test,
                preds
            ),

            "precision": precision_score(
                y_test,
                preds
            ),

            "recall": recall_score(
                y_test,
                preds
            ),

            "f1_score": f1_score(
                y_test,
                preds
            )

        }

        logger.info(f"Metrics: {metrics}")

        return metrics

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def get_feature_importance(
        self,
        feature_names
    ):

        importance = pd.DataFrame({

            "feature": feature_names,
            "importance": self.model.feature_importances_

        })

        importance = importance.sort_values(
            by="importance",
            ascending=False
        )

        return importance

    # =====================================================
    # SAVE MODEL
    # =====================================================

    def save_model(self):

        logger.info("Saving model")

        joblib.dump(
            self.model,
            PD_MODEL_FILE
        )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def load_model(self):

        logger.info("Loading model")

        self.model = joblib.load(
            PD_MODEL_FILE
        )
