from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from models.risk_model import CreditRiskModel
from models.ecl_engine import IFRS9Engine

from utils.logger import get_logger

logger = get_logger("fastapi")

app = FastAPI(

    title="Agentic AI Credit Risk API",

    version="1.0.0"

)

# =========================================================
# LOAD MODEL
# =========================================================

risk_model = CreditRiskModel()

risk_model.load_model()

ecl_engine = IFRS9Engine()

# =========================================================
# REQUEST MODEL
# =========================================================


class LoanRequest(BaseModel):

    loan_amnt: float

    annual_inc: float

    dti: float

    revol_util: float

    installment: float

    fico_range_low: int

    fico_range_high: int

    open_acc: int

    revol_bal: float

# =========================================================
# HEALTH CHECK
# =========================================================


@app.get("/")
def home():

    return {

        "message":
        "Agentic AI Credit Risk API Running"

    }

# =========================================================
# PREDICT
# =========================================================


@app.post("/predict")
def predict(request: LoanRequest):

    logger.info("Received prediction request")

    data = pd.DataFrame([{

        "loan_amnt": request.loan_amnt,

        "annual_inc": request.annual_inc,

        "dti": request.dti,

        "revol_util": request.revol_util,

        "installment": request.installment,

        "fico_range_low": request.fico_range_low,

        "fico_range_high": request.fico_range_high,

        "open_acc": request.open_acc,

        "revol_bal": request.revol_bal,

        "loan_income_ratio":
            request.loan_amnt /
            (request.annual_inc + 1),

        "installment_income_ratio":
            request.installment /
            ((request.annual_inc / 12) + 1),

        "revol_balance_ratio":
            request.revol_bal /
            (request.annual_inc + 1),

        "high_utilization_flag":
            int(request.revol_util > 80),

        "high_dti_flag":
            int(request.dti > 35)

    }])

    pd_score = float(

        risk_model.predict_probability(
            data
        )[0]

    )

    data["pd_score"] = pd_score

    data["default_flag"] = 0

    data["stage"] = data.apply(
        ecl_engine.assign_stage,
        axis=1
    )

    data["lgd"] = data.apply(
        ecl_engine.calculate_lgd,
        axis=1
    )

    data["ead"] = data.apply(
        ecl_engine.calculate_ead,
        axis=1
    )

    data["ecl"] = data.apply(
        ecl_engine.calculate_ecl,
        axis=1
    )

    return {

        "pd_score":
            round(pd_score, 4),

        "stage":
            int(data["stage"].iloc[0]),

        "lgd":
            round(
                float(data["lgd"].iloc[0]),
                4
            ),

        "ead":
            round(
                float(data["ead"].iloc[0]),
                2
            ),

        "ecl":
            round(
                float(data["ecl"].iloc[0]),
                2
            )

    }
