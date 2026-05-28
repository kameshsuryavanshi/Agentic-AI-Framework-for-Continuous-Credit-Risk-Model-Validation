# from pathlib import Path
# from agents.graph_agents import GraphRiskAgent
# from graph.graph_builder import CreditRiskGraphBuilder
# from models.preprocessing import LendingClubPreprocessor
# from models.risk_model import CreditRiskModel
# from models.ecl_engine import IFRS9Engine
# from models.validation import ValidationEngine

# from agents.rag_pipeline import RegulatoryRAG
# from agents.ai_agents import AgentFactory
# from agents.workflow import ValidationWorkflow

# from utils.logger import get_logger

# from config import (
#     RAW_DATA_DIR,
#     REPORT_DIR
# )

# import pandas as pd

# logger = get_logger("main")

# # =========================================================
# # 1. LOAD DATA
# # =========================================================
# logger.info("Loading dataset...")

# dataset_path = RAW_DATA_DIR / "lending_club" / "accepted_2007_to_2018Q4.csv"

# # =========================================================
# # 2. PREPROCESSING
# # =========================================================
# logger.info("Starting preprocessing...")

# preprocessor = LendingClubPreprocessor()
# processed_path = Path("data/processed/final_dataset.parquet")

# if processed_path.exists():
#     logger.info("Loading processed dataset from cache...")
#     df = pd.read_parquet(processed_path)
# else:
#     logger.info("Running preprocessing pipeline...")
#     df = preprocessor.run_pipeline(dataset_path)

#     processed_path.parent.mkdir(parents=True, exist_ok=True)
#     df.to_parquet(processed_path, index=False)
#     logger.info(f"Dataset saved to {processed_path}")

# # =========================================================
# # 3. TRAIN MODEL (PD)
# # =========================================================
# logger.info("Training PD model...")

# risk_model = CreditRiskModel()
# X_train, X_test, y_train, y_test = risk_model.prepare_data(df)

# risk_model.train(X_train, y_train)
# metrics = risk_model.evaluate(X_test, y_test)

# logger.info(f"Model Metrics: {metrics}")

# # Add PD scores to the dataframe
# df["pd_score"] = risk_model.predict_probability(
#     df.drop(columns=["default_flag"]))

# # =========================================================
# # 4. ECL ENGINE
# # =========================================================
# logger.info("Running ECL engine...")
# ecl_engine = IFRS9Engine()
# df = ecl_engine.run(df)

# # =========================================================
# # 5. VALIDATION
# # =========================================================
# logger.info("Running validation engine...")
# validator = ValidationEngine()
# validation_results = validator.run_validation(df)

# # =========================================================
# # 6. RAG SYSTEM
# # =========================================================
# logger.info("Initializing RAG pipeline...")
# rag = RegulatoryRAG()
# vector_store = rag.build_or_load_vectorstore()

# # =========================================================
# # 7. AGENTS & WORKFLOW SETUP
# # =========================================================
# logger.info("Initializing agents...")
# agent_factory = AgentFactory(vector_store)
# agents = agent_factory.build_agents()
# workflow = ValidationWorkflow(agents)

# # =========================================================
# # 8. GRAPH CONSTRUCTION & ANALYSIS (NEW INTEGRATION)
# # =========================================================
# logger.info("Building Knowledge Graph...")

# graph_builder = CreditRiskGraphBuilder()
# # Ensure we pass the correct drift key (check if it's 'drift' or 'drift_analysis' in your validation results)
# drift_data = validation_results.get(
#     "drift_analysis", validation_results.get("drift", {}))

# graph_builder.build_graph(df=df, drift_results=drift_data)

# logger.info("Running Graph AI Agents...")
# graph_agent = GraphRiskAgent()
# graph_analysis = graph_agent.analyze_high_risk_loans()

# # IMPORTANT: Integrate Graph Analysis into the workflow state
# # This ensures the agents "see" the graph findings
# graph_risk_summary = {
#     "high_risk_borrowers": graph_analysis.get("high_risk_borrowers", []),
#     "risk_clusters": graph_analysis.get("risk_clusters", []),
#     "macro_drift_links": graph_analysis.get("macro_drift_links", [])
# }

# # =========================================================
# # 9. PREPARE WORKFLOW STATE
# # =========================================================
# workflow_state = {
#     "portfolio_summary": validation_results.get("summary", {}),
#     "model_metrics": metrics,
#     "drift_metrics": validation_results.get("drift", {}),
#     "bias_metrics": validation_results.get("bias", {}),
#     "risk_distribution": validation_results.get("risk_distribution", {}),
#     "graph_insights": graph_risk_summary  # <-- ADDED GRAPH DATA HERE
# }

# # =========================================================
# # 10. RUN WORKFLOW
# # =========================================================
# logger.info("Executing multi-agent workflow...")
# final_report = workflow.run(workflow_state)

# # =========================================================
# # 11. SAVE REPORT
# # =========================================================
# report_path = REPORT_DIR / "final_validation_report.txt"
# report_path.parent.mkdir(parents=True, exist_ok=True)

# with open(report_path, "w") as f:
#     f.write(final_report["final_report"])

# logger.info("Final report generated successfully.")

# print("\n")
# print("=" * 80)
# print("FINAL AI VALIDATION REPORT")
# print("=" * 80)
# print("\n")
# print(final_report["final_report"])


from pathlib import Path
from agents.graph_agents import GraphRiskAgent
from graph.graph_builder import CreditRiskGraphBuilder
from models.preprocessing import LendingClubPreprocessor
from models.risk_model import CreditRiskModel
from models.ecl_engine import IFRS9Engine
from models.validation import ValidationEngine

from agents.rag_pipeline import RegulatoryRAG
from agents.ai_agents import AgentFactory
from agents.workflow import ValidationWorkflow

from utils.logger import get_logger

from config import (
    RAW_DATA_DIR,
    REPORT_DIR
)

import pandas as pd

logger = get_logger("main")

# =========================================================
# 1. LOAD DATA
# =========================================================
logger.info("Loading dataset...")

dataset_path = RAW_DATA_DIR / "lending_club" / "accepted_2007_to_2018Q4.csv"

# =========================================================
# 2. PREPROCESSING
# =========================================================
logger.info("Starting preprocessing...")

preprocessor = LendingClubPreprocessor()
processed_path = Path("data/processed/final_dataset.parquet")

if processed_path.exists():
    logger.info("Loading processed dataset from cache...")
    df = pd.read_parquet(processed_path)
else:
    logger.info("Running preprocessing pipeline...")
    df = preprocessor.run_pipeline(dataset_path)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.info(f"Dataset saved to {processed_path}")

# =========================================================
# 3. TRAIN MODEL (PD)
# =========================================================
logger.info("Training PD model...")

risk_model = CreditRiskModel()
X_train, X_test, y_train, y_test = risk_model.prepare_data(df)

risk_model.train(X_train, y_train)
metrics = risk_model.evaluate(X_test, y_test)

logger.info(f"Model Metrics: {metrics}")

# Add PD scores to the dataframe
df["pd_score"] = risk_model.predict_probability(
    df.drop(columns=["default_flag"]))

# =========================================================
# 4. ECL ENGINE
# =========================================================
logger.info("Running ECL engine...")
ecl_engine = IFRS9Engine()
df = ecl_engine.run(df)

# =========================================================
# 5. VALIDATION
# =========================================================
logger.info("Running validation engine...")
validator = ValidationEngine()
validation_results = validator.run_validation(df)

# =========================================================
# 6. RAG SYSTEM
# =========================================================
logger.info("Initializing RAG pipeline...")
rag = RegulatoryRAG()
vector_store = rag.build_or_load_vectorstore()

# =========================================================
# 7. AGENTS & WORKFLOW SETUP
# =========================================================
logger.info("Initializing agents...")
agent_factory = AgentFactory(vector_store)
agents = agent_factory.build_agents()
workflow = ValidationWorkflow(agents)

# =========================================================
# 8. GRAPH CONSTRUCTION & ANALYSIS (NEW INTEGRATION)
# =========================================================
logger.info("Building Knowledge Graph...")

graph_builder = CreditRiskGraphBuilder()
# Ensure we pass the correct drift key (check if it's 'drift' or 'drift_analysis' in your validation results)
drift_data = validation_results.get(
    "drift_analysis", validation_results.get("drift", {}))

graph_builder.build_graph(df=df, drift_results=drift_data)

logger.info("Running Graph AI Agents...")
graph_agent = GraphRiskAgent()
graph_analysis = graph_agent.analyze_high_risk_loans()

# IMPORTANT: Integrate Graph Analysis into the workflow state
# This ensures the agents "see" the graph findings
graph_risk_summary = {
    "high_risk_borrowers": graph_analysis.get("high_risk_borrowers", []),
    "risk_clusters": graph_analysis.get("risk_clusters", []),
    "macro_drift_links": graph_analysis.get("macro_drift_links", [])
}

# =========================================================
# 9. PREPARE WORKFLOW STATE
# =========================================================
workflow_state = {
    "portfolio_summary": validation_results.get("summary", {}),
    "model_metrics": metrics,
    "drift_metrics": validation_results.get("drift", {}),
    "bias_metrics": validation_results.get("bias", {}),
    "risk_distribution": validation_results.get("risk_distribution", {}),
    "graph_insights": graph_risk_summary  # <-- ADDED GRAPH DATA HERE
}

# =========================================================
# 10. RUN WORKFLOW
# =========================================================
logger.info("Executing multi-agent workflow...")
final_report = workflow.run(workflow_state)

# =========================================================
# 11. SAVE REPORT
# =========================================================
report_path = REPORT_DIR / "final_validation_report.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w") as f:
    f.write(final_report["final_report"])

logger.info("Final report generated successfully.")

print("\n")
print("=" * 80)
print("FINAL AI VALIDATION REPORT")
print("=" * 80)
print("\n")
print(final_report["final_report"])
