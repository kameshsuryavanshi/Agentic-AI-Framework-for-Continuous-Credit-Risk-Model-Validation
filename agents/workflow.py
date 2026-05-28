from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END

from utils.logger import get_logger

logger = get_logger("workflow")


# =========================================================
# WORKFLOW STATE
# =========================================================

class WorkflowState(TypedDict):

    portfolio_summary: Dict[str, Any]

    model_metrics: Dict[str, Any]

    drift_metrics: Dict[str, Any]

    bias_metrics: Dict[str, Any]

    risk_distribution: Dict[str, Any]

    risk_review_output: str

    drift_output: str

    bias_output: str

    governance_output: str

    final_report: str


# =========================================================
# VALIDATION WORKFLOW
# =========================================================

class ValidationWorkflow:

    def __init__(self, agents):

        self.agents = agents

        self.workflow = self.build_workflow()

    # =====================================================
    # RISK REVIEW NODE
    # =====================================================

    def risk_review_node(
        self,
        state: WorkflowState
    ):

        logger.info(
            "Executing Risk Review Agent"
        )

        input_text = f"""

        Portfolio Summary:
        {state['portfolio_summary']}

        Model Metrics:
        {state['model_metrics']}

        Risk Distribution:
        {state['risk_distribution']}

        Analyze the portfolio risk profile and
        summarize key observations.
        """

        result = self.agents[
            "risk_review_agent"
        ].run(input_text)

        state["risk_review_output"] = result

        return state

    # =====================================================
    # DRIFT NODE
    # =====================================================

    def drift_node(
        self,
        state: WorkflowState
    ):

        logger.info(
            "Executing Drift Agent"
        )

        input_text = f"""

        Drift Metrics:
        {state['drift_metrics']}

        Analyze drift behavior,
        unstable variables,
        and model stability.
        """

        result = self.agents[
            "drift_agent"
        ].run(input_text)

        state["drift_output"] = result

        return state

    # =====================================================
    # BIAS NODE
    # =====================================================

    def bias_node(
        self,
        state: WorkflowState
    ):

        logger.info(
            "Executing Bias Agent"
        )

        input_text = f"""

        Bias Metrics:
        {state['bias_metrics']}

        Analyze fairness risks,
        portfolio disparities,
        and bias observations.
        """

        result = self.agents[
            "bias_agent"
        ].run(input_text)

        state["bias_output"] = result

        return state

    # =====================================================
    # GOVERNANCE NODE
    # =====================================================

    def governance_node(
        self,
        state: WorkflowState
    ):

        logger.info(
            "Executing Governance Agent"
        )

        input_text = f"""

        Portfolio Summary:
        {state['portfolio_summary']}

        Drift Analysis:
        {state['drift_output']}

        Bias Analysis:
        {state['bias_output']}

        Provide governance observations,
        validation concerns,
        and regulatory alignment commentary.
        """

        result = self.agents[
            "governance_agent"
        ].run(

            input_text,
            use_rag=True

        )

        state["governance_output"] = result

        return state

    # =====================================================
    # FINAL REPORT NODE
    # =====================================================

    def reporting_node(
        self,
        state: WorkflowState
    ):

        logger.info(
            "Executing Reporting Agent"
        )

        input_text = f"""

        Create a complete AI-driven
        portfolio validation report.

        Include:
        - Executive Summary
        - Portfolio Overview
        - Risk Assessment
        - Drift Analysis
        - Bias Analysis
        - Governance Observations
        - AI Recommendations

        INPUTS:

        Portfolio Summary:
        {state['portfolio_summary']}

        Model Metrics:
        {state['model_metrics']}

        Risk Review:
        {state['risk_review_output']}

        Drift Analysis:
        {state['drift_output']}

        Bias Analysis:
        {state['bias_output']}

        Governance Analysis:
        {state['governance_output']}
        """

        result = self.agents[
            "reporting_agent"
        ].run(input_text)

        state["final_report"] = result

        return state

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    def build_workflow(self):

        logger.info(
            "Building LangGraph workflow"
        )

        workflow = StateGraph(
            WorkflowState
        )

        # -----------------------------------------
        # ADD NODES
        # -----------------------------------------

        workflow.add_node(
            "risk_review",
            self.risk_review_node
        )

        workflow.add_node(
            "drift_analysis",
            self.drift_node
        )

        workflow.add_node(
            "bias_analysis",
            self.bias_node
        )

        workflow.add_node(
            "governance_review",
            self.governance_node
        )

        workflow.add_node(
            "final_reporting",
            self.reporting_node
        )

        # -----------------------------------------
        # FLOW
        # -----------------------------------------

        workflow.set_entry_point(
            "risk_review"
        )

        workflow.add_edge(
            "risk_review",
            "drift_analysis"
        )

        workflow.add_edge(
            "drift_analysis",
            "bias_analysis"
        )

        workflow.add_edge(
            "bias_analysis",
            "governance_review"
        )

        workflow.add_edge(
            "governance_review",
            "final_reporting"
        )

        workflow.add_edge(
            "final_reporting",
            END
        )

        return workflow.compile()

    # =====================================================
    # RUN WORKFLOW
    # =====================================================

    def run(self, state):

        logger.info(
            "Starting validation workflow"
        )

        result = self.workflow.invoke(
            state
        )

        logger.info(
            "Workflow execution completed"
        )

        return result
