from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL
)

from agents.rag_pipeline import RegulatoryRAG

from utils.logger import get_logger

logger = get_logger("ai_agents")


# =========================================================
# BASE AGENT
# =========================================================

class BaseAgent:

    def __init__(
        self,
        name,
        system_prompt,
        vector_store=None
    ):

        self.name = name

        self.system_prompt = system_prompt

        self.vector_store = vector_store

        self.rag = RegulatoryRAG()

        self.llm = ChatGoogleGenerativeAI(

            model=GEMINI_MODEL,

            google_api_key=GOOGLE_API_KEY,

            temperature=0.2

        )

    # =====================================================
    # GENERATE
    # =====================================================

    def run(
        self,
        user_input,
        use_rag=False
    ):

        context = ""

        if use_rag and self.vector_store:

            context = self.rag.retrieve_context(

                self.vector_store,
                user_input

            )

        final_prompt = f"""

        SYSTEM ROLE:
        {self.system_prompt}

        REGULATORY CONTEXT:
        {context}

        USER INPUT:
        {user_input}

        RESPONSE:
        """

        response = self.llm.invoke(
            final_prompt
        )

        return response.content


# =========================================================
# AGENT FACTORY
# =========================================================

class AgentFactory:

    def __init__(self, vector_store):

        self.vector_store = vector_store

    # =====================================================
    # BUILD AGENTS
    # =====================================================

    def build_agents(self):

        agents = {

            "risk_review_agent": BaseAgent(

                name="RiskReviewAgent",

                system_prompt="""
                You are a senior AI risk analyst.

                Your responsibilities:
                - Analyze portfolio risk
                - Review model metrics
                - Summarize risk behavior
                - Explain risk concentration
                - Generate validation commentary
                """

            ),

            "drift_agent": BaseAgent(

                name="DriftAgent",

                system_prompt="""
                You are a model monitoring expert.

                Your responsibilities:
                - Analyze drift metrics
                - Detect instability
                - Explain PSI movement
                - Identify unstable variables
                """

            ),

            "bias_agent": BaseAgent(

                name="BiasAgent",

                system_prompt="""
                You are an AI fairness specialist.

                Your responsibilities:
                - Analyze bias metrics
                - Review fairness issues
                - Identify risk disparities
                - Summarize fairness observations
                """

            ),

            "governance_agent": BaseAgent(

                name="GovernanceAgent",

                system_prompt="""
                You are an AI governance specialist.

                Your responsibilities:
                - Analyze governance implications
                - Review regulatory alignment
                - Summarize model governance observations
                - Generate governance commentary
                """,

                vector_store=self.vector_store

            ),

            "reporting_agent": BaseAgent(

                name="ReportingAgent",

                system_prompt="""
                You are a senior portfolio reporting expert.

                Your responsibilities:
                - Combine agent outputs
                - Generate final portfolio report
                - Create executive summary
                - Generate AI-driven recommendations
                """

            )

        }

        return agents
