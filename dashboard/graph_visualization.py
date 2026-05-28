from langchain_google_genai import ChatGoogleGenerativeAI

from graph.graph_queries import GraphQueries

from graph.graph_rag_pipeline import GraphRAGPipeline

from config import settings

from utils.logger import get_logger

from pyvis.network import Network

logger = get_logger("graph_agents")


class GraphRiskAgent:

    def __init__(self):

        self.graph_queries = GraphQueries()

        self.graph_rag = GraphRAGPipeline()

        self.llm = ChatGoogleGenerativeAI(

            model="gemini-2.5-flash",

            google_api_key=settings.GOOGLE_API_KEY,

            temperature=0.2

        )

    # =====================================================
    # HIGH RISK ANALYSIS
    # =====================================================

    def analyze_high_risk_loans(self):

        results = (
            self.graph_queries
            .get_high_risk_loans()
        )

        prompt = f"""

        Analyze these high-risk loans.

        Data:
        {results}

        Generate:
        - portfolio risks
        - concentration analysis
        - recommendations
        - executive summary

        """

        response = self.llm.invoke(prompt)

        return response.content

    # =====================================================
    # STAGE 3 ANALYSIS
    # =====================================================

    def analyze_stage3_loans(self):

        results = (
            self.graph_queries
            .get_stage_3_loans()
        )

        prompt = f"""

        Analyze Stage 3 loans.

        Data:
        {results}

        Generate:
        - impairment analysis
        - default behavior
        - risk summary

        """

        response = self.llm.invoke(prompt)

        return response.content

    # =====================================================
    # GRAPH RAG QUERY
    # =====================================================

    def graph_reasoning(
        self,
        question
    ):

        return self.graph_rag.ask_graph(
            question
        )

    # =====================================================
    # SAVE HTML GRAPH
    # =====================================================

    def save_interactive_graph(self):

        G = self.build_graph()

        net = Network(

            height="750px",
            width="100%",
            bgcolor="#111111",
            font_color="white"

        )

        for node in G.nodes():

            net.add_node(

                str(node),

                label=str(node)

            )

        for edge in G.edges():

            net.add_edge(

                str(edge[0]),
                str(edge[1])

            )

        output_path = (
            "outputs/graphs/"
            "knowledge_graph.html"
        )

        net.save_graph(output_path)

        return output_path
