from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from utils.logger import get_logger

logger = get_logger("graph_rag")


class GraphRAGPipeline:

    def __init__(self):

        logger.info("Initializing GraphRAG pipeline")

        # =====================================================
        # NEO4J CONNECTION
        # =====================================================

        self.graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database
        )

        # =====================================================
        # LLM
        # =====================================================

        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0
        )

    # =====================================================
    # QUERY GRAPH
    # =====================================================

    def query(self, cypher_query):

        try:
            logger.info("Executing graph query")

            result = self.graph.query(cypher_query)

            return {
                "success": True,
                "data": result
            }

        except Exception as e:

            logger.error(f"Graph query failed: {e}")

            return {
                "success": False,
                "error": str(e)
            }

    # =====================================================
    # NATURAL LANGUAGE GRAPH QA
    # =====================================================

    def ask(self, question):

        try:

            logger.info(f"Graph question: {question}")

            # Simple schema-aware prompting
            prompt = f"""
            You are a Credit Risk Graph AI assistant.

            Convert the following question into a Neo4j Cypher query.

            Question:
            {question}

            Return ONLY the Cypher query.
            """

            response = self.llm.invoke(prompt)

            cypher_query = response.content.strip()

            logger.info(f"Generated Cypher: {cypher_query}")

            graph_result = self.graph.query(cypher_query)

            return {
                "success": True,
                "query": cypher_query,
                "result": graph_result
            }

        except Exception as e:

            logger.error(f"Graph QA failed: {e}")

            return {
                "success": False,
                "error": str(e)
            }
