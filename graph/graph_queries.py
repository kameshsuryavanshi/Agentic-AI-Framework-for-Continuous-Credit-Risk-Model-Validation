from graph.neo4j_manager import Neo4jManager
from langchain_neo4j import Neo4jGraph


class GraphQueries:

    def __init__(self):

        self.neo4j = Neo4jManager()

    # =====================================================
    # HIGH RISK LOANS
    # =====================================================

    def get_high_risk_loans(self):

        query = """

        MATCH (l:Loan)-[:BELONGS_TO_RISK]->(
            r:RiskSegment {
                name: 'HIGH_RISK'
            }
        )

        RETURN
            l.loan_id AS loan_id,
            l.pd_score AS pd_score,
            l.ecl AS ecl

        LIMIT 20

        """

        return self.neo4j.execute_query(query)

    # =====================================================
    # STAGE 3 LOANS
    # =====================================================

    def get_stage_3_loans(self):

        query = """

        MATCH (l:Loan)-[:BELONGS_TO_STAGE]->(
            s:Stage {
                stage_id: 3
            }
        )

        RETURN
            l.loan_id,
            l.pd_score,
            l.ecl

        LIMIT 20

        """

        return self.neo4j.execute_query(query)

    # =====================================================
    # TOP DRIFT FEATURES
    # =====================================================

    def get_drifted_features(self):

        query = """

        MATCH (f:Feature)

        WHERE f.drift_detected = true

        RETURN
            f.name,
            f.psi

        ORDER BY f.psi DESC

        """

        return self.neo4j.execute_query(query)

    # =====================================================
    # PORTFOLIO STAGE DISTRIBUTION
    # =====================================================

    def get_stage_distribution(self):

        query = """

        MATCH (l:Loan)-[:BELONGS_TO_STAGE]->(
            s:Stage
        )

        RETURN
            s.name AS stage,
            COUNT(l) AS total

        """

        return self.neo4j.execute_query(query)

    # =====================================================
    # HIGH ECL LOANS
    # =====================================================

    def get_high_ecl_loans(self):

        query = """

        MATCH (l:Loan)

        RETURN
            l.loan_id,
            l.ecl,
            l.pd_score

        ORDER BY l.ecl DESC

        LIMIT 20

        """

        return self.neo4j.execute_query(query)

    # =====================================================
    # PORTFOLIO GRAPH
    # =====================================================

    def get_portfolio_graph(self):

        query = """

        MATCH (b:Borrower)-[:HAS_LOAN]->(l:Loan)

        RETURN
            b,
            l

        LIMIT 100

        """

        return self.neo4j.execute_query(query)
