import pandas as pd
import uuid
import math

from graph.neo4j_manager import Neo4jManager
from utils.logger import get_logger

logger = get_logger("CreditRiskGraphBuilder")

BATCH_SIZE = 5000
MAX_GRAPH_ROWS = 100000


class CreditRiskGraphBuilder:

    def __init__(self):
        self.neo4j = Neo4jManager()

    # =====================================================
    # SAFE COLUMN GETTER
    # =====================================================

    def get_first_existing_column(self, df, candidates, default=None):
        for col in candidates:
            if col in df.columns:
                return col
        return default

    # =====================================================
    # PREPARE GRAPH DATASET
    # =====================================================

    def prepare_graph_dataset(self, df):

        logger.info("Preparing optimized graph dataset")

        # -------------------------------------------------
        # Detect columns dynamically
        # -------------------------------------------------

        pd_col = self.get_first_existing_column(
            df,
            [
                "pd_score",
                "predicted_pd",
                "pd_probability",
                "predicted_probability",
                "prob_pd"
            ]
        )

        if pd_col is None:
            raise ValueError(
                f"Could not find PD column. Available columns: {df.columns.tolist()}"
            )

        logger.info(f"Using PD column: {pd_col}")

        stage_col = self.get_first_existing_column(
            df,
            ["stage", "ifrs_stage"]
        )

        loan_col = self.get_first_existing_column(
            df,
            ["loan_amnt", "loan_amount"]
        )

        # -------------------------------------------------
        # Filter important rows only
        # -------------------------------------------------

        high_risk_df = df[df[pd_col] >= 0.40]

        parts = [high_risk_df]

        if stage_col:
            stage_df = df[df[stage_col] >= 2]
            parts.append(stage_df)

        if loan_col:
            large_loans = df[df[loan_col] >= 500000]
            parts.append(large_loans)

        graph_df = pd.concat(parts).drop_duplicates()

        # -------------------------------------------------
        # Enterprise graph cap
        # -------------------------------------------------

        if len(graph_df) > MAX_GRAPH_ROWS:

            logger.info(
                f"Reducing graph rows from {len(graph_df)} to {MAX_GRAPH_ROWS}"
            )

            graph_df = graph_df.sample(
                n=MAX_GRAPH_ROWS,
                random_state=42
            )

        # -------------------------------------------------
        # Create IDs if missing
        # -------------------------------------------------

        if "borrower_id" not in graph_df.columns:
            graph_df["borrower_id"] = [
                str(uuid.uuid4()) for _ in range(len(graph_df))
            ]

        if "loan_id" not in graph_df.columns:
            graph_df["loan_id"] = [
                str(uuid.uuid4()) for _ in range(len(graph_df))
            ]

        logger.info(f"Final graph dataset size: {len(graph_df)}")

        return graph_df

    # =====================================================
    # CONSTRAINTS
    # =====================================================

    def create_constraints(self):

        logger.info("Creating database constraints")

        queries = [

            """
            CREATE CONSTRAINT borrower_id IF NOT EXISTS
            FOR (b:Borrower)
            REQUIRE b.borrower_id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT loan_id IF NOT EXISTS
            FOR (l:Loan)
            REQUIRE l.loan_id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT segment_name IF NOT EXISTS
            FOR (s:Segment)
            REQUIRE s.name IS UNIQUE
            """,

            """
            CREATE CONSTRAINT risk_level IF NOT EXISTS
            FOR (r:RiskCategory)
            REQUIRE r.level IS UNIQUE
            """
        ]

        for query in queries:
            try:
                self.neo4j.execute_query(query)
            except Exception as e:
                logger.warning(f"Constraint warning: {e}")

    # =====================================================
    # CREATE BORROWERS
    # =====================================================

    def create_borrowers(self, df):

        logger.info("Creating borrower nodes")

        income_col = self.get_first_existing_column(
            df,
            ["annual_inc", "annual_income"]
        )

        fico_col = self.get_first_existing_column(
            df,
            ["fico_range_low", "fico_score"]
        )

        total_rows = len(df)

        for i in range(0, total_rows, BATCH_SIZE):

            batch_df = df.iloc[i:i+BATCH_SIZE]

            borrowers = []

            for _, row in batch_df.iterrows():

                borrowers.append({

                    "borrower_id": str(row["borrower_id"]),

                    "annual_income": float(
                        row.get(income_col, 0)
                    ),

                    "fico_score": float(
                        row.get(fico_col, 0)
                    )
                })

            query = """

            UNWIND $rows AS row

            MERGE (b:Borrower {
                borrower_id: row.borrower_id
            })

            SET
                b.annual_income = row.annual_income,
                b.fico_score = row.fico_score

            """

            self.neo4j.execute_query(
                query,
                {"rows": borrowers}
            )

    # =====================================================
    # CREATE LOANS
    # =====================================================

    def create_loans(self, df):

        logger.info("Creating loan nodes")

        pd_col = self.get_first_existing_column(
            df,
            [
                "pd_score",
                "predicted_pd",
                "pd_probability"
            ]
        )

        total_rows = len(df)

        for i in range(0, total_rows, BATCH_SIZE):

            batch_df = df.iloc[i:i+BATCH_SIZE]

            loans = []

            for _, row in batch_df.iterrows():

                loans.append({

                    "loan_id": str(row["loan_id"]),

                    "loan_amnt": float(
                        row.get("loan_amnt", 0)
                    ),

                    "int_rate": float(
                        row.get("int_rate", 0)
                    ),

                    "dti": float(
                        row.get("dti", 0)
                    ),

                    "risk_score": float(
                        row.get(pd_col, 0)
                    ),

                    "default_flag": int(
                        row.get("default_flag", 0)
                    ),

                    "stage": int(
                        row.get("stage", 1)
                    ),

                    "ecl": float(
                        row.get("ecl", 0)
                    )
                })

            query = """

            UNWIND $rows AS row

            MERGE (l:Loan {
                loan_id: row.loan_id
            })

            SET
                l.loan_amnt = row.loan_amnt,
                l.int_rate = row.int_rate,
                l.dti = row.dti,
                l.risk_score = row.risk_score,
                l.default_flag = row.default_flag,
                l.stage = row.stage,
                l.ecl = row.ecl

            """

            self.neo4j.execute_query(
                query,
                {"rows": loans}
            )

    # =====================================================
    # CREATE SEGMENTS
    # =====================================================

    def create_segments(self, df):

        if "grade" not in df.columns:
            logger.warning("No grade column found")
            return

        segments = df["grade"].dropna().unique().tolist()

        query = """

        UNWIND $segments AS segment

        MERGE (s:Segment {
            name: segment
        })

        """

        self.neo4j.execute_query(
            query,
            {"segments": segments}
        )

    # =====================================================
    # CREATE RISK CATEGORIES
    # =====================================================

    def create_risk_categories(self):

        categories = [
            {"level": "LOW"},
            {"level": "MEDIUM"},
            {"level": "HIGH"}
        ]

        query = """

        UNWIND $rows AS row

        MERGE (r:RiskCategory {
            level: row.level
        })

        """

        self.neo4j.execute_query(
            query,
            {"rows": categories}
        )

    # =====================================================
    # CREATE RELATIONSHIPS
    # =====================================================

    def create_relationships(self, df):

        logger.info("Creating relationships")

        total_rows = len(df)

        for i in range(0, total_rows, BATCH_SIZE):

            batch_df = df.iloc[i:i+BATCH_SIZE]

            rels = []

            for _, row in batch_df.iterrows():

                rels.append({

                    "borrower_id": str(row["borrower_id"]),
                    "loan_id": str(row["loan_id"])

                })

            query = """

            UNWIND $rows AS row

            MATCH (b:Borrower {
                borrower_id: row.borrower_id
            })

            MATCH (l:Loan {
                loan_id: row.loan_id
            })

            MERGE (b)-[:HAS_LOAN]->(l)

            """

            self.neo4j.execute_query(
                query,
                {"rows": rels}
            )

        # ---------------------------------------------
        # Segment links
        # ---------------------------------------------

        if "grade" in df.columns:

            query = """

            MATCH (l:Loan)
            MATCH (s:Segment)

            WHERE l.grade = s.name

            MERGE (l)-[:BELONGS_TO]->(s)

            """

            try:
                self.neo4j.execute_query(query)
            except Exception as e:
                logger.warning(f"Segment relation warning: {e}")

        # ---------------------------------------------
        # Risk category links
        # ---------------------------------------------

        query = """

        MATCH (l:Loan)

        MATCH (r:RiskCategory)

        WHERE
            (l.risk_score < 0.2 AND r.level = 'LOW')
            OR
            (l.risk_score >= 0.2 AND l.risk_score < 0.5 AND r.level = 'MEDIUM')
            OR
            (l.risk_score >= 0.5 AND r.level = 'HIGH')

        MERGE (l)-[:HAS_RISK]->(r)

        """

        try:
            self.neo4j.execute_query(query)
        except Exception as e:
            logger.warning(f"Risk relationship warning: {e}")

    # =====================================================
    # DRIFT ANALYSIS
    # =====================================================

    def create_drift_analysis(self, drift_results):

        if not drift_results:
            return

        drifts = []

        for feature, value in drift_results.items():

            try:

                drifts.append({
                    "feature": str(feature),
                    "psi": float(value)
                })

            except:
                continue

        query = """

        UNWIND $rows AS row

        MERGE (d:DriftMetric {
            feature: row.feature
        })

        SET d.psi = row.psi

        """

        self.neo4j.execute_query(
            query,
            {"rows": drifts}
        )

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    def build_graph(self, df, drift_results):

        logger.info("Building knowledge graph...")

        self.create_constraints()

        graph_df = self.prepare_graph_dataset(df)

        self.create_borrowers(graph_df)

        self.create_loans(graph_df)

        self.create_segments(graph_df)

        self.create_risk_categories()

        self.create_relationships(graph_df)

        self.create_drift_analysis(drift_results)

        logger.info("Knowledge graph built successfully")
