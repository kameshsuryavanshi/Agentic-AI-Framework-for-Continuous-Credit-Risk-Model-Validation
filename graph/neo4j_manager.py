# from neo4j import GraphDatabase
# from config import settings

# from utils.logger import get_logger

# logger = get_logger("neo4j_manager")


# class Neo4jManager:

#     def __init__(self):

#         self.driver = GraphDatabase.driver(

#             settings.neo4j_uri,

#             auth=(
#                 settings.NEO4J_USERNAME,
#                 settings.NEO4J_PASSWORD
#             )

#         )

#         logger.info(
#             "Neo4j connection initialized"
#         )

#     # =====================================================
#     # EXECUTE QUERY
#     # =====================================================

#     def execute_query(
#         self,
#         query,
#         parameters=None
#     ):

#         with self.driver.session() as session:

#             result = session.run(
#                 query,
#                 parameters or {}
#             )

#             return [
#                 record.data()
#                 for record in result
#             ]

#     # =====================================================
#     # CREATE CONSTRAINTS
#     # =====================================================

#     def create_constraints(self):

#         constraints = [

#             """
#             CREATE CONSTRAINT borrower_id IF NOT EXISTS
#             FOR (b:Borrower)
#             REQUIRE b.borrower_id IS UNIQUE
#             """,

#             """
#             CREATE CONSTRAINT loan_id IF NOT EXISTS
#             FOR (l:Loan)
#             REQUIRE l.loan_id IS UNIQUE
#             """,

#             """
#             CREATE CONSTRAINT feature_name IF NOT EXISTS
#             FOR (f:Feature)
#             REQUIRE f.name IS UNIQUE
#             """,

#             """
#             CREATE CONSTRAINT regulation_name IF NOT EXISTS
#             FOR (r:Regulation)
#             REQUIRE r.name IS UNIQUE
#             """

#         ]

#         with self.driver.session() as session:

#             for constraint in constraints:

#                 session.run(constraint)

#         logger.info(
#             "Neo4j constraints created"
#         )

#     # =====================================================
#     # CLOSE CONNECTION
#     # =====================================================

#     def close(self):

#         self.driver.close()

#         logger.info(
#             "Neo4j connection closed"
#         )
from neo4j import GraphDatabase
from config import settings
from utils.logger import get_logger

logger = get_logger("Neo4jManager")


class Neo4jManager:

    def __init__(self):

        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(
                settings.neo4j_username,
                settings.neo4j_password
            ),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_timeout=30
        )

        logger.info(
            f"Connected to Neo4j at {settings.neo4j_uri}"
        )

    # =====================================================
    # EXECUTE QUERY
    # =====================================================

    def execute_query(
        self,
        query,
        params=None
    ):

        try:

            with self.driver.session(
                database=settings.neo4j_database
            ) as session:

                result = session.run(
                    query,
                    params or {}
                )

                return [r.data() for r in result]

        except Exception as e:

            logger.error(
                f"Query execution failed: {e}"
            )

            raise e

    # =====================================================
    # BATCH QUERY
    # =====================================================

    def execute_batch_query(
        self,
        query,
        rows,
        batch_size=5000
    ):

        total = len(rows)

        logger.info(
            f"Processing {total} rows in batches..."
        )

        for i in range(0, total, batch_size):

            batch = rows[i:i + batch_size]

            try:

                with self.driver.session(
                    database=settings.neo4j_database
                ) as session:

                    session.run(
                        query,
                        {"rows": batch}
                    )

                logger.info(
                    f"Inserted batch {i} -> {i + len(batch)}"
                )

            except Exception as e:

                logger.error(
                    f"Batch insert failed: {e}"
                )

                raise e

    # =====================================================
    # CREATE CONSTRAINTS
    # =====================================================

    def create_constraints(self):

        logger.info(
            "Creating database constraints..."
        )

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
            """
        ]

        for q in queries:
            self.execute_query(q)

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        self.driver.close()
