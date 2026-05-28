import streamlit as st

import pandas as pd

from graph.graph_queries import GraphQueries


class DashboardComponents:

    def __init__(self):

        self.graph_queries = GraphQueries()

    # =====================================================
    # HIGH RISK TABLE
    # =====================================================

    def high_risk_table(self):

        data = (
            self.graph_queries
            .get_high_risk_loans()
        )

        df = pd.DataFrame(data)

        st.subheader(
            "High Risk Loans"
        )

        st.dataframe(df)

    # =====================================================
    # STAGE DISTRIBUTION
    # =====================================================

    def stage_distribution(self):

        data = (
            self.graph_queries
            .get_stage_distribution()
        )

        df = pd.DataFrame(data)

        st.subheader(
            "Stage Distribution"
        )

        st.bar_chart(
            df.set_index("stage")
        )

    # =====================================================
    # DRIFT TABLE
    # =====================================================

    def drift_table(self):

        data = (
            self.graph_queries
            .get_drifted_features()
        )

        df = pd.DataFrame(data)

        st.subheader(
            "Drifted Features"
        )

        st.dataframe(df)
