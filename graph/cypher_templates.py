HIGH_RISK_QUERY = """

MATCH (b:Borrower)-[:HAS_LOAN]->(l:Loan)
      -[:BELONGS_TO_RISK]->
      (r:RiskSegment)

WHERE r.name = 'HIGH_RISK'

RETURN
    b.borrower_id,
    l.loan_id,
    l.pd_score,
    l.ecl

ORDER BY l.ecl DESC

LIMIT 25

"""

STAGE3_QUERY = """

MATCH (l:Loan)-[:BELONGS_TO_STAGE]->
      (s:Stage)

WHERE s.stage_id = 3

RETURN
    l.loan_id,
    l.pd_score,
    l.ecl

ORDER BY l.ecl DESC

LIMIT 25

"""

DRIFT_QUERY = """

MATCH (f:Feature)

WHERE f.drift_detected = true

RETURN
    f.name,
    f.psi

ORDER BY f.psi DESC

"""

PORTFOLIO_GRAPH_QUERY = """

MATCH (b:Borrower)-[:HAS_LOAN]->(l:Loan)

RETURN
    b,
    l

LIMIT 200

"""
