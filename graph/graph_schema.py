GRAPH_SCHEMA = """

// =========================================================
// BORROWER
// =========================================================

CREATE CONSTRAINT borrower_id IF NOT EXISTS
FOR (b:Borrower)
REQUIRE b.borrower_id IS UNIQUE;

// =========================================================
// LOAN
// =========================================================

CREATE CONSTRAINT loan_id IF NOT EXISTS
FOR (l:Loan)
REQUIRE l.loan_id IS UNIQUE;

// =========================================================
// FEATURE
// =========================================================

CREATE CONSTRAINT feature_name IF NOT EXISTS
FOR (f:Feature)
REQUIRE f.name IS UNIQUE;

// =========================================================
// SEGMENT
// =========================================================

CREATE CONSTRAINT segment_name IF NOT EXISTS
FOR (s:RiskSegment)
REQUIRE s.name IS UNIQUE;

// =========================================================
// REGULATION
// =========================================================

CREATE CONSTRAINT regulation_name IF NOT EXISTS
FOR (r:Regulation)
REQUIRE r.name IS UNIQUE;

// =========================================================
// MACRO ECONOMIC
// =========================================================

CREATE CONSTRAINT macro_name IF NOT EXISTS
FOR (m:MacroFactor)
REQUIRE m.name IS UNIQUE;

"""
