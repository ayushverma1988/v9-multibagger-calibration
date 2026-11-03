"""Regression tests for V11.4 standalone PIT fundamental activation gate."""
import unittest
import pandas as pd
from v11_4_historical_financial_activation_gate import audit_financial_readiness

CONFIG={"conditions":{
    "quality":{"hard_rules":[["sales_growth_5y",">",0.10],["debt_to_equity","<",1]]},
    "value":{"hard_rules":[["pe_lt_industry_pe","==",True]]},
    "earnings":{"hard_rules":[["sales_latest_ge_yoy_quarter","==",True]]},
    "rsi":{"hard_rules":[["rsi14_wilder",">",80]]}
}}

def sample():
    return pd.DataFrame({
        "date":["2024-12-31","2024-12-31","2025-06-30","2025-06-30"],
        "symbol":["A","B","A","B"],
        "historical_asof_utc":["2024-12-31T10:00:00Z"]*2+
                               ["2025-06-30T10:00:00Z"]*2,
        "source_available_utc":["2024-05-29T08:00:00Z",None,None,None],
        "has_verified_annual_yoy_source":[True,False,False,False],
        "revenue_yoy_pct":[0.15,None,None,None],
        "pat_yoy_pct":[0.18,None,None,None]
    })
class PITFinancialGateTests(unittest.TestCase):
    def test_actual_1y_values_do_not_count_as_5y_or_rsi(self):
        folds,queue,audit=audit_financial_readiness(sample(),CONFIG,expected_rows=4,expected_folds=2)
        self.assertEqual(audit["annual_revenue_and_PAT_both_covered_folds"],0)
        self.assertEqual(audit["fully_known_rule_folds"]["quality"],0)
        self.assertIn("sales_growth_5y",audit["fields_still_entirely_missing_in_historical_matrix"])
        self.assertFalse(audit["historical_financial_rule_activation_approved"])
        self.assertEqual(len(queue),2)

    def test_future_filing_is_rejected(self):
        x=sample()
        x.loc[0,"source_available_utc"]="2025-01-02T00:00:00Z"
        with self.assertRaisesRegex(ValueError,"Future/unproven"):
            audit_financial_readiness(x,CONFIG,expected_folds=2)

    def test_unsourced_numeric_is_rejected(self):
        x=sample()
        x.loc[1,"revenue_yoy_pct"]=0.65
        with self.assertRaisesRegex(ValueError,"without verified annual source"):
            audit_financial_readiness(x,CONFIG,expected_folds=2)

    def test_labels_are_never_part_of_matrix(self):
        x=sample();x["y6"]=0
        with self.assertRaisesRegex(ValueError,"future outcome"):
            audit_financial_readiness(x,CONFIG,expected_folds=2)

    def test_unchanged_frozen_universe_is_required(self):
        with self.assertRaisesRegex(ValueError,"stock universe row count"):
            audit_financial_readiness(sample(),CONFIG,expected_rows=18569,expected_folds=2)

    def test_wrong_historical_source_clock_rejected(self):
        x=sample();x.loc[0,"historical_asof_utc"]="2024-12-31T11:30:00Z"
        with self.assertRaisesRegex(ValueError,"15:30"):
            audit_financial_readiness(x,CONFIG,expected_folds=2)

if __name__=="__main__":unittest.main()
