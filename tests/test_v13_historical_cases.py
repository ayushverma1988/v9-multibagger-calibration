from datetime import date,timedelta
import json
import unittest

from v13_historical_cases import add_months,bind_identity,decode_history,features,month_decisions,outcome


def bars():
    return [{'date':date(2023,1,1)+timedelta(days=i),'open':100.,'high':110.,'low':90.,'close':100.,'adj_close':100.,'volume':1000.} for i in range(400)]


class HistoricalCaseAudit(unittest.TestCase):
    def test_decision_uses_next_open_and_charges_both_sides(self):
        rows=bars();rows[10]['open']=500.;rows[11]['open']=120.
        rows[30].update(high=260.,close=240.,adj_close=240.)
        result=outcome(rows,10,6,date(2024,1,1),[r['date'] for r in rows])
        self.assertEqual(result['entry_date'],rows[11]['date'].isoformat())
        self.assertEqual(result['hit2'],0)
        rows[30].update(high=260.,close=243.,adj_close=243.)
        result=outcome(rows,10,6,date(2024,1,1),[r['date'] for r in rows])
        self.assertEqual(result['hit2'],1)

    def test_maturity_and_missing_session_never_become_negative_labels(self):
        rows=bars();grid=[r['date'] for r in rows]
        self.assertEqual(outcome(rows,10,12,date(2023,6,1),grid)['status'],'IMMATURE')
        incomplete=rows[:50]+rows[51:]
        result=outcome(incomplete,10,6,date(2024,1,1),grid)
        self.assertEqual(result['status'],'UNKNOWN_MISSING_OR_NONMATCHING_SESSIONS')
        self.assertIsNone(result['hit2'])
        missing_first=rows[:11]+rows[12:]
        self.assertEqual(outcome(missing_first,10,6,date(2024,1,1),grid)['status'],'UNKNOWN_FIRST_ENTRY_SESSION')

    def test_same_day_stop_and_target_are_conservatively_loss_first(self):
        rows=bars();rows[30].update(high=220.,low=50.,close=210.,adj_close=210.)
        result=outcome(rows,10,6,date(2024,1,1),[r['date'] for r in rows])
        self.assertEqual(result['hit2'],1)
        self.assertEqual(result['loss30_before_double_or_deadline'],1)
        self.assertEqual(result['double_before_loss30'],0)

    def test_features_are_causal_and_scale_invariant(self):
        rows=bars()
        for i,r in enumerate(rows):r['adj_close']=50+i
        before=features(rows,250)
        rows[300]['adj_close']=1000000
        self.assertEqual(before,features(rows,250))
        for r in rows:r['adj_close']*=.01
        after=features(rows,250)
        self.assertEqual(before['rsi14_simple'],after['rsi14_simple'])
        self.assertEqual(before['above_ma50'],after['above_ma50'])
        self.assertAlmostEqual(before['distance_ma200'],after['distance_ma200'])

    def test_official_identity_conflict_rejected(self):
        rows={'APOLLO':{'ISIN NUMBER':'INE000000001','NAME OF COMPANY':'Apollo Micro Systems Limited','SERIES':'EQ'}}
        with self.assertRaises(ValueError):bind_identity({'symbol':'APOLLO','expected_current_isin':'INE000000002'},rows)

    def test_provider_currency_and_security_must_match(self):
        raw=json.dumps({'chart':{'result':[{'meta':{'symbol':'APOLLO.NS','currency':'USD','instrumentType':'EQUITY','exchangeName':'NSI'}}]}}).encode()
        with self.assertRaises(ValueError):decode_history(raw,'APOLLO.NS',date(2026,10,29))

    def test_calendar_month_end_handles_leap_year(self):
        self.assertEqual(add_months(date(2023,8,31),6),date(2024,2,29))

    def test_stale_benchmark_does_not_certify_a_later_deadline(self):
        rows=bars();deadline=add_months(rows[11]['date'],6)
        grid=[r['date'] for r in rows if r['date']<deadline-timedelta(days=4)]
        result=outcome(rows,10,6,date(2024,1,1),grid)
        self.assertEqual(result['status'],'UNKNOWN_CALENDAR_THROUGH_DEADLINE')
        self.assertIsNone(result['hit2'])

    def test_venue_cutoff_never_creates_a_partial_month_decision(self):
        rows=bars()
        decisions=month_decisions(rows,date(2023,1,1),date(2024,1,1),before=date(2023,7,27))
        self.assertEqual(rows[decisions[-1]]['date'],date(2023,6,30))
