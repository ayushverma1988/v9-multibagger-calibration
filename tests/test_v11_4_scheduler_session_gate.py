"""NSE India post-publication, non-weekend, no late-day leakage scheduler rules."""
import unittest
from datetime import datetime, timezone, timedelta
from v11_4_scheduler_session_gate import session_gate


def at(s):
    return datetime.fromisoformat(s)


class NSELocalScheduling(unittest.TestCase):
    def test_real_2026_Oct10_India_Saturday_skip(self):
        result = session_gate(at("2026-10-10T00:23:00+05:30"), "workflow_dispatch")
        self.assertFalse(result["eligible_for_source_check_only"])
        self.assertEqual(result["IST_date"], "2026-10-10")
        self.assertEqual(result["reason"], "NON_TRADING_WEEKEND_IN_ASIA_KOLKATA")

    def test_2026_Oct09_Indian_Friday_before_EOD_rejected(self):
        x = session_gate(at("2026-10-09T18:55:00+05:30"), "schedule")
        self.assertFalse(x["eligible_for_source_check_only"])
        self.assertIn("PUBLICATION_SAFETY_WINDOW", x["reason"])

    def test_2026_Oct09_Indian_Friday_late_eligible_for_verification_not_trading_proof(self):
        x = session_gate(at("2026-10-09T21:45:00+05:30"), "schedule")
        self.assertTrue(x["eligible_for_source_check_only"])
        self.assertEqual(x["IST_date"], "2026-10-09")
        self.assertFalse(x["NSE_traded_on_day_confirmed_by_gate"])
        self.assertFalse(x["exchange_holiday_calendar_confirmed_by_gate"])

    def test_utc_converts_to_indian_next_day_and_refuses_weekend(self):
        x = session_gate(at("2026-10-09T19:15:00+00:00"), "workflow_dispatch")
        self.assertEqual(x["IST_date"], "2026-10-10")
        self.assertFalse(x["eligible_for_source_check_only"])

    def test_exact_21_00_is_allowed_only_on_weekdays(self):
        x = session_gate(at("2026-10-12T21:00:00+05:30"), "schedule")
        self.assertTrue(x["eligible_for_source_check_only"])

    def test_weekday_2059_rejected(self):
        x = session_gate(at("2026-10-12T20:59:59+05:30"), "schedule")
        self.assertFalse(x["eligible_for_source_check_only"])

    def test_2026_Oct11_Sunday_always_rejected(self):
        x = session_gate(at("2026-10-11T22:10:00+05:30"), "schedule")
        self.assertFalse(x["eligible_for_source_check_only"])

    def test_missing_timezone_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"explicit timezone"):
            session_gate(datetime(2026,10,9,22), "workflow_dispatch")

    def test_unapproved_push_event_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            session_gate(at("2026-10-09T22:00:00+05:30"), "push")


if __name__ == "__main__":
    unittest.main()
