"""Dated financial gates for V13 selection; these do not change trained heads."""

import pandas as pd

CFO_PAT_LEGACY_MIN = .40
CFO_PAT_MIN = .35
CFO_PAT_EFFECTIVE_FROM = '2026-11-28'


def annual_cfo_pat_min(source_asof=None):
    """Use 35% from the effective session, retaining 40% for old or undated rows."""
    day = pd.Timestamp(CFO_PAT_EFFECTIVE_FROM)
    if isinstance(source_asof, pd.Series):
        asof = pd.to_datetime(source_asof).dt.normalize()
        return pd.Series(CFO_PAT_LEGACY_MIN, index=source_asof.index).where(
            ~asof.ge(day), CFO_PAT_MIN)
    if source_asof is None or pd.isna(source_asof):
        return CFO_PAT_LEGACY_MIN
    return CFO_PAT_MIN if pd.Timestamp(source_asof).normalize() >= day else CFO_PAT_LEGACY_MIN
