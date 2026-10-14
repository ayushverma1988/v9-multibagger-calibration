"""Conservative business-event taxonomy and exact numerical source binding.

These are evidence repairs, not fitted probability improvements. Older frozen
registries are never edited; cached announcement flags are rechecked at scoring.
"""
import re
import numpy as np

LEGAL=r'order\s+of\s+(?:seizure|attachment)|\bseizure\b|\bcourt\b|\btribunal\b|\badjudicat\w*\b|\benforcement\b|\bFEMA\b|\bpenalt\w*\b|orders?\s+passed|action\(s\)\s+(?:initiated|taken)'

def business_order_status(text,headline=''):
    value=re.sub(r'\s+',' ',str(headline)+' '+str(text))
    if re.search(LEGAL,value,re.I):return 'LEGAL_REGULATORY_NOT_BUSINESS_ORDER'
    if re.search(r'\b(?:cancelled|canceled|terminated|termination|withdrawn|revoked)\b',value,re.I):return 'CANCELLED_OR_WITHDRAWN_NOT_ORDER_WIN'
    if re.search(r'\b(?:plan|plans|proposed|intend|intends|bid|bidding|tendering)\b',value,re.I):return 'INTENT_OR_TENDER_NOT_FIRM_ORDER'
    positive=(r'bagging\s*/?\s*receiving\s+of\s+orders?/contracts?|'
              r'bagging|receiving\s+(?:of\s+)?(?:orders?|contracts?)|'
              r'(?:received|secured|won|awarded|signed|executed).{0,100}(?:work\s+order|purchase\s+order|orders?|contracts?|letter\s+of\s+award)|'
              r'(?:order|contract)\s+(?:win|wins|awarded|received|secured)')
    return 'BUSINESS_ORDER_DISCLOSURE_NOT_REVENUE' if re.search(positive,value,re.I) else 'OTHER_OR_UNRESOLVED'

def bound_number(value,excerpt,currency=False):
    """Exact magnitude with explicit INR/crore/lakh scale; never a free number.

    Currency matching is not proof of forward earnings or economic meaning;
    callers must enforce issuer, clock, basis and interpretation separately.
    """
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):return False
    if currency and not re.search(r'\bINR\b|\bRs\.?|₹|rupees',excerpt,re.I):return False
    for m in re.finditer(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?',excerpt):
        number=float(m.group().replace(',',''));suffix=excerpt[m.end():].lstrip()
        unit=re.match(r'(crores?|cr\b|lakhs?|lacs?|millions?|billions?|thousands?)',suffix,re.I)
        if unit:
            u=unit.group().lower();number*=1e7 if u.startswith(('cro','cr')) else 1e5 if u.startswith(('lak','lac')) else 1e6 if u.startswith('mil') else 1e9 if u.startswith('bil') else 1e3
        if np.isclose(value,number,rtol=1e-9,atol=1e-8):return True
    return False

def capacity_disclosure(text):
    """Disclosed capacity/production, not proof of physical commissioning.

    Commissioner titles and sales/brokerage commission are not capacity.
    Planned capacity may be a disclosure; only reviewed originals can certify
    commissioning, utilisation, commercial operation or attributable earnings.
    """
    value=str(text)
    if re.search(LEGAL,value,re.I):return False
    return bool(re.search(r'\bcapacity\b|\bcommissioned\b|\bcommissioning\b|\bcommercial\s+production\b|commencement\s+of\s+commercial\s+(?:production|operations)',value,re.I))
