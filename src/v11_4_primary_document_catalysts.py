"""Read official issuer attachments; distinguish facts from a causal chain.

Quantified announcements become traceable evidence candidates. They cannot
establish earnings impact, price discovery, or promoter buying by themselves.
"""
from __future__ import annotations
import io
import re

CATEGORIES = {
    "order": r"order\s+(?:wins?|receipt|received|award)|(?:bagged|bagging|receiving|secured|received|awarded).{0,90}(?:orders?|contracts?)|letter\s+of\s+award",
    "capacity": r"capacity\s+(?:expansion|addition)|commission(?:ed|ing)|commercial\s+production|new\s+(?:plant|manufacturing\s+facility)",
    "product": r"(?:launch|introduc|develop).{0,65}(?:new\s+)?product|new\s+product",
    "approval": r"(?:regulatory|USFDA|FDA|CE).{0,50}approval|approval.{0,50}(?:USFDA|FDA)",
    "promoter_disclosure": r"(?:promoter|insider|SAST|acquisition).{0,90}(?:disclosure|shares|purchase|sale)",
}
MONEY = re.compile(r"(?:order\s+value|value\s+of\s+(?:the\s+)?order|aggregate\s+(?:order\s+)?value|orders?\s+worth|contract\s+value|orders?.{0,30}worth)"
                   r"[^.;\n]{0,90}?(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)\s*(crores?|cr\.?|lakhs?|lacs?|million|billion)", re.I)
CAPACITY = re.compile(r"(?:capacity|production\s+capacity)[^.;\n]{0,90}?([\d,]+(?:\.\d+)?)\s*(MWh|MW|GW|MTPA|TPA|MSM|million\s*(?:fibre\s*)?km|tonnes?\s*(?:per\s+annum|p\.?a\.?)|units\s*per\s*(?:year|annum))", re.I)


def classify_document(text, issuer_symbol, issuer_name):
    flat = re.sub(r"\s+", " ", text)
    norm = re.sub(r"[^a-z0-9]", "", flat.lower())
    name = re.sub(r"[^a-z0-9]", "", issuer_name.lower())
    identity = (bool(re.search(r"(?<![A-Z0-9])" + re.escape(issuer_symbol) + r"(?![A-Z0-9])", flat, re.I))
                or (len(name) >= 10 and name in norm))
    if not identity:
        return {"issuer_identity_read": False, "status": "ISSUER_IDENTITY_UNRESOLVED",
                "categories": [], "quantified_evidence": [], "complete_causal_chain_verified": False}
    categories = [key for key, pattern in CATEGORIES.items() if re.search(pattern, flat, re.I)]
    evidence = []
    for category, pattern in (("order", MONEY), ("capacity", CAPACITY)):
        for m in pattern.finditer(flat):
            if category not in categories:
                continue
            value, unit = float(m.group(1).replace(",", "")), m.group(2).lower()
            excerpt = flat[max(0, m.start() - 100):m.end() + 160]
            uncertain = bool(re.search(r"propos|plan(?:ned)?|expect|subject\s+to|potential|may\s+be|cancell", excerpt, re.I))
            evidence.append({"category": category, "quantity": value, "unit_as_disclosed": unit,
                             "source_excerpt": excerpt,
                             "status": "CONDITIONAL_CANDIDATE" if uncertain else "DISCLOSED_QUANTITY_CANDIDATE",
                             "earnings_impact_verified": False})
    return {"issuer_identity_read": True, "status": "PRIMARY_DOCUMENT_READ",
            "categories": categories, "quantified_evidence": evidence,
            "promoter_purchase_direction": "UNKNOWN",
            "complete_causal_chain_verified": False}


def extract_pdf(raw):
    from pypdf import PdfReader
    if not raw.startswith(b"%PDF"):
        raise ValueError("Official attachment is not a PDF")
    doc = PdfReader(io.BytesIO(raw))
    if len(doc.pages) > 80:
        raise ValueError("Attachment exceeds bounded event-document extraction")
    text = "\n".join(p.extract_text() or "" for p in doc.pages)
    if len(text.strip()) < 100:
        raise ValueError("PDF requires OCR; do not manufacture event evidence")
    return text
