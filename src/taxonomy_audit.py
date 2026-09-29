from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import xbrl_normalizer as xn


NUMERIC_TAGS = {"nonfraction", "fraction"}


def _iter_tar_members(path: Path):
    try:
        with tarfile.open(path, "r:*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                yield m.name, f.read()
    except Exception:
        return


def _iter_zip_members(blob: bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for n in zf.namelist():
                low = n.lower()
                if low.endswith((".xml", ".xbrl", ".xhtml", ".html", ".htm")):
                    yield n, zf.read(n)
    except Exception:
        return


def iter_payloads(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith((".tar.gz", ".tgz", ".tar")):
            for name, blob in _iter_tar_members(p):
                if blob[:2] == b"PK":
                    for zn, zb in _iter_zip_members(blob):
                        yield f"{p}:{name}:{zn}", zb
                else:
                    yield f"{p}:{name}", blob
        elif low.endswith(".zip"):
            blob = p.read_bytes()
            for name, z in _iter_zip_members(blob):
                yield f"{p}:{name}", z
        elif low.endswith((".xml", ".xbrl", ".xhtml", ".html", ".htm", ".bin")):
            blob = p.read_bytes()
            if blob[:2] == b"PK":
                for name, z in _iter_zip_members(blob):
                    yield f"{p}:{name}", z
            else:
                yield str(p), blob


def _year_from_source(s: str):
    m = re.search(r"(20(?:1[7-9]|2[0-6]))", s)
    return int(m.group(1)) if m else None


def _is_html(blob: bytes):
    head = blob[:500].lstrip().lower()
    return head.startswith(b"<html") or b"<table" in head or b"<!doctype html" in head


def audit_xbrl(blob: bytes, source: str, mapping: dict, concept_rows, family_counts, file_stats):
    try:
        root = ET.fromstring(blob)
    except Exception:
        return False

    compiled = xn.compile_map(mapping)
    contexts = xn._parse_contexts(root)
    namespaces = set()
    all_concepts = []
    facts = []

    for el in root.iter():
        cref = el.attrib.get("contextRef") or el.attrib.get("contextref")
        if not cref or cref not in contexts:
            continue
        concept = xn._fact_concept(el)
        val = xn.parse_number("".join(el.itertext()).strip(), el.attrib)
        if not pd.notna(val):
            continue
        ns = xn.namespace_uri(el.tag)
        if ns:
            namespaces.add(ns)
        all_concepts.append(concept)
        canonical, conf = xn.map_concept(concept, compiled)
        facts.append((concept, canonical, conf, cref, ns))

    family = xn.classify_taxonomy(namespaces, all_concepts)
    year = _year_from_source(source)
    family_counts[(year, family)] += 1

    mapped = 0
    unmapped = 0
    for concept, canonical, conf, cref, ns in facts:
        key = (
            year,
            family,
            concept,
            canonical or "",
            float(conf),
            ns,
        )
        concept_rows[key] += 1
        if canonical:
            mapped += 1
        else:
            unmapped += 1

    file_stats.append({
        "source": source,
        "year": year,
        "kind": "xbrl",
        "taxonomy_family": family,
        "numeric_facts": len(facts),
        "mapped_facts": mapped,
        "unmapped_facts": unmapped,
        "mapped_fraction": mapped / len(facts) if facts else 0.0,
    })
    return True


def audit_html(blob: bytes, source: str, label_rows, file_stats):
    try:
        tables = pd.read_html(io.BytesIO(blob))
    except Exception:
        return False
    year = _year_from_source(source)
    labels = []
    for t in tables:
        if t.shape[1] < 2:
            continue
        for _, r in t.iloc[:, :2].iterrows():
            label = str(r.iloc[0]).strip()
            val = str(r.iloc[1]).strip()
            if not label or label.lower() in {"nan", "description"}:
                continue
            # Keep rows that look like financial line items.
            if re.search(r"\d", val) or val in {"-", "—"}:
                labels.append(label)
                label_rows[(year, label)] += 1

    file_stats.append({
        "source": source,
        "year": year,
        "kind": "legacy_html",
        "taxonomy_family": "legacy_html",
        "numeric_facts": len(labels),
        "mapped_facts": 0,
        "unmapped_facts": len(labels),
        "mapped_fraction": 0.0,
    })
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mapping", default="config/xbrl_concept_map_v9_5.json")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = xn.load_map(args.mapping)

    concept_rows = Counter()
    label_rows = Counter()
    family_counts = Counter()
    file_stats = []
    scanned = 0
    xbrl_files = 0
    html_files = 0

    for source, blob in iter_payloads(root):
        scanned += 1
        if _is_html(blob):
            if audit_html(blob, source, label_rows, file_stats):
                html_files += 1
            continue
        if audit_xbrl(blob, source, mapping, concept_rows, family_counts, file_stats):
            xbrl_files += 1
        else:
            # Some iXBRL documents may look like HTML but are XML parseable only
            # after decoding; try the HTML path as a last resort.
            if audit_html(blob, source, label_rows, file_stats):
                html_files += 1

        if scanned % 500 == 0:
            print(f"scanned={scanned} xbrl={xbrl_files} html={html_files}", flush=True)

    crows = []
    for (year, family, concept, canonical, conf, ns), count in concept_rows.items():
        crows.append({
            "year": year,
            "taxonomy_family": family,
            "concept": concept,
            "normalized_concept": xn.norm_name(concept),
            "canonical": canonical,
            "confidence": conf,
            "namespace": ns,
            "count": count,
            "mapped": bool(canonical),
        })
    cdf = pd.DataFrame(crows)
    if len(cdf):
        cdf = cdf.sort_values(["mapped", "count"], ascending=[True, False])
    cdf.to_csv(out / "concept_inventory.csv", index=False)

    ldf = pd.DataFrame([
        {"year": y, "label": label, "count": count}
        for (y, label), count in label_rows.items()
    ])
    if len(ldf):
        ldf = ldf.sort_values("count", ascending=False)
    ldf.to_csv(out / "legacy_label_inventory.csv", index=False)

    fdf = pd.DataFrame(file_stats)
    fdf.to_csv(out / "file_audit.csv", index=False)

    fam = pd.DataFrame([
        {"year": y, "taxonomy_family": family, "files": count}
        for (y, family), count in family_counts.items()
    ])
    fam.to_csv(out / "taxonomy_family_counts.csv", index=False)

    top_unmapped = (
        cdf[cdf["mapped"] == False].head(1000).copy()
        if len(cdf) else pd.DataFrame()
    )
    top_unmapped.to_csv(out / "top_unmapped_concepts.csv", index=False)

    summary = {
        "payloads_scanned": scanned,
        "xbrl_files": xbrl_files,
        "legacy_html_files": html_files,
        "distinct_concepts": int(cdf["concept"].nunique()) if len(cdf) else 0,
        "mapped_distinct_concepts": int(cdf.loc[cdf["mapped"], "concept"].nunique()) if len(cdf) else 0,
        "unmapped_distinct_concepts": int(cdf.loc[~cdf["mapped"], "concept"].nunique()) if len(cdf) else 0,
        "numeric_facts": int(fdf["numeric_facts"].sum()) if len(fdf) else 0,
        "mapped_numeric_facts": int(fdf["mapped_facts"].sum()) if len(fdf) else 0,
        "mapped_numeric_fraction": (
            float(fdf["mapped_facts"].sum() / fdf["numeric_facts"].sum())
            if len(fdf) and fdf["numeric_facts"].sum() else 0.0
        ),
        "years": sorted(int(x) for x in fdf["year"].dropna().unique()) if len(fdf) else [],
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
