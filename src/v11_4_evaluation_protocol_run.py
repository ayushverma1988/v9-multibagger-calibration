"""Freeze actual existing research weights and a prospective evaluation plan.

No retrospective selections are registered. The public receipt pins the
private protocol; source-qualified full-model evaluation remains pending.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib

from v11_4_independent_evaluation_ledger import freeze_protocol
from v11_4_robust_research_model import FEATURES, VERSION


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(model, source_features, output, public_key):
    package = joblib.load(model)
    if package.get("features") != list(FEATURES) or package.get("model_version") != VERSION:
        raise ValueError("Actual package does not match the previously tested fixed research variant")
    root = Path(output)
    private = root / "ledger_PRIVATE"
    private.mkdir(parents=True, exist_ok=True)
    source_paths = ["v11_4_independent_evaluation_ledger.py", "v11_4_robust_research_model.py",
                    "v11_4_verified_current_financials.py", "v11_4_primary_document_catalysts.py",
                    "v11_4_exchange_reference_universe.py", "v11_4_source_blocker_recovery.py",
                    "v11_4_verified_promoter_transactions.py",
                    "v11_4_enriched_current_source_audit.py",
                    "v11_4_forward_research_release.py", "v11_4_standalone_train_walkforward.py"]
    recipe = {"model_version": VERSION, "feature_names": list(FEATURES), "C": .03,
              "market_training_tail_quantiles": [.005, .995],
              "source_code_sha256": {p: file_hash(Path(__file__).parent / p) for p in source_paths},
              "source_reference_snapshot_sha256": file_hash(source_features),
              "source_reference_snapshot_is_after_seen_Oct9_decision": True,
              "financial_catalyst_enrichment_is_audited_overlay_not_trained_inputs": True,
              "no_scores_from_other_versions": True,
              "future_same_recipe_retraining_prohibited_in_this_ledger": True}
    stamp = datetime.now(timezone.utc).isoformat()
    protocol, digest = freeze_protocol(private, model, recipe, stamp)
    shutil.copyfile(model, private / "locked_model.joblib")
    # Copy source only; no outcomes and no private old-version rankings.
    shutil.copyfile(source_features, private / "reference_source_features_PRIVATE.parquet")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        commit = "UNKNOWN"
    summary = {"scope": "INDEPENDENT_EVALUATION_INFRASTRUCTURE_FROZEN_NO_OUTCOMES_YET", "frozen_at_utc": stamp,
               "source_code_commit": commit, "protocol_sha256": digest, "frozen_model_sha256": file_hash(model),
               "model_version": VERSION, "horizon_actual_market_sessions": 126,
               "new_prospective_observations_registered": 0, "new_mature_blind_outcomes": 0,
               "already_seen_Oct9_selection_not_backfilled": True,
               "acceptance_gates": protocol["acceptance_gates"],
               "full_financial_catalyst_model_source_qualified": False,
               "full_NSE_BSE_scoring_coverage": False, "production_approved": False}
    (root / "independent_evaluation_status_summary.json").write_text(json.dumps(summary, indent=2))
    # Authenticated encryption permits an owner-only exact replay download.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(private.rglob("*")):
            if p.is_file():
                archive.write(p, str(p.relative_to(private)))
    key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    aad = digest.encode()
    pub = serialization.load_pem_public_key(Path(public_key).read_bytes())
    wrap = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    packet = {k: base64.b64encode(v).decode() for k, v in {
        "key": wrap, "nonce": nonce, "aad": aad, "payload": AESGCM(key).encrypt(nonce, buffer.getvalue(), aad)}.items()}
    (root / "owner_evaluation_bundle_ENCRYPTED.json").write_text(json.dumps(packet))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source-features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--owner-public-key", required=True)
    a = p.parse_args()
    run(a.model, a.source_features, a.output, a.owner_public_key)


if __name__ == "__main__":
    main()
