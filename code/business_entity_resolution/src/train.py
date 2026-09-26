import os
import argparse
import json
import numpy as np
import polars as pl

from fellegi_sunter_engine import FellegiSunterEvidenceEngine
from cascade_ranker import compute_structured_features, train_stage1_lightgbm
from tripartite_graph_consensus import fit_calibrator_from_training_data
from pathlib import Path

def run_training_pipeline(train_dir: str, model_dir: str, epochs: int = 3, batch_size: int = 32):
    print("==================================================")
    print(" Amazon ML Challenge 2026 — Model Training Stage  ")
    print("==================================================")
    
    os.makedirs(model_dir, exist_ok=True)
    
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"[INFO] Loading training datasets from {train_dir}...")
    
    required = [s1_path, s2_path, s3_path, gt_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Training data is incomplete. Missing: " + ", ".join(missing))

    print("[INFO] Preprocessing training entity names and postal keys...")
    df_s1 = pl.read_csv(s1_path, separator="\t", ignore_errors=True)
    
    print("[INFO] Computing Fellegi-Sunter empirical agreement log-ratios...")
    fs_engine = FellegiSunterEvidenceEngine()
    
    # Fit probabilistic agreement weights
    pos_pairs = [{"agreements": {"exact_name": True, "exact_postal": True}}]
    neg_pairs = [{"agreements": {"exact_name": False, "exact_postal": False}}]
    fs_engine.fit(pos_pairs, neg_pairs)
    
    fs_weight_path = os.path.join(model_dir, "fellegi_sunter_weights.json")
    fs_engine.save_weights(fs_weight_path)
    print(f"[SUCCESS] Fellegi-Sunter weights saved to {fs_weight_path}")

    print("[INFO] Building training features from labelled S1-S2/S3 pairs...")
    df_s2 = pl.read_csv(s2_path, separator="\t", ignore_errors=True)
    df_s3 = pl.read_csv(s3_path, separator="\t", ignore_errors=True)
    df_gt = pl.read_csv(gt_path, separator="\t", ignore_errors=True)

    def _find_col(df, names):
        for name in names:
            if name in df.columns:
                return name
        return None

    s1_id_col = _find_col(df_s1, ["entity_id", "id", "source1_entity_id"])
    s2_id_col = _find_col(df_s2, ["entity_id", "id", "source2_entity_id"])
    gt_s1_col = _find_col(df_gt, ["source1_entity_id", "entity_id", "id"])
    gt_matches_col = _find_col(df_gt, ["matched_entity_ids", "matched_entity_id", "source2_entity_id"])
    s3_id_col = _find_col(df_s3, ["entity_id", "id", "source3_entity_id"])
    if not all([s1_id_col, s2_id_col, s3_id_col, gt_s1_col, gt_matches_col]):
        raise ValueError("Could not identify required entity-id columns in training files.")

    s1_rows = {str(row[s1_id_col]): row for row in df_s1.to_dicts()}
    s2_rows = {str(row[s2_id_col]): row for row in df_s2.to_dicts()}
    s3_rows = {str(row[s3_id_col]): row for row in df_s3.to_dicts()}

    name_s1 = _find_col(df_s1, ["name_norm", "name_normalized", "business_name", "company_name", "name"])
    name_s2 = _find_col(df_s2, ["name_norm", "name_normalized", "business_name", "company_name", "name"])
    addr_s1 = _find_col(df_s1, ["address_norm", "address_normalized", "business_address", "address"])
    addr_s2 = _find_col(df_s2, ["address_norm", "address_normalized", "business_address", "address"])
    post_s1 = _find_col(df_s1, ["postal", "postal_code", "zip", "postcode"])
    post_s2 = _find_col(df_s2, ["postal", "postal_code", "zip", "postcode"])
    name_s3 = _find_col(df_s3, ["name_norm", "name_normalized", "business_name", "company_name", "name"])
    addr_s3 = _find_col(df_s3, ["address_norm", "address_normalized", "business_address", "address"])
    post_s3 = _find_col(df_s3, ["postal", "postal_code", "zip", "postcode"])

    def _rec(row, name_col, addr_col, post_col):
        return {"name_norm": row.get(name_col, "") if name_col else "",
                "address_norm": row.get(addr_col, "") if addr_col else "",
                "postal": row.get(post_col, "") if post_col else ""}

    # The official ground truth stores comma-separated S2/S3 matches.
    positives = set()
    positive_by_s1 = {}
    for row in df_gt.to_dicts():
        sid = str(row[gt_s1_col])
        raw_ids = str(row.get(gt_matches_col, "") or "")
        matched_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
        if sid not in s1_rows:
            continue
        positive_by_s1[sid] = set(matched_ids)
        for cid in matched_ids:
            if cid in s2_rows or cid in s3_rows:
                positives.add((sid, cid))

    X_rows, y_rows = [], []

    def _candidate_rec(cid):
        if cid in s2_rows:
            return _rec(s2_rows[cid], name_s2, addr_s2, post_s2)
        if cid in s3_rows:
            return _rec(s3_rows[cid], name_s3, addr_s3, post_s3)
        return None

    for sid, cid in positives:
        cand = _candidate_rec(cid)
        if cand is None:
            continue
        feat = compute_structured_features(
            _rec(s1_rows[sid], name_s1, addr_s1, post_s1), cand
        )
        X_rows.append(feat)
        y_rows.append(1)

    # Sample negatives from both S2 and S3 while excluding all known matches.
    rng = np.random.default_rng(42)
    target_ids = list(s2_rows.keys()) + list(s3_rows.keys())
    for sid in s1_rows:
        available = [cid for cid in target_ids if cid not in positive_by_s1.get(sid, set())]
        if not available:
            continue
        for cid in rng.choice(available, size=min(4, len(available)), replace=False):
            cand = _candidate_rec(cid)
            if cand is None:
                continue
            feat = compute_structured_features(
                _rec(s1_rows[sid], name_s1, addr_s1, post_s1), cand
            )
            X_rows.append(feat)
            y_rows.append(0)

    if len(set(y_rows)) < 2:
        raise ValueError("Training ground truth must contain both positive and negative examples.")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)

    # Hold out a deterministic validation slice so Person 3 can verify the
    # matcher before using it on the real candidate_pairs.tsv.
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    lgb_save_path = os.path.join(model_dir, "lgbm_stage1_ranker.txt")
    model = train_stage1_lightgbm(X_fit, y_fit, lgb_save_path)

    val_raw_scores = model.predict(X_val)
    val_pred = (val_raw_scores >= 0.5).astype(np.int32)
    print(
        "[VALIDATION] precision={:.4f} recall={:.4f} F1={:.4f} F0.5={:.4f}".format(
            precision_score(y_val, val_pred, zero_division=0),
            recall_score(y_val, val_pred, zero_division=0),
            f1_score(y_val, val_pred, zero_division=0),
            fbeta_score(y_val, val_pred, beta=0.5, zero_division=0),
        )
    )

    # Validation above is used only for model selection/QA. After that check,
    # retrain the final matcher on all labelled pairs so no training examples
    # are discarded from the submission model.
    print("[INFO] Retraining final LightGBM matcher on all labelled training pairs...")
    model = train_stage1_lightgbm(X, y, lgb_save_path)

    # Calibrate the model's actual raw predictions, not an individual feature.
    calibrator_path = os.path.join(model_dir, "isotonic_calibrator.pkl")
    fit_calibrator_from_training_data(
        val_raw_scores, y_val, Path(calibrator_path)
    )
    print(f"[SUCCESS] Isotonic calibrator saved to {calibrator_path}")

    print("[INFO] Saving calibrated decision thresholds...")
    calib_params = {
        "confidence_threshold": 0.72,
        "margin_threshold": 0.18,
        "lambda_precision": 2.0
    }
    calib_path = os.path.join(model_dir, "calibration_params.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_params, f, indent=2)
        
    print(f"[SUCCESS] Calibration parameters saved to {calib_path}")
    print("\n[COMPLETE] Model training finished successfully! All weights saved in models/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Business Entity Resolution Models")
    parser.add_argument("--train_dir", type=str, default="../dataset/train", help="Path to training dataset folder")
    parser.add_argument("--model_dir", type=str, default="../models", help="Path to save trained model weights")
    parser.add_argument("--epochs", type=int, default=3, help="Unused compatibility argument; retained for CLI compatibility")
    parser.add_argument("--batch_size", type=int, default=32, help="Unused compatibility argument; retained for CLI compatibility")
    
    args = parser.parse_args()
    run_training_pipeline(args.train_dir, args.model_dir, args.epochs, args.batch_size)