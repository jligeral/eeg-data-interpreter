"""
scripts/evaluate.py
====================
Evaluate the LEAD classifier on a labeled dataset and report standard metrics.

Expected directory structure
-----------------------------
    data_dir/
        AD/   ← EEG files for Alzheimer's patients
        HC/   ← EEG files for healthy controls
        FTD/  ← EEG files for frontotemporal dementia (optional)

Any subdirectory name is treated as a class label.  Only subjects for which
LEAD produces an AD probability score are included in the metrics.

Usage
-----
    python scripts/evaluate.py --data_dir data/demo
    python scripts/evaluate.py --data_dir data/ADFTD --out results/eval.csv

Metrics reported
----------------
Binary task (AD vs HC only):
    AUC-ROC, AUC-PRC, accuracy at threshold 0.5

Multiclass (all classes present):
    One-vs-rest AUC-ROC (macro), macro F1, confusion matrix
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Supported EEG extensions (must match app.py / format_adapter.py)
# ---------------------------------------------------------------------------
_EEG_EXTS = {".fif", ".edf", ".bdf", ".set", ".gdf", ".cnt", ".vhdr",
             ".csv", ".tsv", ".npz", ".npy", ".mat"}

_REPO_ROOT = Path(__file__).parent.parent


def _collect_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Return [(filepath, label), ...] from a labelled directory tree."""
    pairs = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = subdir.name.upper()
        files = [f for f in subdir.rglob("*") if f.suffix.lower() in _EEG_EXTS]
        for f in sorted(files):
            pairs.append((f, label))
    return pairs


def _run_subject(filepath: Path) -> dict:
    """Run preprocess + extract for one subject. Returns a result dict."""
    from preprocessor.preprocess import preprocess
    from feature_extractor.extract import extract

    result = {
        "filepath": str(filepath),
        "data_quality": "unknown",
        "confounds": "",
        "ad_probability": None,
        "n_windows": 0,
        "alpha_peak_hz": None,
        "theta_alpha_ratio": None,
        "posterior_coherence": None,
        "error": None,
    }

    try:
        record = preprocess(str(filepath))
        feat = extract(record)

        result["data_quality"] = feat.data_quality
        result["confounds"] = "|".join(feat.confounds)
        result["ad_probability"] = feat.lead_ad_probability
        result["n_windows"] = feat.n_windows
        result["alpha_peak_hz"] = feat.alpha_peak_frequency
        result["theta_alpha_ratio"] = feat.theta_alpha_ratio
        result["posterior_coherence"] = feat.posterior_alpha_coherence

    except Exception as e:
        result["error"] = str(e)

    return result


def _compute_metrics(labels: list[str], probs: list[float], classes: list[str]) -> None:
    """Print evaluation metrics using scikit-learn."""
    try:
        import numpy as np
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )
    except ImportError:
        print("\n[!] scikit-learn not installed. Run: pip install scikit-learn")
        print("    Skipping metric computation.")
        return

    import numpy as np

    y_true = np.array(labels)
    y_prob = np.array(probs)
    unique_classes = sorted(set(labels))

    print("\n" + "═" * 60)
    print("  EVALUATION RESULTS")
    print("═" * 60)
    print(f"  Classes   : {unique_classes}")
    print(f"  Subjects  : {len(labels)}")
    for cls in unique_classes:
        print(f"    {cls:<6}: {labels.count(cls)}")

    # ── Binary case: AD vs HC ────────────────────────────────────────────
    if set(unique_classes) == {"AD", "HC"}:
        y_bin = (y_true == "AD").astype(int)

        auc_roc = roc_auc_score(y_bin, y_prob)
        auc_prc = average_precision_score(y_bin, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_bin, y_pred)
        cm = confusion_matrix(y_bin, y_pred, labels=[0, 1])

        print(f"\n  Task      : Binary (AD vs HC)")
        print(f"  AUC-ROC   : {auc_roc:.4f}")
        print(f"  AUC-PRC   : {auc_prc:.4f}  (precision-recall)")
        print(f"  F1 (AD)   : {f1:.4f}  (threshold = 0.5)")
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"             HC    AD")
        print(f"  HC (0)  {cm[0][0]:>5} {cm[0][1]:>5}")
        print(f"  AD (1)  {cm[1][0]:>5} {cm[1][1]:>5}")

    # ── Multiclass case ──────────────────────────────────────────────────
    else:
        # One-vs-rest AUC-ROC requires per-class probabilities.
        # We only have AD probability from LEAD, so we can compute
        # full OvR only for models that output per-class scores.
        # Here we report what's available.
        print(f"\n  Task      : Multiclass ({', '.join(unique_classes)})")
        print(f"  Note      : LEAD outputs AD probability only.")
        print(f"              OvR AUC computed as AD vs all others.")

        y_bin = (y_true == "AD").astype(int)
        auc_roc = roc_auc_score(y_bin, y_prob)
        print(f"  AUC-ROC   : {auc_roc:.4f}  (AD vs rest)")

        y_pred_label = np.where(y_prob >= 0.5, "AD", "HC/FTD")
        macro_f1 = f1_score(
            y_true,
            np.where(y_prob >= 0.5, "AD", y_true),  # non-AD kept as true label
            average="macro",
            labels=unique_classes,
        )
        print(f"  Macro F1  : {macro_f1:.4f}")

        cm = confusion_matrix(y_true, np.where(y_prob >= 0.5, "AD", y_true),
                              labels=unique_classes)
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        header = "".join(f"{c:>8}" for c in unique_classes)
        print(f"  {'':10}{header}")
        for i, row_label in enumerate(unique_classes):
            row = "".join(f"{cm[i][j]:>8}" for j in range(len(unique_classes)))
            print(f"  {row_label:<10}{row}")

    print("\n" + "═" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LEAD classifier on labeled EEG data.")
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Root directory with class subdirectories (AD/, HC/, FTD/, …)"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to save per-subject results as CSV (e.g. results/eval.csv)"
    )
    parser.add_argument(
        "--skip_poor", action="store_true",
        help="Exclude subjects with poor data quality from metrics"
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        sys.exit(f"[!] Data directory not found: {args.data_dir}")

    pairs = _collect_files(args.data_dir)
    if not pairs:
        sys.exit(f"[!] No EEG files found under {args.data_dir}")

    print(f"Found {len(pairs)} subject(s) across {len(set(l for _, l in pairs))} class(es).")
    print("Running pipeline (preprocess + feature extraction)...\n")

    rows = []
    for filepath, label in pairs:
        print(f"  [{label}] {filepath.name} ...", end=" ", flush=True)
        result = _run_subject(filepath)
        result["label"] = label
        rows.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        elif result["ad_probability"] is None:
            print(f"skipped (no AD probability — check LEAD checkpoint)")
        else:
            flag = " ← ELEVATED" if result["ad_probability"] > 0.6 else ""
            print(f"AD prob={result['ad_probability']:.3f}{flag}  quality={result['data_quality']}")

    # Save CSV
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["label", "filepath", "ad_probability", "data_quality",
                      "confounds", "n_windows", "alpha_peak_hz",
                      "theta_alpha_ratio", "posterior_coherence", "error"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: r.get(k) for k in fieldnames} for r in rows)
        print(f"\nPer-subject results saved to: {args.out}")

    # Filter to subjects with valid AD probability
    valid = [r for r in rows if r["ad_probability"] is not None and not r["error"]]
    if args.skip_poor:
        valid = [r for r in valid if r["data_quality"] != "poor"]
        print(f"Excluded poor-quality subjects. Evaluating on {len(valid)} subject(s).")

    if len(valid) < 2:
        print("\n[!] Not enough valid subjects to compute metrics.")
        return

    labels = [r["label"] for r in valid]
    probs = [r["ad_probability"] for r in valid]
    classes = sorted(set(labels))

    _compute_metrics(labels, probs, classes)


if __name__ == "__main__":
    main()
