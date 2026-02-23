#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def macro_accuracy(df: pd.DataFrame) -> float:
    # accuracy por clase (true_species) y luego promedio (macro)
    acc_by = df.groupby("true_species").apply(lambda g: (g["pred_species"] == g["true_species"]).mean())
    return float(acc_by.mean())

def balanced_accuracy(df: pd.DataFrame) -> float:
    # en multiclase, balanced acc = promedio de recalls por clase
    rec_by = df.groupby("true_species").apply(lambda g: (g["pred_species"] == g["true_species"]).mean())
    return float(rec_by.mean())

def macro_f1_excluding_no_detect(df: pd.DataFrame) -> float:
    """
    Macro-F1 multiclase considerando clases reales (true_species),
    tratando NO_DETECT como una etiqueta más en pred, pero NO como clase objetivo.
    Implementación simple:
      - para cada clase c: precision/recall con pred==c vs true==c
      - f1_c = 2PR/(P+R)
    """
    classes = sorted(df["true_species"].unique().tolist())
    y_true = df["true_species"].astype(str).values
    y_pred = df["pred_species"].astype(str).values

    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="outputs/q_sweep_YYYYMMDD_HHMMSS")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    if not sweep.exists():
        raise FileNotFoundError(sweep)

    rows_metrics = []
    rows_dist = []

    qdirs = sorted([p for p in sweep.glob("q_*") if p.is_dir()])
    for qdir in qdirs:
        q = float(qdir.name.split("_", 1)[1])
        for split in ["train", "val", "test"]:
            rcsv = qdir / split / "results.csv"
            if not rcsv.exists():
                continue
            df = pd.read_csv(rcsv)

            # métricas “básicas” (las que tú quieres primero)
            N = len(df)
            global_acc = float((df["pred_species"] == df["true_species"]).mean())
            no_detect = float((df["pred_species"] == "NO_DETECT").mean())
            mac_acc = macro_accuracy(df)
            bal_acc = balanced_accuracy(df)  # igual a macro recall en este setting
            mac_f1 = macro_f1_excluding_no_detect(df)

            rows_metrics.append({
                "q": q,
                "split": split,
                "N": N,
                "global_acc": global_acc,
                "macro_acc": mac_acc,
                "balanced_acc": bal_acc,
                "macro_f1": mac_f1,
                "no_detect_rate": no_detect,
                "results_csv": str(rcsv),
            })

            # distancias (long format) para analizar distribución por clase
            # guardamos best_distance, detect/correct, etc.
            keep = df[["true_species", "pred_species", "detected", "correct", "best_distance"]].copy()
            keep["q"] = q
            keep["split"] = split
            rows_dist.append(keep)

    outdir = sweep / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(rows_metrics).sort_values(["split", "q"]).reset_index(drop=True)
    metrics_path = outdir / "metrics_by_q_split.csv"
    metrics.to_csv(metrics_path, index=False)

    dist = pd.concat(rows_dist, ignore_index=True) if rows_dist else pd.DataFrame()
    dist_path = outdir / "distances_long.csv"
    dist.to_csv(dist_path, index=False)

    print("✅ Wrote:", metrics_path)
    print("✅ Wrote:", dist_path)

if __name__ == "__main__":
    main()