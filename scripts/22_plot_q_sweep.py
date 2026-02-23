#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    summ = sweep / "summary"
    metrics = pd.read_csv(summ / "metrics_by_q_split.csv")
    dist = pd.read_csv(summ / "distances_long.csv")

    outplots = summ / "plots"
    outplots.mkdir(parents=True, exist_ok=True)

    # --------- Curvas suaves (sin inventar ajuste aún: línea + puntos) ----------
    m = metrics[metrics["split"] == args.split].sort_values("q")

    for col, title, fname in [
        ("global_acc", "Global Accuracy vs q", f"curve_global_acc_{args.split}.png"),
        ("macro_acc", "Macro Accuracy vs q", f"curve_macro_acc_{args.split}.png"),
        ("macro_f1", "Macro-F1 vs q", f"curve_macro_f1_{args.split}.png"),
        ("no_detect_rate", "NO_DETECT Rate vs q", f"curve_no_detect_{args.split}.png"),
    ]:
        plt.figure()
        plt.plot(m["q"].values, (m[col].values * 100.0), marker="o")
        plt.xlabel("q")
        plt.ylabel(f"{col} (%)")
        plt.title(title + f" ({args.split})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outplots / fname, dpi=200)
        plt.close()

    # --------- Distancias por especie: boxplot (por q) ----------
    d = dist[dist["split"] == args.split].copy()
    # opción: mirar SOLO casos detectados (para geometría de asignación),
    # y también mirar todos (incluye NO_DETECT).
    for only_detected, tag in [(True, "detected_only"), (False, "all")]:
        dd = d[d["detected"] == True].copy() if only_detected else d.copy()

        species_list = sorted(dd["true_species"].unique().tolist())
        qs = sorted(dd["q"].unique().tolist())

        # Para cada especie, hacemos una figura: boxplot de best_distance vs q
        for sp in species_list:
            sps = dd[dd["true_species"] == sp]
            data = []
            labels = []
            for q in qs:
                x = sps[sps["q"] == q]["best_distance"].dropna().values
                data.append(x)
                labels.append(f"{q:.3f}")

            plt.figure(figsize=(10,4))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.xlabel("q")
            plt.ylabel("best_distance")
            plt.title(f"best_distance distribution vs q | {sp} | {args.split} | {tag}")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(outplots / f"box_best_distance_{sp}_{args.split}_{tag}.png", dpi=200)
            plt.close()

        # ECDF comparativa por especie (usamos el q más alto y más bajo para contraste)
        q_lo, q_hi = min(qs), max(qs)
        plt.figure(figsize=(8,5))
        for sp in species_list:
            x_lo = dd[(dd["true_species"] == sp) & (dd["q"] == q_lo)]["best_distance"].dropna().values
            x_hi = dd[(dd["true_species"] == sp) & (dd["q"] == q_hi)]["best_distance"].dropna().values
            if len(x_lo) > 0:
                xx, yy = ecdf(x_lo)
                plt.plot(xx, yy, label=f"{sp} q={q_lo:.3f}")
            if len(x_hi) > 0:
                xx, yy = ecdf(x_hi)
                plt.plot(xx, yy, linestyle="--", label=f"{sp} q={q_hi:.3f}")
        plt.xlabel("best_distance")
        plt.ylabel("ECDF")
        plt.title(f"ECDF best_distance (q low vs high) | {args.split} | {tag}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outplots / f"ecdf_best_distance_{args.split}_{tag}.png", dpi=200)
        plt.close()

    print("✅ Plots in:", outplots)

if __name__ == "__main__":
    main()