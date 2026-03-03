#!/usr/bin/env python
"""
normalized_chunks_common.py

- Load 3 combined_analysis.csv files (n_bytes, n_flows, n_packets)
- Merge them on id_time
- Normalize all three signals to [0, 1]
- Compute common anomalies where ANY 2/3 signals say anomaly
  * for detector flags (is_detector_anomaly == 1)
  * for pattern flags  (is_pattern_anomaly == 1)
- Plot ONE normalized signal view split into 4 chunks,
  showing common anomalies.
- NOW ALSO COUNTS anomalies and prints percentages.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
CSV_PATHS = [
    r"results\detec_res_103_1hour\n_bytes\combined_analysis.csv",
    r"results\detec_res_103_1hour\n_flows\combined_analysis.csv",
    r"results\detec_res_103_1hour\n_packets\combined_analysis.csv",
]

LABELS = ["n_bytes", "n_flows", "n_packets"]

OUTPUT_FIG = Path("results") / "normalized_common_4chunks.png"


# ---------------------------------------------------------------
# Helper: load required columns and rename per signal
# ---------------------------------------------------------------
def load_subset(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    sub = df[[
        "id_time",
        "orig_value",
        "is_detector_anomaly",
        "is_pattern_anomaly",
    ]].copy()

    sub = sub.rename(columns={
        "orig_value": f"orig_value_{label}",
        "is_detector_anomaly": f"is_detector_anomaly_{label}",
        "is_pattern_anomaly": f"is_pattern_anomaly_{label}",
    })
    return sub


def main():
    # -----------------------------------------------------------
    # 1. Merge 3 CSVs on id_time
    # -----------------------------------------------------------
    merged = None
    for path, label in zip(CSV_PATHS, LABELS):
        sub = load_subset(path, label)
        merged = sub if merged is None else pd.merge(merged, sub, on="id_time", how="inner")

    if merged is None or merged.empty:
        print("Error: merged dataset is empty.")
        return

    merged = merged.sort_values("id_time").reset_index(drop=True)
    total_points = len(merged)

    # -----------------------------------------------------------
    # 2. Normalize all three signals to [0,1]
    # -----------------------------------------------------------
    for label in LABELS:
        col = f"orig_value_{label}"
        y = merged[col].astype(float)
        if y.max() != y.min():
            merged[f"norm_{label}"] = (y - y.min()) / (y.max() - y.min())
        else:
            merged[f"norm_{label}"] = 0.0

    # -----------------------------------------------------------
    # 3. Common anomalies (ANY 2 of 3)
    # -----------------------------------------------------------
    det_cols = [f"is_detector_anomaly_{lbl}" for lbl in LABELS]
    pat_cols = [f"is_pattern_anomaly_{lbl}" for lbl in LABELS]

    merged[det_cols] = merged[det_cols].astype(int)
    merged[pat_cols] = merged[pat_cols].astype(int)

    merged["common_detector_2of3"] = merged[det_cols].sum(axis=1) >= 2
    merged["common_pattern_2of3"] = merged[pat_cols].sum(axis=1) >= 2

    merged["norm_mean"] = merged[[f"norm_{lbl}" for lbl in LABELS]].mean(axis=1)

    # -----------------------------------------------------------
    # 4. Count & percentage statistics
    # -----------------------------------------------------------
    print("\n=== ANOMALY SUMMARY ===")

    for label in LABELS:
        det_mask = merged[f"is_detector_anomaly_{label}"] == 1
        pat_mask = merged[f"is_pattern_anomaly_{label}"] == 1

        det_count = det_mask.sum()
        pat_count = pat_mask.sum()

        det_pct = 100 * det_count / total_points
        pat_pct = 100 * pat_count / total_points

        print(
            f"{label}: "
            f"detector {det_count} ({det_pct:.2f}%), "
            f"pattern {pat_count} ({pat_pct:.2f}%)"
        )

    common_det_count = merged["common_detector_2of3"].sum()
    common_pat_count = merged["common_pattern_2of3"].sum()

    print(
        f"\nCommon detector anomalies (>=2/3): "
        f"{common_det_count} ({100 * common_det_count / total_points:.2f}%)"
    )

    print(
        f"Common pattern anomalies (>=2/3): "
        f"{common_pat_count} ({100 * common_pat_count / total_points:.2f}%)\n"
    )

    # -----------------------------------------------------------
    # 5. Plot – with bottom legend
    # -----------------------------------------------------------
    n = len(merged)
    chunk_size = math.ceil(n / 4)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

    for i in range(4):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        if start >= end:
            axes[i].axis("off")
            continue

        chunk = merged.iloc[start:end]
        ax = axes[i]
        x = chunk["id_time"]

        # Plot normalized signals
        for label in LABELS:
            y = chunk[f"norm_{label}"]
            # Only show legend text labels once; underscore hides duplicates
            line_label = label if i == 0 else f"_{label}"
            ax.plot(x, y, label=line_label, linewidth=1)

        # Common anomaly markers
        det_mask = chunk["common_detector_2of3"]
        pat_mask = chunk["common_pattern_2of3"]
        y_mean = chunk["norm_mean"]

        ax.scatter(
            x[det_mask],
            y_mean[det_mask],
            marker="X",
            c="black",
            s=45,
            label="common detector anomaly (>=2/3)" if i == 0 else "_det",
        )

        ax.scatter(
            x[pat_mask],
            y_mean[pat_mask],
            marker="o",
            c="red",
            s=45,
            label="common pattern anomaly (>=2/3)" if i == 0 else "_pat",
        )

        ax.set_ylabel(f"Chunk {i + 1}")
        ax.set_xlabel("id_time")  # X axis label on every subplot
        ax.grid(True, linestyle="--", alpha=0.3)

        # Ensure x tick labels are visible on all subplots
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha="right")

    # -------------------------------------------------------
    # Bottom legend (centered) with fontsize ~20px
    # -------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=20,
        ncol=len(labels),       # all entries in one row
        bbox_to_anchor=(0.5, 0.01),  # slightly below the axes area
    )

    # Leave space at bottom for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
