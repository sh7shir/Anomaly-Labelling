#!/usr/bin/env python
# and_.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# 1. CONFIG: point these to the TWO combined_analysis.csv files
# -------------------------------------------------------------------
CSV_PATHS = [
    r"results\detec_res_103_1hour\n_flows\combined_analysis.csv",
    r"results\detec_res_103_1hour\n_packets\combined_analysis.csv",
]

LABELS = ["n_flows", "n_packets"]  # only plot flows and packets

OUTPUT_FIG = Path("results") / "anomalies_flows_packets.png"


# -------------------------------------------------------------------
# Helper: load only the needed columns and rename per-file
# -------------------------------------------------------------------
def load_subset(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    sub = df[["id_time", "orig_value",
              "is_detector_anomaly", "is_pattern_anomaly"]].copy()

    sub = sub.rename(columns={
        "orig_value": f"orig_value_{label}",
        "is_detector_anomaly": f"is_detector_anomaly_{label}",
        "is_pattern_anomaly": f"is_pattern_anomaly_{label}",
    })
    return sub


def main():
    # ----------------------------------------------------------------
    # 1. Read the two CSVs and merge them on id_time
    # ----------------------------------------------------------------
    merged = None
    for path, label in zip(CSV_PATHS, LABELS):
        sub = load_subset(path, label)
        if merged is None:
            merged = sub
        else:
            merged = pd.merge(merged, sub, on="id_time", how="inner")

    if merged is None or merged.empty:
        print("No data after merging – check file paths and contents.")
        return

    # ----------------------------------------------------------------
    # 2. Plot: 2 subplots with anomaly counts in labels (legend outside)
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x = merged["id_time"]

    for i, label in enumerate(LABELS):
        ax = axes[i]
        y = merged[f"orig_value_{label}"]

        ax.plot(x, y, linewidth=1.0, label=f"{label} – orig_value")

        det_mask = merged[f"is_detector_anomaly_{label}"] == 1
        pat_mask = merged[f"is_pattern_anomaly_{label}"] == 1

        det_count = int(det_mask.sum())
        pat_count = int(pat_mask.sum())

        # detector anomalies
        ax.scatter(
            x[det_mask],
            y[det_mask],
            marker="X",
            c="black",
            s=50,
            label=f"detector anomaly (n={det_count})",
        )

        # pattern anomalies
        ax.scatter(
            x[pat_mask],
            y[pat_mask],
            marker="o",
            c="red",
            s=50,
            label=f"pattern anomaly (n={pat_count})",
        )

        ax.set_ylabel(label, fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.3)

        # --> Legend outside on the right
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=20,
            frameon=False
        )

    axes[-1].set_xlabel("id_time", fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])  # extra right space for legend

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
