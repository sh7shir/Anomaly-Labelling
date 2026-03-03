import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.99
MIN_VOTES_FOR_ALARM = 3
ROLLING_WINDOW = 288
PATTERN_STD_MULT = 2.5

# NAB / Lavin and Ahmad Settings
ANOMALY_WINDOW_SIZE = 'NAB'      # can be int or 'NAB'
NORMALIZE_METRICS = True
ADJUST_ANOMALIES_POSEDGE = True

# Folder containing *_addlabels*.csv files
DATA_ONLINE_SUBDIR = "data_online"


# --- HELPER FUNCTIONS (LAVIN & AHMAD APPROACH) ---

def calculate_aws(orig_aws, nab_mode: bool, dataset_length: int, num_of_flags: int):
    """
    Calculates the size of the anomaly window (AW) based on dataset length and anomaly count.
    Ref: Lavin and Ahmad
    """
    if orig_aws == 'NAB' or nab_mode:
        if num_of_flags != 0:
            return int(np.ceil(.1 * dataset_length / num_of_flags))
        else:
            return int(np.ceil(.1 * dataset_length))
    else:
        return int(orig_aws)


def create_anomaly_windows(ground_truth: list, aws: int, length: int):
    """
    Creates the 'valid' windows around ground truth points.
    Returns a list where >0 indicates inside a valid window.
    ground_truth: list of integer indices (your GT indices)
    """
    anomaly_windows = [0] * length
    steps_since_middle = aws + 1
    overwrite_from = 0
    ground_truth_count = 0

    for y in range(length):
        # if we are at a timestep with a flagged anomaly:
        if ground_truth.count(y) > 0:
            steps_since_middle += 1
            ground_truth_count += 1
            # if two windows conflict, overwrite previous window
            if steps_since_middle < aws:
                overwrite_from = int(y - np.floor((steps_since_middle - 1) / 2))
            else:
                overwrite_from = y - aws

            # don't overwrite negative indices
            overwrite_from = max(overwrite_from, 0)

            # overwrite array with the anomaly window
            for x in range(max(overwrite_from, 0), min(y + aws + 1, length - 1)):
                anomaly_windows[x] = ground_truth_count
            steps_since_middle = 0
        else:
            if ground_truth_count > 0:
                steps_since_middle += 1

    return anomaly_windows


def adjust_anomaly_signals_posedge(detections, adjust: bool):
    """
    Only take into account no->yes changes (first timestamps) of the signal.
    detections: list of bool
    """
    if adjust:
        anomalies_adjusted = [False] * len(detections)
        for y in range(len(detections)):
            if y == 0:
                anomalies_adjusted[y] = detections[y]
            else:
                anomalies_adjusted[y] = detections[y] and (not detections[y - 1])
        return anomalies_adjusted
    else:
        return detections


def measure_confusion_matrix(detections, anomaly_windows, aws: int, normalize: bool):
    """
    Calculates TP, FP, TN, FN based on Lavin/Ahmad logic.
    detections: list of bool
    anomaly_windows: list of ints (0=normal, >0 = inside some anomaly window)
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    last_detected_anomaly = 0

    for y in range(len(detections)):
        if y == 0:
            continue

        # Outside anomaly window
        if anomaly_windows[y] == 0:
            # Just exited a window
            if anomaly_windows[y - 1] > 0:
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1

            if detections[y]:
                false_positives += 1

        # Inside anomaly window
        else:
            # If window changed or we just entered one
            if anomaly_windows[y - 1] != anomaly_windows[y] and anomaly_windows[y - 1] != 0:
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1

            if detections[y]:
                if last_detected_anomaly != anomaly_windows[y]:
                    true_positives += 1
                    last_detected_anomaly = anomaly_windows[y]

    # Handle the very last window case if dataset ends inside a window
    if anomaly_windows[-1] > 0 and last_detected_anomaly != anomaly_windows[-1]:
        false_negatives += 1

    if normalize:
        # Avoid division by zero
        denom = (2 * aws + 1) if aws > 0 else 1
        false_positives = false_positives / denom

        # Approximate TN calculation normalization
        true_negatives = len(detections) / denom - true_positives - false_positives - false_negatives
    else:
        true_negatives = len(detections) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, true_negatives, false_negatives


def calculate_metrics(tp, fp, tn, fn):
    """
    Returns Precision, Recall, F-Score, MCC (Adjusted).
    """
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom != 0 else -1
    mcc_adj = (mcc + 1) * 0.5

    return precision, recall, f_score, mcc_adj


# --- YOUR EXISTING GROUND-TRUTH LOADER (INDEX-BASED) ---

def load_ground_truth_indices_for_folder(data_online_dir, folder_name):
    """
    Load ground truth anomaly indices for a given folder from data_online/.

    It searches for: *{folder_name}*addlabels*.csv

    FORMAT:
      - 1st column: just an index (0,1,2,...) -> IGNORE
      - 2nd column: anomaly index in x-axis -> USE as anomaly positions

    Returns: list of integer indices where anomalies occur.
    """
    if not os.path.isdir(data_online_dir):
        print(f"   [Info] No data_online dir found, no ground-truth for {folder_name}.")
        return []

    pattern = os.path.join(data_online_dir, f"*{folder_name}*addlabels*.csv")
    matches = glob.glob(pattern)

    if not matches:
        print(f"   [Info] No *addlabels* file found in data_online for {folder_name}.")
        return []

    label_file = matches[0]
    print(f"   -> Using label file: {os.path.basename(label_file)}")

    try:
        lab = pd.read_csv(label_file, header=None)

        if lab.shape[1] < 2:
            print(
                f"   [Warning] Label file {os.path.basename(label_file)} "
                f"has < 2 columns, cannot extract anomaly indices."
            )
            return []

        # 2nd column is the anomaly index in x-axis
        anomaly_idx = (
            pd.to_numeric(lab.iloc[:, 1], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        print(f"   -> Loaded {len(anomaly_idx)} anomaly indices from labels.")
        return anomaly_idx

    except Exception as e:
        print(f"   [Warning] Could not load/parse label file for {folder_name}: {e}")
        return []


# --- MAIN ANALYSIS FUNCTION (INTEGRATED WITH LAVIN/AHMAD) ---

def analyze_folder(folder_path, folder_name, output_base, data_online_dir):
    # --- STEP 1: Load Master Data (Use first detector CSV as master) ---
    detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not detector_files:
        print(f"\n[!] No CSV files found in {folder_name}, skipping.")
        return

    print(f"\nProcessing: {folder_name}...")

    master_file = detector_files[0]
    try:
        df = pd.read_csv(master_file)

        # Standardize Master Columns
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'timestep'}, inplace=True)

        if 'value' in df.columns and 'orig_value' not in df.columns:
            df.rename(columns={'value': 'orig_value'}, inplace=True)

        if 'orig_value' not in df.columns:
            # If there is no obvious 'value' column, create a placeholder
            df['orig_value'] = np.nan

        df = df.reset_index(drop=True)
        print(f"   -> Master Data Loaded from {os.path.basename(master_file)}: {len(df)} rows.")

    except Exception as e:
        print(f"   [Error] Failed to read master file: {e}")
        return

    # --- STEP 2: Merge Detector Results (index-based only) ---
    detector_score_cols = []

    for f in detector_files:
        try:
            det_df = pd.read_csv(f)

            # 1. Get Detector Name (Column 'algorithm' OR Filename)
            if 'algorithm' in det_df.columns and pd.notna(det_df['algorithm'].iloc[0]):
                det_name = str(det_df['algorithm'].iloc[0])
            else:
                base = os.path.basename(f)
                det_name = (
                    base.replace("online_results_00001_", "")
                        .replace(f"_{folder_name}_origdata", "")
                        .replace(".csv", "")
                )

            # 2. Identify Score Column
            score_col_source = None
            if 'anomaly_score' in det_df.columns:
                score_col_source = 'anomaly_score'
            elif 'score' in det_df.columns:
                score_col_source = 'score'
            elif 'value' in det_df.columns:
                score_col_source = 'value'

            if score_col_source is None:
                print(f"      [Warning] No score column found in {os.path.basename(f)}, skipping.")
                continue

            unique_score_col = f'score_{det_name}'
            det_df.rename(columns={score_col_source: unique_score_col}, inplace=True)

            det_df = det_df.reset_index(drop=True)

            if len(det_df) != len(df):
                print(
                    f"      [Warning] {det_name} length mismatch "
                    f"({len(det_df)} vs {len(df)}). Aligning by truncating/padding."
                )

            # Align lengths: pad with NaN or truncate as needed
            merged_col = pd.Series(np.nan, index=range(len(df)), name=unique_score_col)
            min_len = min(len(df), len(det_df))
            merged_col.iloc[:min_len] = det_df[unique_score_col].iloc[:min_len].values
            df[unique_score_col] = merged_col

            detector_score_cols.append(unique_score_col)

        except Exception as e:
            print(f"      [Error] Processing {os.path.basename(f)}: {e}")

    # Fill NaNs in detector scores with 0
    if detector_score_cols:
        df[detector_score_cols] = df[detector_score_cols].fillna(0)
        print(f"   -> Successfully merged {len(detector_score_cols)} detectors.")
    else:
        print("   -> [Warning] No detector scores were merged!")

    # --- STEP 3: Calculations ---
    # A. Detector Consensus
    if detector_score_cols:
        df['detector_votes'] = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).sum(axis=1)
        df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)
    else:
        df['is_detector_anomaly'] = 0

    # B. Pattern Analysis (on orig_value, using index as time)
    if df['orig_value'].isnull().all():
        df['expected_pattern'] = 0
        df['pattern_deviation'] = 0
        df['is_pattern_anomaly'] = 0
    else:
        df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
        df['expected_pattern'] = (
            df['orig_value']
            .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
            .median()
        )
        df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])

        threshold = (
            df['pattern_deviation'].mean() +
            (PATTERN_STD_MULT * df['pattern_deviation'].std())
        )
        df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # C. Ground Truth: indices from 2nd column of *_addlabels*.csv
    gt_indices = load_ground_truth_indices_for_folder(data_online_dir, folder_name)
    df['is_ground_truth'] = False
    if gt_indices:
        for idx in gt_indices:
            if 0 <= idx < len(df):
                df.at[idx, 'is_ground_truth'] = True
    gt_count = int(df['is_ground_truth'].sum())

    # D. Final Combined Anomaly Signal
    df['is_final_anomaly'] = (
        (df['is_detector_anomaly'] == 1) | (df['is_pattern_anomaly'] == 1)
    ).astype(int)

    print(f"   -> Found {df['is_detector_anomaly'].sum()} Detector Alerts")
    print(f"   -> Found {df['is_pattern_anomaly'].sum()} Pattern Anomalies")
    print(f"   -> Found {df['is_final_anomaly'].sum()} Final Anomalies (Detector OR Pattern)")
    print(f"   -> Found {gt_count} Ground Truth Points")

    # --- STEP 3.5: Lavin/Ahmad Metrics based on index GT ---
    aws = 0
    if gt_count > 0:
        # A. Calculate Window Size (AW)
        aws = calculate_aws(
            ANOMALY_WINDOW_SIZE,
            ANOMALY_WINDOW_SIZE == 'NAB',
            len(df),
            gt_count
        )
        print(f"   -> Calculated Anomaly Window (AW): {aws} steps")

        # B. Create Windows
        anomaly_windows = create_anomaly_windows(gt_indices, aws, len(df))

        # C. Adjust Detections (PosEdge)
        raw_detections = df['is_final_anomaly'].astype(bool).tolist()
        final_detections = adjust_anomaly_signals_posedge(
            raw_detections, ADJUST_ANOMALIES_POSEDGE
        )

        # D. Confusion Matrix
        tp, fp, tn, fn = measure_confusion_matrix(
            final_detections, anomaly_windows, aws, NORMALIZE_METRICS
        )

        # E. Metrics
        prec, rec, f1, mcc = calculate_metrics(tp, fp, tn, fn)

        print("-" * 50)
        print(f"   [METRICS - LAVIN/AHMAD (NAB)]")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F-Score:   {f1:.4f}")
        print(f"   MCC(adj):  {mcc:.4f}")
        print(f"   TP={tp:.3f}, FP={fp:.3f}, TN={tn:.3f}, FN={fn:.3f}")
        print("-" * 50)
    else:
        print("   [Info] No Ground Truth. Skipping NAB metrics.")

    # --- STEP 4: Save & Visualize ---
    out_csv = os.path.join(output_base, "combined_csv")
    os.makedirs(out_csv, exist_ok=True)
    df.to_csv(os.path.join(out_csv, f"{folder_name}_combined.csv"), index=False)

    fig, ax = plt.subplots(figsize=(18, 8))

    total_combined = int(df['is_final_anomaly'].sum())
    ax.set_title(
        f'Analysis: {folder_name}\n'
        f'AW={aws} | Final Anomalies={total_combined} '
        f'(Detector OR Pattern)'
    )

    indices = df.index

    # Original signal + trend
    ax.plot(indices, df['orig_value'], alpha=0.6, label='Value')
    ax.plot(indices, df['expected_pattern'], color='black',
            linestyle='--', alpha=0.5, label='Trend')

    # Detector anomalies
    det_idx = df.index[df['is_detector_anomaly'] == 1]
    det_count = len(det_idx)
    if det_count > 0:
        ax.scatter(
            det_idx,
            df.loc[det_idx, 'orig_value'],
            c='black',
            marker='x',
            s=50,
            label=f"Detector (n={det_count})",
            zorder=5
        )

    # Pattern anomalies
    pat_idx = df.index[df['is_pattern_anomaly'] == 1]
    pat_count = len(pat_idx)
    if pat_count > 0:
        ax.scatter(
            pat_idx,
            df.loc[pat_idx, 'orig_value'],
            c='red',
            marker='o',
            s=20,
            label=f"Pattern Break (n={pat_count})",
            zorder=10
        )

    # Ground truth + windows
    if gt_count > 0:
        gt_idx = df.index[df['is_ground_truth'] == 1]
        ax.scatter(
            gt_idx,
            df.loc[gt_idx, 'orig_value'],
            c='green',
            marker='*',
            s=200,
            label=f"Ground Truth (n={gt_count})",
            zorder=4
        )
        # Draw vertical lines
        for i in gt_idx:
            ax.axvline(x=i, color='green', alpha=0.3)
        # Draw AW spans
        if aws > 0:
            for i in gt_idx:
                start_w = max(0, i - aws)
                end_w = min(len(df), i + aws)
                ax.axvspan(start_w, end_w, color='green', alpha=0.08)

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Value")
    ax.set_xlabel("Index")

    step = max(1, len(df) // 12)
    ax.set_xticks(indices[::step])

    out_plot = os.path.join(output_base, "plots")
    os.makedirs(out_plot, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_plot, f"plot_{folder_name}.png"))
    plt.show()
    plt.close()


# --- EXECUTION ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    data_online_dir = os.path.join(base_dir, DATA_ONLINE_SUBDIR)

    print(f"Scanning {base_dir} for data folders...")

    found_any = False
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue

            # Skip results directory and data_online itself
            if entry.name in [os.path.basename(results_dir), DATA_ONLINE_SUBDIR]:
                continue

            folder_path = entry.path
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            if not csv_files:
                continue

            analyze_folder(folder_path, entry.name, results_dir, data_online_dir)
            found_any = True

    if found_any:
        print(f"\nDone. Results saved to: {results_dir}")
    else:
        print("\nNo valid folders with CSV files found.")
