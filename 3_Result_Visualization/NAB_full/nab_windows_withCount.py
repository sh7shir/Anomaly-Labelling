import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.99
MIN_VOTES_FOR_ALARM = 3
ROLLING_WINDOW = 288  
PATTERN_STD_MULT = 2.5

# NAB / Lavin and Ahmad Settings
ANOMALY_WINDOW_SIZE = 'NAB'
NORMALIZE_METRICS = True
ADJUST_ANOMALIES_POSEDGE = True

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
    """
    anomaly_windows = list([0] * length)
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
            overwrite_from = np.max([overwrite_from, 0])
            
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
    """
    if adjust:
        anomalies_adjusted = list([False] * len(detections))
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
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    last_detected_anomaly = 0

    for y in range(len(detections)):
        if y == 0: continue
        
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
    denom = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    mcc = (tp*tn - fp*fn) / denom if denom != 0 else -1
    mcc_adj = (mcc + 1) * 0.5 

    return precision, recall, f_score, mcc_adj

# --- DATA LOADING UTILS ---

def load_ground_truth(base_dir):
    label_file = os.path.join(base_dir, "combined_labels.json")
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def get_labels_for_folder(folder_name, all_labels):
    if not all_labels: return []
    for filename, timestamps in all_labels.items():
        if folder_name in filename:
            return timestamps
    return []

# --- MAIN ANALYSIS LOGIC ---

def analyze_folder(folder_path, folder_name, output_base, ground_truth_timestamps):
    # 1. Load Master Data
    orig_path = os.path.join(folder_path, "orig")
    if not os.path.exists(orig_path): return 

    orig_files = glob.glob(os.path.join(orig_path, "*.csv"))
    if not orig_files: return

    print(f"\nProcessing: {folder_name}...")
    master_file = orig_files[0]
    
    try:
        df = pd.read_csv(master_file)
        if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'timestep'}, inplace=True)
        if 'value' in df.columns: df.rename(columns={'value': 'orig_value'}, inplace=True)
            
        df['dt_object'] = pd.to_datetime(df['timestep'])
        df = df.sort_values(by='dt_object').reset_index(drop=True)
    except Exception as e:
        print(f"   [Error] Failed to read master file: {e}")
        return

    # 2. Merge Detector Results
    detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
    detector_score_cols = []

    for f in detector_files:
        try:
            det_df = pd.read_csv(f)
            # Basic name extraction
            base = os.path.basename(f)
            det_name = base.replace("online_results_00001_", "").replace(f"_{folder_name}_origdata", "").replace(".csv", "")
            
            score_col_source = None
            if 'anomaly_score' in det_df.columns: score_col_source = 'anomaly_score'
            elif 'score' in det_df.columns: score_col_source = 'score'
            elif 'value' in det_df.columns: score_col_source = 'value'
            
            if score_col_source:
                unique_score_col = f'score_{det_name}'
                det_df.rename(columns={score_col_source: unique_score_col}, inplace=True)
                
                # Merge
                merged_successfully = False
                if 'timestamp' in det_df.columns:
                    det_df['dt_object'] = pd.to_datetime(det_df['timestamp'])
                    temp_merge = pd.merge(df[['dt_object']], det_df[['dt_object', unique_score_col]], on='dt_object', how='left')
                    if temp_merge[unique_score_col].notna().sum() > 0:
                        df[unique_score_col] = temp_merge[unique_score_col]
                        merged_successfully = True
                
                if not merged_successfully:
                    det_df = det_df.reset_index(drop=True)
                    df[unique_score_col] = det_df[unique_score_col]
                
                detector_score_cols.append(unique_score_col)

        except Exception as e:
            pass

    if detector_score_cols:
        df[detector_score_cols] = df[detector_score_cols].fillna(0)

    # 3. Calculations (Vote & Pattern)
    if detector_score_cols:
        df['detector_votes'] = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).sum(axis=1)
        df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)
    else:
        df['is_detector_anomaly'] = 0

    # Pattern check
    df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
    df['expected_pattern'] = df['orig_value'].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).median()
    df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
    threshold = df['pattern_deviation'].mean() + (PATTERN_STD_MULT * df['pattern_deviation'].std())
    df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # Final Combined
    df['is_final_anomaly'] = (df['is_detector_anomaly'] == 1) | (df['is_pattern_anomaly'] == 1)
    df['is_final_anomaly'] = df['is_final_anomaly'].astype(int)

    # 4. Ground Truth & Metrics (Lavin/Ahmad Integration)
    gt_count = 0
    aws = 0 
    
    # Identify GT rows
    df['is_ground_truth'] = False
    gt_indices = []
    
    if ground_truth_timestamps:
        gt_dt = pd.to_datetime(ground_truth_timestamps)
        df['is_ground_truth'] = df['dt_object'].isin(gt_dt)
        gt_indices = df.index[df['is_ground_truth']].tolist()
        gt_count = len(gt_indices)

    print(f"   -> Found {df['is_final_anomaly'].sum()} Total Anomalies detected")
    print(f"   -> Found {gt_count} Ground Truth Labels")

    if gt_count > 0:
        # A. Calculate Window Size (AW)
        aws = calculate_aws(ANOMALY_WINDOW_SIZE, ANOMALY_WINDOW_SIZE == 'NAB', len(df), gt_count)
        print(f"   -> Calculated Anomaly Window (AW): {aws} steps")

        # B. Create Windows
        anomaly_windows = create_anomaly_windows(gt_indices, aws, len(df))

        # C. Adjust Detections (PosEdge)
        raw_detections = df['is_final_anomaly'].astype(bool).tolist()
        final_detections = adjust_anomaly_signals_posedge(raw_detections, ADJUST_ANOMALIES_POSEDGE)

        # D. Confusion Matrix
        tp, fp, tn, fn = measure_confusion_matrix(final_detections, anomaly_windows, aws, NORMALIZE_METRICS)

        # E. Metrics
        prec, rec, f1, mcc = calculate_metrics(tp, fp, tn, fn)

        print("-" * 50)
        print(f"   [METRICS - LAVIN/AHMAD (NAB)]")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F-Score:   {f1:.4f}")
        print(f"   MCC(adj):  {mcc:.4f}")
        print("-" * 50)
    else:
        print("   [Info] No Ground Truth. Skipping metrics.")

    # 5. Save & Plot
    out_csv = os.path.join(output_base, "combined_csv")
    os.makedirs(out_csv, exist_ok=True)
    df.to_csv(os.path.join(out_csv, f"{folder_name}_combined.csv"), index=False)

    if 'orig_value' in df.columns:
        # Calculate Counts
        det_idx = df.index[df['is_detector_anomaly'] == 1]
        pat_idx = df.index[df['is_pattern_anomaly'] == 1]
        
        det_count = len(det_idx)
        pat_count = len(pat_idx)
        total_combined = df['is_final_anomaly'].sum()

        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Update Title with Total Count
        ax.set_title(f'Analysis: {folder_name}\nLavin/Ahmad Metrics (AW={aws}) | Total Anomalies Detected: {total_combined}')
        
        indices = df.index
        # Plot using indices
        ax.plot(indices, df['orig_value'], color='#1f77b4', alpha=0.6, label='Value')
        ax.plot(indices, df['expected_pattern'], color='black', linestyle='--', alpha=0.5, label='Trend')

        # Plot Alerts
        if len(det_idx) > 0:
            ax.scatter(det_idx, df.loc[det_idx, 'orig_value'], c='black', marker='x', s=50, 
                       label=f'Detector Alert (n={det_count})', zorder=8)
        
        if len(pat_idx) > 0:
            ax.scatter(pat_idx, df.loc[pat_idx, 'orig_value'], c='red', marker='o', s=20, 
                       label=f'Pattern Alert (n={pat_count})', zorder=9)

        # Plot Ground Truth & Valid Windows
        if gt_count > 0:
            gt_idx = df.index[df['is_ground_truth'] == 1]
            ax.scatter(gt_idx, df.loc[gt_idx, 'orig_value'], c='green', marker='*', s=200, label='Ground Truth', zorder=7)
            
            # Visualize the Calculated AW (Lavin/Ahmad Window)
            for i in gt_idx:
                start_w = max(0, i - aws)
                end_w = min(len(df), i + aws)
                ax.axvspan(start_w, end_w, color='green', alpha=0.1) 
                ax.axvline(x=i, color='green', alpha=0.3)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")
        
        # Format X-axis to use INDEX
        step = max(1, len(df) // 12)
        ax.set_xticks(indices[::step])
        ax.set_xticklabels(indices[::step]) # No rotation needed usually for simple integers, but added checks if needed
        ax.set_xlabel("Time Step (Index)")

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
    
    print(f"Scanning {base_dir} for data folders...")
    labels_map = load_ground_truth(base_dir)

    found_any = False
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                if os.path.exists(os.path.join(entry.path, "orig")):
                    current_labels = get_labels_for_folder(entry.name, labels_map)
                    analyze_folder(entry.path, entry.name, results_dir, current_labels)
                    found_any = True

    if found_any:
        print(f"\nDone. Results saved to: {results_dir}")
    else:
        print("\nNo valid folders found.")