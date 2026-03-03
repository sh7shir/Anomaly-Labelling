import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.7463644592132516
MIN_VOTES_FOR_ALARM = 5
ROLLING_WINDOW = 653
PATTERN_STD_MULT = 4.773172549914575

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

def calculate_metrics(y_true, y_pred):
    """
    Calculates Precision, Recall, F1, and MCC.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0.0

    return precision, recall, f1, mcc

def analyze_folder(folder_path, folder_name, output_base, ground_truth_timestamps):
    # --- STEP 1: Load Master Data from 'orig' folder ---
    orig_path = os.path.join(folder_path, "orig")
    
    if not os.path.exists(orig_path):
        return 

    orig_files = glob.glob(os.path.join(orig_path, "*.csv"))
    if not orig_files:
        print(f"   [!] 'orig' folder exists but is empty in {folder_name}")
        return

    print(f"\nProcessing: {folder_name}...")
    
    # Load Master File
    master_file = orig_files[0]
    try:
        df = pd.read_csv(master_file)
        
        # Standardize Master Columns
        if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'timestep'}, inplace=True)
        if 'value' in df.columns: df.rename(columns={'value': 'orig_value'}, inplace=True)
            
        if 'timestep' not in df.columns or 'orig_value' not in df.columns:
            print(f"   [Error] Master file missing 'timestamp' or 'value' columns.")
            return

        # Convert to Datetime Objects
        df['dt_object'] = pd.to_datetime(df['timestep'])
        df = df.sort_values(by='dt_object').reset_index(drop=True)
        print(f"   -> Master Data Loaded: {len(df)} rows.")
        
    except Exception as e:
        print(f"   [Error] Failed to read master file: {e}")
        return

    # --- STEP 2: Merge Detector Results ---
    detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
    detector_score_cols = []

    for f in detector_files:
        try:
            det_df = pd.read_csv(f)
            
            # 1. Get Detector Name
            if 'algorithm' in det_df.columns:
                det_name = str(det_df['algorithm'].iloc[0])
            else:
                base = os.path.basename(f)
                det_name = base.replace("online_results_00001_", "").replace(f"_{folder_name}_origdata", "").replace(".csv", "")
            
            # 2. Identify Score Column
            score_col_source = None
            if 'anomaly_score' in det_df.columns: score_col_source = 'anomaly_score'
            elif 'score' in det_df.columns: score_col_source = 'score'
            elif 'value' in det_df.columns and 'timestamp' in det_df.columns: 
                score_col_source = 'value'
            
            if score_col_source is None:
                continue
                
            unique_score_col = f'score_{det_name}'
            det_df.rename(columns={score_col_source: unique_score_col}, inplace=True)
            
            # 3. MERGE STRATEGY
            merged_successfully = False
            
            # A. DateTime Merge (Preferred)
            if 'timestamp' in det_df.columns:
                det_df['dt_object'] = pd.to_datetime(det_df['timestamp'])
                temp_merge = pd.merge(df[['dt_object']], det_df[['dt_object', unique_score_col]], on='dt_object', how='left')
                
                if temp_merge[unique_score_col].notna().sum() > 0:
                    df[unique_score_col] = temp_merge[unique_score_col]
                    merged_successfully = True
            
            # B. Index Merge (Fallback)
            if not merged_successfully:
                det_df = det_df.reset_index(drop=True)
                df[unique_score_col] = det_df[unique_score_col]
                merged_successfully = True

            if merged_successfully:
                detector_score_cols.append(unique_score_col)

        except Exception as e:
            print(f"      [Error] Processing {os.path.basename(f)}: {e}")

    # Fill NaNs
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

    # B. Pattern Analysis
    if df['orig_value'].isnull().all():
        df['expected_pattern'] = 0
        df['is_pattern_anomaly'] = 0
    else:
        df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
        df['expected_pattern'] = df['orig_value'].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).median()
        df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
        
        threshold = df['pattern_deviation'].mean() + (PATTERN_STD_MULT * df['pattern_deviation'].std())
        df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # C. Combine (Union)
    df['is_final_anomaly'] = (df['is_detector_anomaly'] == 1) | (df['is_pattern_anomaly'] == 1)
    df['is_final_anomaly'] = df['is_final_anomaly'].astype(int)

    # D. Ground Truth Matching
    gt_count = 0
    df['is_ground_truth'] = False
    if ground_truth_timestamps:
        gt_dt = pd.to_datetime(ground_truth_timestamps)
        df['is_ground_truth'] = df['dt_object'].isin(gt_dt)
        gt_count = df['is_ground_truth'].sum()

    print(f"   -> Found {df['is_detector_anomaly'].sum()} Detector Alerts")
    print(f"   -> Found {df['is_pattern_anomaly'].sum()} Pattern Anomalies")
    print(f"   -> Found {df['is_final_anomaly'].sum()} Combined Anomalies")
    print(f"   -> Found {gt_count} Ground Truth Matches")

    # --- STEP 4: Calculate Metrics ---
    if gt_count > 0:
        # 1. Detector Consensus
        prec, rec, f1, mcc = calculate_metrics(df['is_ground_truth'], df['is_detector_anomaly'])
        print("-" * 40)
        print(f"   [METRICS] Detector Consensus Only:")
        print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

        # 2. Pattern Analysis
        prec_p, rec_p, f1_p, mcc_p = calculate_metrics(df['is_ground_truth'], df['is_pattern_anomaly'])
        print(f"   [METRICS] Pattern Analysis Only:")
        print(f"   Precision: {prec_p:.4f} | Recall: {rec_p:.4f} | F1: {f1_p:.4f} | MCC: {mcc_p:.4f}")

        # 3. Combined (Union)
        prec_c, rec_c, f1_c, mcc_c = calculate_metrics(df['is_ground_truth'], df['is_final_anomaly'])
        print(f"   [METRICS] Combined (Union):")
        print(f"   Precision: {prec_c:.4f} | Recall: {rec_c:.4f} | F1: {f1_c:.4f} | MCC: {mcc_c:.4f}")
        print("-" * 40)
    else:
        print("   [Info] No Ground Truth labels found. Skipping metrics calculation.")


    # --- STEP 5: Save & Visualize ---
    out_csv = os.path.join(output_base, "combined_csv")
    os.makedirs(out_csv, exist_ok=True)
    df.to_csv(os.path.join(out_csv, f"{folder_name}_combined.csv"), index=False)

    if 'orig_value' in df.columns:
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_title(f'Analysis: {folder_name}')
        
        indices = df.index
        # Plot Original Data
        ax.plot(indices, df['orig_value'], color='#1f77b4', alpha=0.6, label='Value')
        ax.plot(indices, df['expected_pattern'], color='black', linestyle='--', alpha=0.5, label='Trend')

        # Plot Detector Alerts
        det_idx = df.index[df['is_detector_anomaly'] == 1]
        if len(det_idx) > 0:
            ax.scatter(det_idx, df.loc[det_idx, 'orig_value'], c='black', marker='x', s=50, label='Detector', zorder=5)

        # Plot Pattern Alerts
        pat_idx = df.index[df['is_pattern_anomaly'] == 1]
        if len(pat_idx) > 0:
            ax.scatter(pat_idx, df.loc[pat_idx, 'orig_value'], c='red', marker='o', s=20, label='Pattern', zorder=10)

        # Plot Ground Truth
        if gt_count > 0:
            gt_idx = df.index[df['is_ground_truth'] == 1]
            ax.scatter(gt_idx, df.loc[gt_idx, 'orig_value'], c='green', marker='*', s=200, label='Ground Truth', zorder=4)
            for i in gt_idx:
                ax.axvline(x=i, color='green', alpha=0.3)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")
        
        # --- X-TICKS: Use 'Orig' Timestamps ---
        step = max(1, len(df) // 12)
        ax.set_xticks(indices[::step])
        ax.set_xticklabels(df['dt_object'].dt.strftime('%Y-%m-%d').iloc[::step], rotation=45, ha='right')

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
        print("\nNo valid folders found (Folders must contain an 'orig' subfolder).")