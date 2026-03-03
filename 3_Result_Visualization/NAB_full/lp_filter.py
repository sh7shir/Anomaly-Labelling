import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.99   
MIN_VOTES_FOR_ALARM = 3
ROLLING_WINDOW = 288  
PATTERN_STD_MULT = 3.0
EVALUATION_WINDOW = 100 

# --- FILTER SETTINGS ---
APPLY_FILTER = True
FILTER_ORDER = 2
# Cutoff frequency as a fraction of the Nyquist frequency (0.0 to 1.0).
# Lower = Smoother. 0.1 means removing top 90% of high frequencies.
FILTER_CUTOFF_FREQ = 0.1 

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

def get_continuous_events(binary_series):
    if binary_series.sum() == 0:
        return []
    diff = np.diff(np.concatenate(([0], binary_series.values, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    events = []
    for s, e in zip(starts, ends):
        events.append((s, e - 1)) 
    return events

def calculate_window_metrics(df, pred_col, gt_timestamps, window_size=EVALUATION_WINDOW):
    if not gt_timestamps or pred_col not in df.columns:
        return 0, 0, 0, 0

    gt_indices = df.index[df['is_ground_truth'] == 1].tolist()
    total_gt = len(gt_indices)
    pred_events = get_continuous_events(df[pred_col])
    total_pred = len(pred_events)

    if total_gt == 0:
        return 0, 0, 0, 0

    tp_gt = 0
    used_pred_events = set()

    for gt_idx in gt_indices:
        start_w = max(0, gt_idx - window_size)
        end_w = min(len(df) - 1, gt_idx + window_size)
        
        found = False
        for i, (p_start, p_end) in enumerate(pred_events):
            if p_start <= end_w and start_w <= p_end:
                found = True
                used_pred_events.add(i)
        
        if found:
            tp_gt += 1

    recall = tp_gt / total_gt
    tp_pred = len(used_pred_events)
    precision = tp_pred / total_pred if total_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, 0.0

def apply_lowpass_filter(data, cutoff=FILTER_CUTOFF_FREQ, order=FILTER_ORDER):
    """
    Applies a Butterworth low-pass filter to the data.
    """
    # 1. Handle NaNs (Interpolate)
    # Filter requires continuous data without NaNs
    clean_data = data.interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    
    # 2. Design Filter
    # b, a = numerator/denominator polynomials
    b, a = butter(order, cutoff, btype='low', analog=False)
    
    # 3. Apply Filter (filtfilt applies forward & backward to eliminate phase lag)
    y = filtfilt(b, a, clean_data)
    return y

def analyze_folder(folder_path, folder_name, output_base, ground_truth_timestamps):
    # --- STEP 1: Load Master Data ---
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
        
        if 'timestep' not in df.columns or 'orig_value' not in df.columns: return

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
            if 'algorithm' in det_df.columns:
                det_name = str(det_df['algorithm'].iloc[0])
            else:
                base = os.path.basename(f)
                det_name = base.replace("online_results_00001_", "").replace(f"_{folder_name}_origdata", "").replace(".csv", "")
            
            score_col_source = None
            if 'anomaly_score' in det_df.columns: score_col_source = 'anomaly_score'
            elif 'score' in det_df.columns: score_col_source = 'score'
            elif 'value' in det_df.columns and 'timestamp' in det_df.columns: score_col_source = 'value'
            
            if score_col_source is None: continue
                
            unique_score_col = f'score_{det_name}'
            det_df.rename(columns={score_col_source: unique_score_col}, inplace=True)
            
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
                merged_successfully = True

            if merged_successfully:
                detector_score_cols.append(unique_score_col)

        except Exception as e:
            print(f"      [Error] Processing {os.path.basename(f)}: {e}")

    if detector_score_cols:
        df[detector_score_cols] = df[detector_score_cols].fillna(0)
    else:
        print("   -> [Warning] No detector scores were merged!")

    # --- STEP 3: Calculations ---
    
    # 3a. Filter Signal (New Step)
    if APPLY_FILTER and 'orig_value' in df.columns:
        print(f"   -> Applying Low-Pass Filter (Cutoff={FILTER_CUTOFF_FREQ})...")
        df['proc_value'] = apply_lowpass_filter(df['orig_value'])
    else:
        df['proc_value'] = df['orig_value']

    # 3b. Detector Consensus
    if detector_score_cols:
        df['detector_votes'] = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).sum(axis=1)
        df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)
    else:
        df['is_detector_anomaly'] = 0

    # 3c. Pattern Analysis (Using Filtered 'proc_value')
    if df['proc_value'].isnull().all():
        df['expected_pattern'] = 0
        df['is_pattern_anomaly'] = 0
    else:
        # Interpolate again just in case, though filter handles it
        df['proc_value'] = df['proc_value'].interpolate(limit_direction='both')
        
        # Calculate Trend
        df['expected_pattern'] = df['proc_value'].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).median()
        
        # Calculate Deviation
        df['pattern_deviation'] = np.abs(df['proc_value'] - df['expected_pattern'])
        
        # Dynamic Threshold
        threshold = df['pattern_deviation'].mean() + (PATTERN_STD_MULT * df['pattern_deviation'].std())
        df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # 3d. Combined Analysis
    df['is_final_anomaly'] = (df['is_detector_anomaly'] == 1) | (df['is_pattern_anomaly'] == 1)
    df['is_final_anomaly'] = df['is_final_anomaly'].astype(int)

    # 3e. Ground Truth
    gt_count = 0
    df['is_ground_truth'] = False
    if ground_truth_timestamps:
        gt_dt = pd.to_datetime(ground_truth_timestamps)
        df['is_ground_truth'] = df['dt_object'].isin(gt_dt)
        gt_count = df['is_ground_truth'].sum()

    print(f"   -> Found {df['is_final_anomaly'].sum()} Total Combined Anomalies")
    print(f"   -> Found {gt_count} Ground Truth Matches")

    # --- STEP 4: Metrics ---
    if gt_count > 0:
        print("-" * 50)
        print(f"   [METRICS - WINDOW BASED (Win={EVALUATION_WINDOW})]")
        p, r, f, _ = calculate_window_metrics(df, 'is_final_anomaly', ground_truth_timestamps)
        print(f"   Combined Results:    P={p:.4f} | R={r:.4f} | F1={f:.4f}")
        print("-" * 50)

    # --- STEP 5: Save & Visualize ---
    out_csv = os.path.join(output_base, "combined_csv")
    os.makedirs(out_csv, exist_ok=True)
    df.to_csv(os.path.join(out_csv, f"{folder_name}_combined.csv"), index=False)

    if 'orig_value' in df.columns:
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_title(f'Combined Analysis (Filtered): {folder_name}')
        
        indices = df.index
        
        # Plot Raw Data (Faint)
        ax.plot(indices, df['orig_value'], color='gray', alpha=0.3, label='Raw Value')
        
        # Plot Filtered Data (Solid)
        ax.plot(indices, df['proc_value'], color='#1f77b4', alpha=0.9, linewidth=1.5, label='Filtered Value')
        
        # Plot Pattern Trend
        ax.plot(indices, df['expected_pattern'], color='black', linestyle='--', alpha=0.5, label='Trend')

        # Plot Anomalies
        det_idx = df.index[df['is_detector_anomaly'] == 1]
        if len(det_idx) > 0:
            ax.scatter(det_idx, df.loc[det_idx, 'proc_value'], c='black', marker='x', s=50, label='Detector Alert', zorder=5)

        pat_idx = df.index[df['is_pattern_anomaly'] == 1]
        if len(pat_idx) > 0:
            ax.scatter(pat_idx, df.loc[pat_idx, 'proc_value'], c='red', marker='o', s=20, label='Pattern Alert', zorder=6)

        if gt_count > 0:
            gt_idx = df.index[df['is_ground_truth'] == 1]
            ax.scatter(gt_idx, df.loc[gt_idx, 'proc_value'], c='green', marker='*', s=200, label='Ground Truth', zorder=7)
            for i in gt_idx:
                ax.axvspan(max(0, i-EVALUATION_WINDOW), min(len(df), i+EVALUATION_WINDOW), color='green', alpha=0.1)
                ax.axvline(x=i, color='green', alpha=0.3)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")
        
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
        print("\nNo valid folders found.")