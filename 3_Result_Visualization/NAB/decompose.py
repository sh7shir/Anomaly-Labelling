import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.85   
MIN_VOTES_FOR_ALARM = 2
ROLLING_WINDOW = 288  
PATTERN_STD_MULT = 3.0
EVALUATION_WINDOW = 50 

# --- DECOMPOSITION SETTINGS ---
USE_DECOMPOSITION = True       
DECOMP_PERIOD = 288            # 24 hours * 12 (5-min intervals)

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
    if binary_series.sum() == 0: return []
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

    if total_gt == 0: return 0, 0, 0, 0

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
        if found: tp_gt += 1

    recall = tp_gt / total_gt
    tp_pred = len(used_pred_events)
    precision = tp_pred / total_pred if total_pred > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, 0.0

def decompose_signal(series, period):
    """
    Performs a robust Seasonal-Trend decomposition.
    Returns: trend, seasonal, residual
    """
    # 1. Trend: Rolling Median (Robust to anomalies)
    trend = series.rolling(window=period, center=True, min_periods=1).median()
    
    # 2. Detrend
    detrended = series - trend
    
    # 3. Seasonality: Group by time-of-day index
    # We assume data is continuous 5-min intervals
    # Create an index 0..287 repeating
    time_indices = np.arange(len(series)) % period
    
    # Calculate median value for each time slot
    seasonal_map = detrended.groupby(time_indices).median()
    
    # Map back to full series
    seasonal = pd.Series(time_indices).map(seasonal_map)
    
    # 4. Residual
    residual = detrended - seasonal
    
    # Fill NaNs (edges of trend) with 0
    residual = residual.fillna(0)
    
    return trend, seasonal, residual

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
        
        # Interpolate small gaps for cleaner decomposition
        df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
        
    except Exception as e:
        print(f"   [Error] {e}")
        return

    # --- STEP 2: Merge Detectors ---
    detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
    detector_score_cols = []
    for f in detector_files:
        try:
            det_df = pd.read_csv(f)
            if 'algorithm' in det_df.columns:
                det_name = str(det_df['algorithm'].iloc[0])
            else:
                det_name = os.path.basename(f).replace("online_results_00001_", "").replace(f"_{folder_name}_origdata", "").replace(".csv", "")
            
            score_col = 'anomaly_score' if 'anomaly_score' in det_df.columns else ('score' if 'score' in det_df.columns else ('value' if 'value' in det_df.columns and 'timestamp' in det_df.columns else None))
            if not score_col: continue

            unique_col = f'score_{det_name}'
            det_df.rename(columns={score_col: unique_col}, inplace=True)
            
            if 'timestamp' in det_df.columns:
                det_df['dt_object'] = pd.to_datetime(det_df['timestamp'])
                temp = pd.merge(df[['dt_object']], det_df[['dt_object', unique_col]], on='dt_object', how='left')
                if temp[unique_col].notna().sum() > 0:
                    df[unique_col] = temp[unique_col]
                else:
                    df[unique_col] = det_df[unique_col].reset_index(drop=True)
            else:
                df[unique_col] = det_df[unique_col].reset_index(drop=True)
            
            detector_score_cols.append(unique_col)
        except: pass

    if detector_score_cols:
        df[detector_score_cols] = df[detector_score_cols].fillna(0)

    # --- STEP 3: ANALYSIS LOGIC ---

    # 3a. Signal Decomposition
    if USE_DECOMPOSITION:
        trend, seasonal, residual = decompose_signal(df['orig_value'], DECOMP_PERIOD)
        df['trend_comp'] = trend
        df['seasonal_comp'] = seasonal
        df['residual_comp'] = residual
        
        # We run pattern detection on the RESIDUAL (The Noise)
        df['analysis_metric'] = df['residual_comp'].abs()
    else:
        df['analysis_metric'] = df['orig_value']

    # 3b. Pattern Logic (On Residual)
    # We look for residuals that are much larger than the typical noise level
    # Calculate dynamic threshold based on the residual's own statistics
    
    # We use a short rolling window on the residual to find local bursts of noise
    df['local_noise_lvl'] = df['analysis_metric'].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).median()
    df['noise_deviation'] = np.abs(df['analysis_metric'] - df['local_noise_lvl'])
    
    # Threshold: Mean Noise + 3 * Std Noise
    threshold = df['analysis_metric'].mean() + (PATTERN_STD_MULT * df['analysis_metric'].std())
    
    # Alternatively: Simple Z-score on residual
    # threshold = df['analysis_metric'].std() * PATTERN_STD_MULT
    
    df['is_pattern_anomaly'] = (df['analysis_metric'] > threshold).astype(int)

    # 3c. Detector Logic
    if detector_score_cols:
        df['detector_votes'] = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).sum(axis=1)
        df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)
    else:
        df['is_detector_anomaly'] = 0

    # 3d. Combined (Union)
    df['is_final_anomaly'] = (df['is_detector_anomaly'] == 1) | (df['is_pattern_anomaly'] == 1)
    df['is_final_anomaly'] = df['is_final_anomaly'].astype(int)

    # 3e. Ground Truth
    gt_count = 0
    df['is_ground_truth'] = False
    if ground_truth_timestamps:
        gt_dt = pd.to_datetime(ground_truth_timestamps)
        df['is_ground_truth'] = df['dt_object'].isin(gt_dt)
        gt_count = df['is_ground_truth'].sum()

    # --- STEP 4: Metrics ---
    if gt_count > 0:
        print("-" * 50)
        print(f"   [METRICS - WINDOW BASED (Win={EVALUATION_WINDOW})]")
        p, r, f, _ = calculate_window_metrics(df, 'is_final_anomaly', ground_truth_timestamps)
        print(f"   Final Results:       P={p:.4f} | R={r:.4f} | F1={f:.4f}")
        print("-" * 50)

    # --- STEP 5: Save & Visualize ---
    out_csv = os.path.join(output_base, "combined_csv")
    os.makedirs(out_csv, exist_ok=True)
    df.to_csv(os.path.join(out_csv, f"{folder_name}_combined.csv"), index=False)

    if 'orig_value' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        fig.suptitle(f'Decomposition Analysis: {folder_name}', fontsize=16)
        
        indices = df.index
        
        # --- SUBPLOT 1: Main Signal + Anomalies ---
        ax1.set_title("Original Signal & Final Anomalies")
        ax1.plot(indices, df['orig_value'], color='#1f77b4', alpha=0.7, label='Original')
        if USE_DECOMPOSITION:
             ax1.plot(indices, df['trend_comp'] + df['seasonal_comp'], color='black', linestyle='--', alpha=0.4, label='Trend + Seasonal')

        # Anomalies
        final_idx = df.index[df['is_final_anomaly'] == 1]
        if len(final_idx) > 0:
            ax1.scatter(final_idx, df.loc[final_idx, 'orig_value'], c='red', marker='o', s=40, label='Final Anomaly', zorder=10)

        # Ground Truth
        if gt_count > 0:
            gt_idx = df.index[df['is_ground_truth'] == 1]
            ax1.scatter(gt_idx, df.loc[gt_idx, 'orig_value'], c='green', marker='*', s=200, label='Ground Truth', zorder=20)
            for i in gt_idx:
                ax1.axvspan(max(0, i-EVALUATION_WINDOW), min(len(df), i+EVALUATION_WINDOW), color='green', alpha=0.1)

        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Value")

        # --- SUBPLOT 2: The Residuals (What we analyzed) ---
        if USE_DECOMPOSITION:
            ax2.set_title("Signal Residuals (Noise) - Pattern Detection happens here")
            ax2.plot(indices, df['residual_comp'], color='purple', alpha=0.7, label='Residual')
            
            # Show threshold
            # Since threshold is dynamic/scalar, we can't easily plot a line unless we store it as a column
            # But we can plot the anomaly points on the residual
            pat_idx = df.index[df['is_pattern_anomaly'] == 1]
            if len(pat_idx) > 0:
                ax2.scatter(pat_idx, df.loc[pat_idx, 'residual_comp'], c='red', marker='x', s=30, label='High Residual')

            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylabel("Residual")

        # X-TICKS
        step = max(1, len(df) // 12)
        ax2.set_xticks(indices[::step])
        ax2.set_xticklabels(df['dt_object'].dt.strftime('%Y-%m-%d').iloc[::step], rotation=45, ha='right')

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
    if found_any: print(f"\nDone. Results saved to: {results_dir}")