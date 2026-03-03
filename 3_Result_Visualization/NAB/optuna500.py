import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
import optuna
import logging

# Set Optuna logging to warning to avoid spamming the console
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIGURATION (FIXED) ---
# NAB / Lavin and Ahmad Settings
ANOMALY_WINDOW_SIZE = 'NAB'
NORMALIZE_METRICS = True
ADJUST_ANOMALIES_POSEDGE = True

# --- HELPER FUNCTIONS (LAVIN & AHMAD) ---

def calculate_aws(orig_aws, nab_mode: bool, dataset_length: int, num_of_flags: int):
    if orig_aws == 'NAB' or nab_mode:
        if num_of_flags != 0:
            return int(np.ceil(.1 * dataset_length / num_of_flags))
        else:
            return int(np.ceil(.1 * dataset_length))
    else:
        return int(orig_aws)

def create_anomaly_windows(ground_truth: list, aws: int, length: int):
    anomaly_windows = list([0] * length)
    steps_since_middle = aws + 1
    overwrite_from = 0
    ground_truth_count = 0

    for y in range(length):
        if ground_truth.count(y) > 0:
            steps_since_middle += 1
            ground_truth_count += 1
            if steps_since_middle < aws:
                overwrite_from = int(y - np.floor((steps_since_middle - 1) / 2))
            else:
                overwrite_from = y - aws
            overwrite_from = np.max([overwrite_from, 0])
            for x in range(max(overwrite_from, 0), min(y + aws + 1, length - 1)):
                anomaly_windows[x] = ground_truth_count
            steps_since_middle = 0
        else:
            if ground_truth_count > 0:
                steps_since_middle += 1
    return anomaly_windows

def adjust_anomaly_signals_posedge(detections, adjust: bool):
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
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    last_detected_anomaly = 0

    for y in range(len(detections)):
        if y == 0: continue
        if anomaly_windows[y] == 0:
            if anomaly_windows[y - 1] > 0:
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            if detections[y]:
                false_positives += 1
        else:
            if anomaly_windows[y - 1] != anomaly_windows[y] and anomaly_windows[y - 1] != 0:
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            if detections[y]:
                if last_detected_anomaly != anomaly_windows[y]:
                    true_positives += 1
                    last_detected_anomaly = anomaly_windows[y]

    if anomaly_windows[-1] > 0 and last_detected_anomaly != anomaly_windows[-1]:
        false_negatives += 1

    if normalize:
        denom = (2 * aws + 1) if aws > 0 else 1
        false_positives = false_positives / denom
        true_negatives = len(detections) / denom - true_positives - false_positives - false_negatives
    else:
        true_negatives = len(detections) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, true_negatives, false_negatives

def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    denom = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    mcc = (tp*tn - fp*fn) / denom if denom != 0 else -1
    mcc_adj = (mcc + 1) * 0.5 
    return precision, recall, f_score, mcc_adj

# --- DATA LOADING (RUN ONCE) ---

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

def preload_data(base_dir):
    """
    Reads all CSVs, merges detector columns, and prepares everything needed 
    for the optimization loop so we don't touch the disk during Optuna trials.
    """
    print("--- Pre-loading Data for Optimization ---")
    labels_map = load_ground_truth(base_dir)
    loaded_datasets = []

    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, "orig")):
                folder_name = entry.name
                folder_path = entry.path
                
                # 1. Load Master
                orig_files = glob.glob(os.path.join(folder_path, "orig", "*.csv"))
                if not orig_files: continue
                
                try:
                    df = pd.read_csv(orig_files[0])
                    if 'timestamp' in df.columns: df.rename(columns={'timestamp': 'timestep'}, inplace=True)
                    if 'value' in df.columns: df.rename(columns={'value': 'orig_value'}, inplace=True)
                    df['dt_object'] = pd.to_datetime(df['timestep'])
                    df = df.sort_values(by='dt_object').reset_index(drop=True)
                    
                    # Interpolate values once (saves time in loop)
                    df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
                except:
                    continue

                # 2. Load and Merge Detectors
                detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
                detector_score_cols = []
                
                for f in detector_files:
                    try:
                        det_df = pd.read_csv(f)
                        base = os.path.basename(f)
                        det_name = base.replace("online_results_00001_", "").replace(f"_{folder_name}_origdata", "").replace(".csv", "")
                        
                        score_col_source = None
                        if 'anomaly_score' in det_df.columns: score_col_source = 'anomaly_score'
                        elif 'score' in det_df.columns: score_col_source = 'score'
                        elif 'value' in det_df.columns: score_col_source = 'value'
                        
                        if score_col_source:
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
                            
                            detector_score_cols.append(unique_score_col)
                    except: pass
                
                if detector_score_cols:
                    df[detector_score_cols] = df[detector_score_cols].fillna(0)

                # 3. Ground Truth Prep
                gt_timestamps = get_labels_for_folder(folder_name, labels_map)
                gt_indices = []
                aws = 0
                anomaly_windows = []
                
                if gt_timestamps:
                    gt_dt = pd.to_datetime(gt_timestamps)
                    is_gt = df['dt_object'].isin(gt_dt)
                    gt_indices = df.index[is_gt].tolist()
                    
                    # Pre-calculate Window Size (AWS) and Window Array
                    # Since dataset length and GT count are constant, this is constant
                    aws = calculate_aws(ANOMALY_WINDOW_SIZE, ANOMALY_WINDOW_SIZE == 'NAB', len(df), len(gt_indices))
                    anomaly_windows = create_anomaly_windows(gt_indices, aws, len(df))

                # Store essential data for optimization
                loaded_datasets.append({
                    'name': folder_name,
                    'df': df, # Contains timestamps, values, detector scores
                    'det_cols': detector_score_cols,
                    'gt_indices': gt_indices,
                    'aws': aws,
                    'anomaly_windows': anomaly_windows, # Pre-calculated windows
                    'has_gt': len(gt_indices) > 0
                })

    print(f"Loaded {len(loaded_datasets)} datasets.")
    return loaded_datasets

# --- OPTUNA OBJECTIVE ---

def objective(trial, datasets):
    # 1. Suggest Hyperparameters
    det_score_thresh = trial.suggest_float('DETECTOR_SCORE_THRESH', 0.5, 0.999)
    min_votes = trial.suggest_int('MIN_VOTES_FOR_ALARM', 1, 5) 
    rolling_window = trial.suggest_int('ROLLING_WINDOW', 24, 500)
    pattern_std_mult = trial.suggest_float('PATTERN_STD_MULT', 1.5, 6.0)

    f1_scores = []

    for data in datasets:
        if not data['has_gt']: continue

        df = data['df'] 
        det_cols = data['det_cols']
        
        # A. Detector Logic
        # Note: We must clamp min_votes to the actual number of detectors available
        actual_min_votes = min(min_votes, len(det_cols)) if det_cols else 1
        
        if det_cols:
            # Fast vectorized calculation
            votes = (df[det_cols] > det_score_thresh).sum(axis=1)
            is_det_anom = (votes >= actual_min_votes).to_numpy()
        else:
            is_det_anom = np.zeros(len(df), dtype=bool)

        # B. Pattern Logic
        # We perform rolling calculation here as window size changes
        # Rolling can be slow, but it's optimized in pandas C backend
        expected = df['orig_value'].rolling(window=rolling_window, center=True, min_periods=1).median()
        deviation = np.abs(df['orig_value'] - expected)
        # Handle case where deviation is all 0 or NaN
        dev_mean = deviation.mean()
        dev_std = deviation.std()
        
        thresh = dev_mean + (pattern_std_mult * dev_std)
        is_pat_anom = (deviation > thresh).to_numpy()

        # C. Combine
        final_anom = is_det_anom | is_pat_anom
        
        # D. Metrics
        # Adjust PosEdge (Lavin)
        # Manual posedge logic on numpy array for speed
        # True if current is True and prev was False
        final_anom_posedge = np.zeros_like(final_anom, dtype=bool)
        final_anom_posedge[0] = final_anom[0]
        final_anom_posedge[1:] = final_anom[1:] & (~final_anom[:-1])

        tp, fp, tn, fn = measure_confusion_matrix(final_anom_posedge, data['anomaly_windows'], data['aws'], NORMALIZE_METRICS)
        
        _, _, f1, _ = calculate_metrics(tp, fp, tn, fn)
        f1_scores.append(f1)

    # Return mean F1 score (maximize this)
    if not f1_scores: return 0.0
    return np.mean(f1_scores)

# --- VISUALIZATION (FINAL RUN) ---

def run_final_analysis(datasets, best_params, output_base):
    print("\n--- Generating Final Plots with Best Parameters ---")
    
    # Extract params
    det_score_thresh = best_params['DETECTOR_SCORE_THRESH']
    min_votes = best_params['MIN_VOTES_FOR_ALARM']
    rolling_window = best_params['ROLLING_WINDOW']
    pattern_std_mult = best_params['PATTERN_STD_MULT']

    for data in datasets:
        df = data['df'].copy() # Copy so we don't mess up original for any reason
        det_cols = data['det_cols']
        folder_name = data['name']
        aws = data['aws']
        gt_indices = data['gt_indices']

        # Logic
        actual_min_votes = min(min_votes, len(det_cols)) if det_cols else 1
        
        # Detectors
        det_mask = np.zeros(len(df), dtype=bool)
        if det_cols:
             votes = (df[det_cols] > det_score_thresh).sum(axis=1)
             det_mask = (votes >= actual_min_votes).to_numpy()
        
        # Pattern
        df['expected_pattern'] = df['orig_value'].rolling(window=rolling_window, center=True, min_periods=1).median()
        df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
        thresh = df['pattern_deviation'].mean() + (pattern_std_mult * df['pattern_deviation'].std())
        pat_mask = (df['pattern_deviation'] > thresh).to_numpy()

        # Final
        df['is_final_anomaly'] = (det_mask | pat_mask).astype(int)
        
        # Metrics
        final_detections = adjust_anomaly_signals_posedge(df['is_final_anomaly'].astype(bool).tolist(), ADJUST_ANOMALIES_POSEDGE)
        
        prec, rec, f1, mcc = 0,0,0,0
        if data['has_gt']:
            tp, fp, tn, fn = measure_confusion_matrix(final_detections, data['anomaly_windows'], aws, NORMALIZE_METRICS)
            prec, rec, f1, mcc = calculate_metrics(tp, fp, tn, fn)

        # Plot
        det_idx = np.where(det_mask)[0]
        pat_idx = np.where(pat_mask)[0]
        total_combined = df['is_final_anomaly'].sum()

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_title(f'Analysis: {folder_name}\nLavin/Ahmad (AW={aws}) | F1={f1:.3f} | Total: {total_combined}')
        
        indices = df.index
        ax.plot(indices, df['orig_value'], color='#1f77b4', alpha=0.6, label='Value')
        ax.plot(indices, df['expected_pattern'], color='black', linestyle='--', alpha=0.5, label='Trend')

        if len(det_idx) > 0:
            ax.scatter(det_idx, df.loc[det_idx, 'orig_value'], c='black', marker='x', s=50, 
                       label=f'Detector (n={len(det_idx)})', zorder=8)
        
        if len(pat_idx) > 0:
            ax.scatter(pat_idx, df.loc[pat_idx, 'orig_value'], c='red', marker='o', s=20, 
                       label=f'Pattern (n={len(pat_idx)})', zorder=9)

        if len(gt_indices) > 0:
            ax.scatter(gt_indices, df.loc[gt_indices, 'orig_value'], c='green', marker='*', s=200, label='Ground Truth', zorder=7)
            for i in gt_indices:
                start_w = max(0, i - aws)
                end_w = min(len(df), i + aws)
                ax.axvspan(start_w, end_w, color='green', alpha=0.1) 
                ax.axvline(x=i, color='green', alpha=0.3)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")
        
        step = max(1, len(df) // 12)
        ax.set_xticks(indices[::step])
        ax.set_xticklabels(indices[::step])
        ax.set_xlabel("Time Step (Index)")

        out_plot = os.path.join(output_base, "plots")
        os.makedirs(out_plot, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_plot, f"plot_{folder_name}_optimized.png"))
        print(f"Saved plot for {folder_name}")
        plt.show()
        plt.close()

        # Save CSV
        out_csv = os.path.join(output_base, "combined_csv")
        os.makedirs(out_csv, exist_ok=True)
        df.to_csv(os.path.join(out_csv, f"{folder_name}_optimized.csv"), index=False)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results_optimized")
    
    # 1. LOAD DATA ONCE
    datasets = preload_data(base_dir)

    if not datasets:
        print("No valid data found.")
        exit()

    # 2. RUN OPTUNA
    print("\n--- Starting Optuna Optimization (500 trials) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, datasets), n_trials=500, show_progress_bar=True)

    print("\n--- Optimization Finished ---")
    print("Best F-Score (Avg):", study.best_value)
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 3. RUN FINAL VISUALIZATION
    run_final_analysis(datasets, study.best_params, results_dir)
    print(f"\nDone. Optimized results saved to: {results_dir}")