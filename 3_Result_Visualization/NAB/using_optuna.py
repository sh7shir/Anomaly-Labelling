import pandas as pd
import numpy as np
import glob
import os
import json
import optuna
import sys

# --- CONFIGURATION ---
EVALUATION_WINDOW = 100  # Fixed Metric Parameter

# --- HELPER FUNCTIONS ---

def load_ground_truth(base_dir):
    label_file = os.path.join(base_dir, "combined_labels.json")
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load labels: {e}")
            pass
    return {}

def get_labels_for_folder(folder_name, all_labels):
    if not all_labels: return []
    for filename, timestamps in all_labels.items():
        if folder_name in filename:
            return timestamps
    return []

def get_continuous_events(binary_array):
    """
    Fast numpy-based event finder.
    """
    binary_array = np.nan_to_num(binary_array).astype(int) # Safety: ensure 0/1 int
    if binary_array.sum() == 0:
        return []
    
    # Pad with 0 to detect start/end correctly
    diff = np.diff(np.concatenate(([0], binary_array, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    events = []
    for s, e in zip(starts, ends):
        events.append((s, e - 1))
    return events

def calculate_f1_numpy(pred_binary, gt_indices, window_size=EVALUATION_WINDOW):
    """
    Calculates F1 score using pre-calculated binary array and GT indices.
    """
    total_gt = len(gt_indices)
    pred_events = get_continuous_events(pred_binary)
    total_pred = len(pred_events)

    if total_gt == 0:
        return 0.0

    tp_gt = 0
    used_pred_events = set()

    for gt_idx in gt_indices:
        start_w = gt_idx - window_size
        end_w = gt_idx + window_size
        
        found = False
        for i, (p_start, p_end) in enumerate(pred_events):
            # Check overlap
            if p_start <= end_w and start_w <= p_end:
                found = True
                used_pred_events.add(i)
        
        if found:
            tp_gt += 1

    recall = tp_gt / total_gt
    tp_pred = len(used_pred_events)
    precision = tp_pred / total_pred if total_pred > 0 else 0.0
    
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# --- DATASET CLASS ---

class Dataset:
    def __init__(self, name, orig_values, detector_scores, gt_indices):
        self.name = name
        self.orig_values = orig_values       # Numpy Array (Float)
        self.detector_scores = detector_scores # Numpy Matrix (Float)
        self.gt_indices = gt_indices         # List of Integers

def preload_data(base_dir):
    datasets = []
    labels_map = load_ground_truth(base_dir)
    
    print(f"Scanning: {base_dir}")
    
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, "orig")):
                folder_name = entry.name
                
                # 1. Load Master File
                orig_files = glob.glob(os.path.join(entry.path, "orig", "*.csv"))
                if not orig_files: continue
                
                try:
                    df = pd.read_csv(orig_files[0])
                    
                    # Normalize columns
                    cols = df.columns
                    if 'timestamp' in cols: df.rename(columns={'timestamp': 'timestep'}, inplace=True)
                    if 'value' in cols: df.rename(columns={'value': 'orig_value'}, inplace=True)
                    
                    if 'timestep' not in df.columns or 'orig_value' not in df.columns:
                        continue

                    df['dt_object'] = pd.to_datetime(df['timestep'])
                    df = df.sort_values(by='dt_object').reset_index(drop=True)
                    
                    # Pre-fill NaNs in value
                    df['orig_value'] = df['orig_value'].interpolate(limit_direction='both').fillna(0)
                    
                except Exception as e:
                    print(f"  [Error] Failed loading {folder_name}: {e}")
                    continue

                # 2. Merge Detector Results (Robustly)
                detector_files = glob.glob(os.path.join(entry.path, "*.csv"))
                score_col_names = []
                
                for f in detector_files:
                    try:
                        if 'orig' in f: continue
                        
                        det_df = pd.read_csv(f)
                        
                        # Identify score column
                        score_col = None
                        for c in ['anomaly_score', 'score', 'value']:
                            if c in det_df.columns:
                                score_col = c
                                break
                        
                        if score_col is None: continue
                        
                        det_name = os.path.basename(f).replace(".csv", "")
                        unique_col = f'score_{det_name}'
                        
                        # Prepare for merge
                        det_df.rename(columns={score_col: unique_col}, inplace=True)
                        
                        # Fix: Ensure timestamps exist
                        if 'timestamp' in det_df.columns:
                            det_df['dt_object'] = pd.to_datetime(det_df['timestamp'])
                        else:
                            # If no timestamp, assume index match if lengths match
                            if len(det_df) == len(df):
                                df[unique_col] = det_df[unique_col].values
                                score_col_names.append(unique_col)
                                continue
                            else:
                                continue # Cannot merge without timestamp or exact length

                        # Fix: Drop duplicates in detector file to prevent merge explosion
                        det_df = det_df.drop_duplicates(subset=['dt_object'])
                        
                        # Fix: Safe Left Join
                        df = pd.merge(df, det_df[['dt_object', unique_col]], on='dt_object', how='left')
                        score_col_names.append(unique_col)
                        
                    except Exception as e:
                        # print(f"    Skipping detector {f}: {e}")
                        pass
                
                # Fill missing scores with 0
                if score_col_names:
                    df[score_col_names] = df[score_col_names].fillna(0)
                    det_scores_matrix = df[score_col_names].values
                else:
                    det_scores_matrix = np.zeros((len(df), 0))

                # 3. Get GT Indices
                gt_timestamps = get_labels_for_folder(folder_name, labels_map)
                gt_indices = []
                if gt_timestamps:
                    gt_dt = pd.to_datetime(gt_timestamps)
                    # Use isin to find indices
                    gt_indices = df.index[df['dt_object'].isin(gt_dt)].tolist()

                # Save purely numerical data to Dataset object for speed
                datasets.append(Dataset(
                    name=folder_name,
                    orig_values=df['orig_value'].values,
                    detector_scores=det_scores_matrix,
                    gt_indices=gt_indices
                ))
                
                print(f"  Loaded {folder_name}: {det_scores_matrix.shape[1]} detectors, {len(gt_indices)} GT labels")

    return datasets

# --- OPTIMIZATION LOGIC ---

def objective(trial, datasets):
    # Hyperparameters
    det_thresh = trial.suggest_float('DETECTOR_SCORE_THRESH', 0.5, 0.999)
    min_votes = trial.suggest_int('MIN_VOTES_FOR_ALARM', 1, 5)
    roll_win = trial.suggest_int('ROLLING_WINDOW', 50, 1000, step=10)
    pat_std = trial.suggest_float('PATTERN_STD_MULT', 1.0, 5.0)

    f1_scores = []
    
    for ds in datasets:
        if not ds.gt_indices:
            continue
            
        try:
            # 1. Detector Logic (Vectorized Numpy)
            if ds.detector_scores.shape[1] > 0:
                votes = (ds.detector_scores > det_thresh).sum(axis=1)
                # Cap required votes by available detectors
                req_votes = min(min_votes, ds.detector_scores.shape[1])
                is_det = (votes >= req_votes).astype(int)
            else:
                is_det = np.zeros(len(ds.orig_values), dtype=int)

            # 2. Pattern Logic (Vectorized Numpy)
            # Create a Pandas Series temporarily only for the rolling calculation which is complex in pure numpy
            # (Creating a series is fast enough compared to a full DF copy)
            vals_series = pd.Series(ds.orig_values)
            expected = vals_series.rolling(window=roll_win, center=True, min_periods=1).median().values
            
            deviation = np.abs(ds.orig_values - expected)
            
            # Avoid NaN issues in stats
            valid_dev = deviation[~np.isnan(deviation)]
            if len(valid_dev) == 0:
                is_pat = np.zeros(len(ds.orig_values), dtype=int)
            else:
                thresh = np.mean(valid_dev) + (pat_std * np.std(valid_dev))
                is_pat = (deviation > thresh).astype(int)

            # 3. Combine
            final_anomalies = (is_det | is_pat).astype(int)
            
            # 4. Metric
            f1 = calculate_f1_numpy(final_anomalies, ds.gt_indices)
            f1_scores.append(f1)
            
        except Exception as e:
            # print(f"Trial failed on {ds.name}: {e}")
            return 0.0

    if not f1_scores:
        return 0.0
        
    return np.mean(f1_scores)

# --- EXECUTION ---

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Data
    all_datasets = preload_data(base_dir)
    
    if not all_datasets:
        print("No valid datasets found (Check 'orig' folders).")
        sys.exit()

    # 2. Optimize
    print("\n--- Starting Optuna ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce noise
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, all_datasets), n_trials=10000)

    # 3. Report
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best Avg F1 Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # 4. Visualization
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        print("\nAttempting to display plots...")
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

    except ImportError:
        print("\n(Install 'plotly' to see interactive Optuna plots)")
    except Exception as e:
        print(f"\nCould not display plots: {e}")