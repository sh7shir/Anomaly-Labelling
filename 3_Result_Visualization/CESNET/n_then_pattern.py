import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D   # NEW: for custom legend handles

# --- CONFIGURATION ---
DETECTOR_SCORE_THRESH = 0.99   
MIN_VOTES_FOR_ALARM = 3
ROLLING_WINDOW = 24  
PATTERN_STD_MULT = 4

def load_ground_truth(base_dir):
    label_file = os.path.join(base_dir, "combined_labels.json")
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def load_global_time(base_dir):
    """Loads the global times_1_hour.csv file explicitly looking for 'time' column."""
    time_file = os.path.join(base_dir, "times_1_hour.csv")
    if not os.path.exists(time_file):
        print("[Error] times_1_hour.csv not found in root.")
        return None
    
    try:
        df_time = pd.read_csv(time_file)
        if 'time' not in df_time.columns:
            print(f"[Error] Column 'time' not found in {time_file}.")
            return None
            
        df_time['dt_object'] = pd.to_datetime(df_time['time'])
        return df_time
    except Exception as e:
        print(f"[Error] Failed to load global time file: {e}")
        return None

def load_weekends(base_dir):
    """
    Loads weekends_and_holidays.csv.
    Returns a dictionary: {datetime.date: 'Weekend' or 'Holiday'}
    """
    fpath = os.path.join(base_dir, "weekends_and_holidays.csv")
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            # Normalize columns
            df.columns = [c.strip() for c in df.columns]
            
            if 'Date' in df.columns and 'Type' in df.columns:
                df['dt'] = pd.to_datetime(df['Date']).dt.date
                return pd.Series(df.Type.values, index=df.dt).to_dict()
        except Exception as e:
            print(f"[Warning] Could not parse weekends_and_holidays.csv: {e}")
    return {}

def get_labels_for_folder(folder_name, sub_metric, all_labels):
    if not all_labels:
        return []
    matches = []
    for filename, timestamps in all_labels.items():
        if folder_name in filename and sub_metric in filename:
            matches.extend(timestamps)
    return matches

def analyze_metric_folder(folder_path, dataset_name, metric_name,
                          output_base, ground_truth_timestamps,
                          global_time_df, holiday_map):
    print(f"\nProcessing: {dataset_name} -> {metric_name}...")
    
    detector_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not detector_files:
        print(f"   [!] No CSV files found in {metric_name}")
        return

    # --- STEP 1: Construct Master Data ---
    try:
        df = global_time_df.copy()
        
        # Read first detector file for 'value'
        ref_df = pd.read_csv(detector_files[0])
        
        val_col = None
        if 'value' in ref_df.columns:
            val_col = 'value'
        elif 'orig_value' in ref_df.columns:
            val_col = 'orig_value'
        
        if val_col:
            if len(ref_df) != len(df):
                print(f"   [Warning] Length mismatch. Aligning by index.")
            ref_vals = ref_df[val_col].reset_index(drop=True)
            df['orig_value'] = ref_vals.reindex(df.index).values
        else:
            print("   [Error] Could not find 'value' column.")
            return

        df = df.sort_values(by='dt_object').reset_index(drop=True)
        
        # Map Dates to Types
        df['date_only'] = df['dt_object'].dt.date
        
    except Exception as e:
        print(f"   [Error] Master dataframe error: {e}")
        return

    # --- STEP 2: Merge Detector Results ---
    detector_score_cols = []
    for f in detector_files:
        try:
            det_df = pd.read_csv(f)
            
            if 'algorithm' in det_df.columns and pd.notna(det_df['algorithm'].iloc[0]):
                det_name = str(det_df['algorithm'].iloc[0])
            else:
                base = os.path.basename(f)
                clean_name = (
                    base.replace("online_results_00001_", "")
                        .replace(f"_{metric_name}_origdata", "")
                        .replace(".csv", "")
                )
                det_name = clean_name
            
            score_col_source = None
            if 'anomaly_score' in det_df.columns:
                score_col_source = 'anomaly_score'
            elif 'score' in det_df.columns:
                score_col_source = 'score'
            
            if score_col_source is None:
                continue
                
            unique_score_col = f'score_{det_name}'
            det_df.rename(columns={score_col_source: unique_score_col}, inplace=True)
            
            if len(det_df) != len(df):
                det_df = det_df.reset_index(drop=True).reindex(df.index)
            
            df[unique_score_col] = det_df[unique_score_col]
            detector_score_cols.append(unique_score_col)

        except Exception as e:
            print(f"      [Error] file {os.path.basename(f)}: {e}")

    if detector_score_cols:
        df[detector_score_cols] = df[detector_score_cols].fillna(0)

    # --- STEP 3: Calculations ---
    # Detector voting
    if detector_score_cols:
        df['detector_votes'] = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).sum(axis=1)
        df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)
    else:
        df['is_detector_anomaly'] = 0

    # Pattern-based anomalies
    if df['orig_value'].isnull().all():
        df['expected_pattern'] = 0
        df['is_pattern_anomaly'] = 0
    else:
        df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
        df['expected_pattern'] = df['orig_value'].rolling(
            window=ROLLING_WINDOW, center=True, min_periods=1
        ).median()
        df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
        threshold = df['pattern_deviation'].mean() + (
            PATTERN_STD_MULT * df['pattern_deviation'].std()
        )
        df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # Ground truth
    gt_count = 0
    df['is_ground_truth'] = False
    if ground_truth_timestamps:
        gt_dt = pd.to_datetime(ground_truth_timestamps)
        df['is_ground_truth'] = df['dt_object'].isin(gt_dt)
        gt_count = df['is_ground_truth'].sum()

    print(
        f"   -> Alerts: Detector={df['is_detector_anomaly'].sum()} | "
        f"Pattern={df['is_pattern_anomaly'].sum()} | GT Matches={gt_count}"
    )

    # --- STEP 4: Save & Visualize ---
    out_dir = os.path.join(output_base, dataset_name, metric_name)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "combined_analysis.csv"), index=False)

    if 'orig_value' in df.columns:
        # Set global font sizes
        plt.rcParams.update({
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 20,
        })

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.set_title(f'Analysis: {dataset_name} - {metric_name}')
        
        indices = df.index
        
        # 1. SHADE WEEKENDS & HOLIDAYS (Background)
        unique_dates = df['date_only'].unique()
        has_weekend = False
        has_holiday = False
        
        for d in unique_dates:
            d_type = holiday_map.get(d)
            if d_type in ['Weekend', 'Holiday']:
                d_indices = df.index[df['date_only'] == d]
                if len(d_indices) > 0:
                    start_idx = d_indices[0]
                    end_idx = d_indices[-1] + 1 
                    
                    if d_type == 'Weekend':
                        color = 'lightgreen'
                        has_weekend = True
                    else:
                        color = 'lightpink'
                        has_holiday = True
                        
                    ax.axvspan(start_idx, end_idx, color=color,
                               alpha=0.3, zorder=0, lw=0)

        # 2. PLOT DATA (Foreground)
        ax.plot(indices, df['orig_value'],
                color='#1f77b4', alpha=0.8,
                label='Value', zorder=2)
        ax.plot(indices, df['expected_pattern'],
                color='black', linestyle='--', alpha=0.5,
                label='Trend', zorder=2)

        # 3. ALERTS & GT
        det_idx = df.index[df['is_detector_anomaly'] == 1]
        if len(det_idx) > 0:
            ax.scatter(
                det_idx,
                df.loc[det_idx, 'orig_value'],
                c='black', marker='x', s=70,
                label=f'Detector({len(det_idx)})',
                zorder=5
            )

        pat_idx = df.index[df['is_pattern_anomaly'] == 1]
        if len(pat_idx) > 0:
            ax.scatter(
                pat_idx,
                df.loc[pat_idx, 'orig_value'],
                c='red', marker='o', s=60,
                label=f'Pattern Break({len(pat_idx)})',
                zorder=6
            )

        if gt_count > 0:
            gt_idx = df.index[df['is_ground_truth'] == 1]
            ax.scatter(
                gt_idx,
                df.loc[gt_idx, 'orig_value'],
                c='green', marker='*', s=250,
                label=f'Ground Truth({gt_count})',
                zorder=4
            )
            for i in gt_idx:
                ax.axvline(x=i, color='green', alpha=0.3, zorder=3)

        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_ylabel("Value")

        # --- CUSTOM LEGEND ON THE RIGHT (SEPARATE LABELS) ---
        legend_items = []

        # Value line
        legend_items.append((
            Line2D([0], [0], color='#1f77b4', linewidth=3),
            'Value'
        ))

        # Trend line
        legend_items.append((
            Line2D([0], [0], color='black', linestyle='--', linewidth=2),
            'Trend'
        ))

        # Detector points
        if len(det_idx) > 0:
            legend_items.append((
                Line2D([0], [0], marker='x', color='black',
                       markersize=10, linewidth=0),
                f"Detector ({len(det_idx)})"
            ))

        # Pattern points
        if len(pat_idx) > 0:
            legend_items.append((
                Line2D([0], [0], marker='o', color='red',
                       markersize=10, linewidth=0),
                f"Pattern Break ({len(pat_idx)})"
            ))

        # Ground truth
        if gt_count > 0:
            legend_items.append((
                Line2D([0], [0], marker='*', color='green',
                       markersize=14, linewidth=0),
                f"Ground Truth ({gt_count})"
            ))

        # Weekend / holiday shading
        if has_weekend:
            legend_items.append((
                Patch(facecolor='lightgreen', alpha=0.3),
                "Weekend"
            ))
        if has_holiday:
            legend_items.append((
                Patch(facecolor='lightpink', alpha=0.3),
                "Holiday"
            ))

        ax.legend(
            [h for h, _ in legend_items],
            [t for _, t in legend_items],
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=20,
            frameon=True
        )

        # X-TICKS
        step = max(1, len(df) // 12)
        ax.set_xticks(indices[::step])
        ax.set_xticklabels(
            df['dt_object'].dt.strftime('%Y-%m-%d').iloc[::step],
            rotation=45, ha='right'
        )

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot.png"), bbox_inches="tight")
        plt.show()
        plt.close()

# --- EXECUTION ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    
    print(f"1. Loading global configuration from {base_dir}...")
    global_time_df = load_global_time(base_dir)
    labels_map = load_ground_truth(base_dir)
    holiday_map = load_weekends(base_dir)

    if global_time_df is None:
        print("CRITICAL: Cannot proceed without global time file.")
        exit()

    print(f"2. Scanning for detection result folders (detec_res_*)...")
    
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("detec_res_"):
                dataset_name = entry.name
                with os.scandir(entry.path) as sub_entries:
                    for sub in sub_entries:
                        if sub.is_dir():
                            metric_name = sub.name 
                            current_labels = get_labels_for_folder(
                                dataset_name, metric_name, labels_map
                            )
                            
                            analyze_metric_folder(
                                sub.path, 
                                dataset_name, 
                                metric_name, 
                                results_dir, 
                                current_labels,
                                global_time_df,
                                holiday_map
                            )

    print(f"\nDone. Results saved to: {results_dir}")
