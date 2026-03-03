import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURATION ---
FOLDERS = ["n_packets", "n_flows", "n_bytes"]

# 1. Detector Vote Settings
DETECTOR_SCORE_THRESH = 0.90  
MIN_VOTES_FOR_ALARM = 3       

# 2. Pattern Analysis Settings
ROLLING_WINDOW = 24           
PATTERN_STD_MULT = 3.0        

def load_holidays(base_dir):
    """
    Looks for weekends_and_holidays.csv in the base directory.
    Returns a set of dates (strings YYYY-MM-DD) if found, else empty set.
    """
    holiday_file = os.path.join(base_dir, "weekends_and_holidays.csv")
    holiday_dates = set()
    
    if os.path.exists(holiday_file):
        print(f" -> Found holiday file: {holiday_file}")
        try:
            df_h = pd.read_csv(holiday_file)
            if 'Date' in df_h.columns:
                # Normalize to string YYYY-MM-DD
                holiday_dates = set(pd.to_datetime(df_h['Date']).dt.strftime('%Y-%m-%d'))
            print(f"    Loaded {len(holiday_dates)} holiday/weekend entries.")
        except Exception as e:
            print(f"    Warning: Could not parse holiday file. {e}")
    else:
        print(" -> No 'weekends_and_holidays.csv' found. Skipping holiday overlay.")
        
    return holiday_dates

def analyze_folder_unified(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder_name)
    
    print(f"Processing Unified Analysis: {folder_name}...")

    # --- STEP 1: Load All Columns into Single DataFrame ---
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print(f"No files found in {folder_name}")
        return

    df = None
    detector_score_cols = []

    print(f"   -> Merging data from {len(all_files)} files into one DataFrame...")

    for f in all_files:
        try:
            # Load raw data
            temp_df = pd.read_csv(f)
            
            # Identify detector name
            if 'algorithm' in temp_df.columns:
                det_name = str(temp_df['algorithm'].iloc[0])
            else:
                det_name = os.path.basename(f).split('.')[0]
                # print(f"      Note: 'algorithm' column missing in {os.path.basename(f)}, using filename '{det_name}'")
            
            # Rename score column
            score_col_name = f'anomaly_score_{det_name}'
            
            if 'anomaly_score' in temp_df.columns:
                temp_df.rename(columns={'anomaly_score': score_col_name}, inplace=True)
            else:
                # If input is raw data (timestamp, value) without scores, skip or handle differently
                # Assuming standard input format as per original code
                print(f"      Warning: 'anomaly_score' missing in file {f}")
                continue

            # Identify columns to merge
            cols_to_use = ['timestep', score_col_name]
            
            if df is None:
                # Initialize master DF
                initial_cols = ['timestep', score_col_name]
                if 'orig_value' in temp_df.columns: initial_cols.append('orig_value')
                if 'dataset' in temp_df.columns: initial_cols.append('dataset')
                
                df = temp_df[initial_cols].copy()
            else:
                # Merge subsequent files
                if 'timestep' in temp_df.columns:
                    df = pd.merge(df, temp_df[cols_to_use], on='timestep', how='outer')
                else:
                    print(f"      Error: 'timestep' missing in {f}, cannot merge.")
                    continue

            detector_score_cols.append(score_col_name)

        except Exception as e:
            print(f"      Warning: Could not process {f}: {e}")

    # Ensure rows are sorted by timestep
    if df is not None:
        df = df.sort_values(by='timestep').reset_index(drop=True)
        # Convert timestep to datetime objects for handling holidays/plotting
        df['dt_object'] = pd.to_datetime(df['timestep'], errors='coerce')
    else:
        print("Error: DataFrame could not be created.")
        return

    # --- STEP 2: Calculate Detector Votes ---
    print("   -> Calculating Consensus...")
    
    if not detector_score_cols:
        print("Error: No valid detector score columns found.")
        return

    votes_df = (df[detector_score_cols] > DETECTOR_SCORE_THRESH).astype(int)
    df['detector_votes'] = votes_df.sum(axis=1)
    df['is_detector_anomaly'] = (df['detector_votes'] >= MIN_VOTES_FOR_ALARM).astype(int)

    # --- STEP 3: Pattern Recognition (Rolling Median) ---
    df['orig_value'] = df['orig_value'].interpolate(limit_direction='both')
    df['expected_pattern'] = df['orig_value'].rolling(window=ROLLING_WINDOW, center=True, min_periods=1).median()
    df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
    
    dev_mean = df['pattern_deviation'].mean()
    dev_std = df['pattern_deviation'].std()
    threshold = dev_mean + (PATTERN_STD_MULT * dev_std)
    
    df['is_pattern_anomaly'] = (df['pattern_deviation'] > threshold).astype(int)

    # --- NEW STEP: Holiday/Weekend Check ---
    holiday_dates = load_holidays(base_dir)
    df['is_holiday'] = False
    if holiday_dates:
        # Check if the date part of the timestamp is in the holiday set
        df['date_str'] = df['dt_object'].dt.strftime('%Y-%m-%d')
        df['is_holiday'] = df['date_str'].isin(holiday_dates)

    # --- NEW STEP: Count Anomalies ---
    count_detector = df['is_detector_anomaly'].sum()
    count_pattern = df['is_pattern_anomaly'].sum()
    count_holidays = df[df['is_holiday'] == True]['is_pattern_anomaly'].sum() # Example: pattern anomalies on holidays

    print("\n" + "="*40)
    print(f"ANOMALY REPORT: {folder_name}")
    print(f"Total Rows Processed: {len(df)}")
    print("-" * 40)
    print(f"Detector Anomalies (Votes >= {MIN_VOTES_FOR_ALARM}): {count_detector}")
    print(f"Pattern Anomalies (Deviation > {PATTERN_STD_MULT} std): {count_pattern}")
    if holiday_dates:
        print(f"Pattern Anomalies occurring on Holidays/Weekends: {count_holidays}")
    print("="*40 + "\n")

    # --- STEP 4: Save CSV ---
    out_dir_csv = os.path.join(base_dir, "combined_unified")
    os.makedirs(out_dir_csv, exist_ok=True)
    df.to_csv(os.path.join(out_dir_csv, f"{folder_name}_unified.csv"), index=False)

    # --- STEP 5: Visualization ---
    fig, ax1 = plt.subplots(figsize=(18, 8))
    fig.suptitle(f'Unified Anomaly Analysis: {folder_name}', fontsize=16)

    ax1.set_title(f"Detector Votes vs. Pattern Breaks\n(Detector Count: {count_detector} | Pattern Count: {count_pattern})")
    
    x_indices = df.index 
    
    # 5a. Highlight Holidays/Weekends (Shaded Background)
    if holiday_dates:
        # We need to find continuous ranges of holidays to shade nicely
        # Simple approach: shade vertical span for every row that is a holiday
        # Using fill_between is more efficient than iterating individual bars
        ax1.fill_between(x_indices, df['orig_value'].min(), df['orig_value'].max(), 
                         where=df['is_holiday'], color='gray', alpha=0.2, label='Weekend/Holiday', zorder=0)

    # 5b. Main Data
    ax1.plot(x_indices, df['orig_value'], color='#1f77b4', alpha=0.6, label='Original Value')
    ax1.plot(x_indices, df['expected_pattern'], color='black', linestyle='--', alpha=0.5, label='Trend')

    # 5c. Detector Anomalies
    det_indices = df.index[df['is_detector_anomaly'] == 1]
    det_values = df.loc[det_indices, 'orig_value']
    ax1.scatter(det_indices, det_values, 
                color='black', marker='x', s=30, zorder=6, label=f'Detector Alert ({count_detector})')

    # 5d. Pattern Anomalies
    pat_indices = df.index[df['is_pattern_anomaly'] == 1]
    pat_values = df.loc[pat_indices, 'orig_value']
    ax1.scatter(pat_indices, pat_values, 
                color='red', marker='o', s=20, zorder=5, label=f'Pattern Break ({count_pattern})')

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Custom X-Ticks
    step_size = max(1, len(df) // 12)
    ax1.set_xticks(x_indices[::step_size])
    ax1.set_xticklabels(df['timestep'].iloc[::step_size], rotation=45, ha='right')
    
    # Save & Show
    out_dir_plot = os.path.join(base_dir, "plots_unified")
    os.makedirs(out_dir_plot, exist_ok=True)
    plot_path = os.path.join(out_dir_plot, f"unified_{folder_name}.png")
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"-> Saved unified analysis to {plot_path}")
    
    plt.show()
    plt.close()
    print("-" * 50)

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure this list matches your actual folder names
    for folder in FOLDERS:
        if os.path.exists(os.path.join(os.path.dirname(__file__), folder)):
            analyze_folder_unified(folder)
        else:
            print(f"Folder '{folder}' not found. Please create it and add 'det_res' CSV files.")