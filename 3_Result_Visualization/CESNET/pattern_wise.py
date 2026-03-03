import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# --- HELPER: PATTERN DETECTION ALGORITHM ---
def compute_pattern_anomalies(df_with_time):
    """
    Applies seasonal pattern deviation logic.
    Requires a DataFrame with 'orig_value' and a Datetime Index.
    """
    temp_df = df_with_time.copy()
    
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        return pd.Series(0, index=temp_df.index)

    # 1. Define Periodicity
    temp_df['time_of_day'] = temp_df.index.time
    
    # 2. Learn Expected Pattern (Median)
    pattern_template = temp_df.groupby('time_of_day')['orig_value'].median()
    temp_df['expected_pattern'] = temp_df['time_of_day'].map(pattern_template)
    
    # 3. Calculate Deviation
    temp_df['pattern_deviation'] = np.abs(temp_df['orig_value'] - temp_df['expected_pattern'])
    
    # 4. Smooth Deviation (Window ~ 2 hours)
    rolling_window = 12 
    temp_df['smoothed_deviation'] = temp_df['pattern_deviation'].rolling(window=rolling_window, center=True).mean().fillna(0)
    
    # 5. Threshold (Mean + 3 Std)
    dev_mean = temp_df['smoothed_deviation'].mean()
    dev_std = temp_df['smoothed_deviation'].std()
    threshold = dev_mean + (3.0 * dev_std)
    
    return (temp_df['smoothed_deviation'] > threshold).astype(int)

# --- CORE PROCESSING FUNCTION ---
def process_anomaly_files(config, folder_name, folder_path, time_file_path, holiday_file_path):
    print(f"\n{'='*60}")
    print(f"STARTING ANALYSIS FOR: {folder_name}")
    print(f"{'='*60}\n")
    
    # --- 1. Load Time Data ---
    time_df = None
    if os.path.exists(time_file_path):
        try:
            time_df = pd.read_csv(time_file_path)
            if 'id_time' in time_df.columns and 'time' in time_df.columns:
                time_df['time'] = pd.to_datetime(time_df['time'])
            else:
                print(f"Warning: {time_file_path} missing columns.")
                time_df = None
        except Exception as e:
            print(f"Error loading time file: {e}")

    # --- 2. Process Standard Detectors (CSVs) ---
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{folder_path}'. Skipping.")
        return

    anomaly_dfs = []
    plot_data_df = None
    first_file_processed = False
    
    anomaly_score_threshold = config.get('ANOMALY_SCORE_THRESHOLD', 0.95)

    print(f"Found {len(csv_files)} standard detectors.")

    for f_path in csv_files:
        try:
            filename = os.path.basename(f_path)
            df = pd.read_csv(f_path)
            
            if 'Unnamed: 0' in df.columns: df = pd.read_csv(f_path, index_col=0)
            if 'timestep' not in df.columns:
                if df.index.name == 'timestep': df.reset_index(inplace=True)
                else: continue 

            if 'orig_value' not in df.columns or 'anomaly_score' not in df.columns:
                continue

            df = df.set_index('timestep')

            if not first_file_processed:
                plot_data_df = df[['orig_value']].copy()
                first_file_processed = True

            match = re.search(r'_\d{5}_(.*)_origdata\.csv', filename)
            name_part = match.group(1) if match else filename.rsplit('.', 1)[0]
            
            col_name = f'is_anomaly_{name_part}'
            df[col_name] = (df['anomaly_score'] > anomaly_score_threshold).astype(int)
            anomaly_dfs.append(df[[col_name]])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not first_file_processed or not anomaly_dfs:
        print("Error: No valid data processed. Skipping.")
        return

    # Combine Standard Detectors
    combined_anomalies = pd.concat(anomaly_dfs, axis=1)
    final_df = plot_data_df.join(combined_anomalies, how='left')
    
    # Identify Standard Columns
    standard_cols = [col for col in final_df.columns if col.startswith('is_anomaly')]
    final_df[standard_cols] = final_df[standard_cols].fillna(0).astype(int)

    # --- 3. MERGE TIME (Required for Pattern Verification) ---
    has_time = False
    if time_df is not None:
        try:
            final_df = final_df.merge(time_df, left_index=True, right_on='id_time', how='left')
            if 'time' in final_df.columns:
                final_df = final_df.set_index('time')
                has_time = True
                print("Time data merged successfully.")
        except Exception as e:
            print(f"Time merge failed: {e}")

    # --- 4. STAGE 1: VOTING (Standard Detectors) ---
    min_votes = config.get('MIN_VOTES_FOR_ANOMALY', 2)
    final_df['standard_vote_count'] = final_df[standard_cols].sum(axis=1)
    
    # Candidates: Points that passed the vote
    final_df['is_candidate'] = (final_df['standard_vote_count'] >= min_votes).astype(int)
    candidate_count = final_df['is_candidate'].sum()
    print(f">> Stage 1 (Voting >= {min_votes}): Found {candidate_count} candidates.")

    # --- 5. STAGE 2: VERIFICATION (Pattern Algo) ---
    if has_time:
        print(">> Stage 2: Verifying candidates with Pattern Recognition...")
        final_df['pattern_anomaly_flag'] = compute_pattern_anomalies(final_df)
        
        # LOGIC: CONFIRMED = (Voted YES) AND (Pattern YES)
        final_df['final_anomaly_score'] = (final_df['is_candidate'] & final_df['pattern_anomaly_flag']).astype(int)
    else:
        print("Warning: No time data. Skipping verification stage (accepting all candidates).")
        final_df['final_anomaly_score'] = final_df['is_candidate']

    # Stats
    total = len(final_df)
    confirmed = final_df['final_anomaly_score'].sum()
    rejected = candidate_count - confirmed
    print(f">> Final Result: {confirmed} confirmed anomalies.")
    print(f">> Rejected by Pattern Verifier: {rejected}")

    # --- 6. HOLIDAY DATA ---
    if has_time and os.path.exists(holiday_file_path):
        try:
            non_work_df = pd.read_csv(holiday_file_path)
            if 'Date' in non_work_df.columns and 'Type' in non_work_df.columns:
                non_work_df['Date'] = pd.to_datetime(non_work_df['Date']).dt.normalize()
                weekend_dates = set(non_work_df[non_work_df['Type'] == 'Weekend']['Date'])
                holiday_dates = set(non_work_df[non_work_df['Type'] == 'Holiday']['Date'])
                
                final_df['normalized_date'] = final_df.index.tz_convert(None).normalize() if final_df.index.tz else final_df.index.normalize()
                final_df['is_weekend'] = final_df['normalized_date'].isin(weekend_dates).astype(int)
                final_df['is_holiday'] = final_df['normalized_date'].isin(holiday_dates).astype(int)
        except Exception: pass

    # --- SAVE RESULTS ---
    parent_dir = os.path.dirname(folder_path) 
    combined_dir = os.path.join(parent_dir, "combined")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    
    output_path = os.path.join(combined_dir, f"{folder_name}.csv")
    final_df.to_csv(output_path)
    print(f"Results saved to: {output_path}")

    # --- 7. PLOTTING ---
    print(f"Generating plot...")
    fig, ax1 = plt.subplots(figsize=(18, 8))
    fig.suptitle(f'Anomaly Verification: {folder_name}\n(Votes >= {min_votes} AND Pattern Verified)', fontsize=16)

    ax1.set_xlabel('Time' if has_time else 'Timestep')
    ax1.set_ylabel('Value', color='tab:blue')
    ax1.plot(final_df.index, final_df['orig_value'], color='tab:blue', alpha=0.7, label='Value', zorder=1)
    
    # Background Shading
    legend_patches = []
    median_diff = pd.Timedelta(hours=1) if has_time else 1
    if has_time:
         try: median_diff = final_df.index.to_series().diff().median()
         except: pass

    def add_shading(col, color, label):
        if col in final_df.columns and final_df[col].sum() > 0:
            final_df['grp'] = (final_df[col].diff() != 0).cumsum()
            for _, g in final_df[final_df[col] == 1].groupby('grp'):
                ax1.axvspan(g.index.min(), g.index.max() + median_diff, color=color, alpha=0.3, zorder=-1, lw=0)
            legend_patches.append(Rectangle((0,0),1,1, color=color, alpha=0.3, label=label))

    add_shading('is_weekend', 'lightgreen', 'Weekend')
    add_shading('is_holiday', 'lightpink', 'Holiday')

    # 1. Plot REJECTED Candidates (Orange)
    rejected_df = final_df[(final_df['is_candidate'] == 1) & (final_df['final_anomaly_score'] == 0)]
    if not rejected_df.empty:
        ax1.scatter(rejected_df.index, rejected_df['orig_value'], color='orange', label='Rejected Candidate (Voted Yes, Pattern No)', zorder=2, s=15, marker='x')

    # 2. Plot CONFIRMED Anomalies (Red)
    confirmed_df = final_df[final_df['final_anomaly_score'] == 1]
    if not confirmed_df.empty:
        ax1.scatter(confirmed_df.index, confirmed_df['orig_value'], color='red', label='Confirmed Anomaly (Voted Yes + Pattern Yes)', zorder=3, s=25)

    # Clusters (on Confirmed only)
    cluster_len = config.get('CLUSTER_RUN_LENGTH', 5)
    final_df['c_grp'] = (final_df['final_anomaly_score'].diff() != 0).cumsum()
    cluster_found = False
    
    for _, g in final_df[final_df['final_anomaly_score'] == 1].groupby('c_grp'):
        if len(g) >= cluster_len:
            cluster_found = True
            mn, mx = g['orig_value'].min(), g['orig_value'].max()
            pad = (mx - mn) * 0.1 if mx != mn else 1
            rect = Rectangle((g.index.min(), mn-pad), (g.index.max()+median_diff)-g.index.min(), (mx-mn)+(2*pad), color='grey', alpha=0.2, zorder=0)
            ax1.add_patch(rect)
            
    if cluster_found:
        legend_patches.append(Rectangle((0,0),1,1, color='grey', alpha=0.2, label=f'Cluster ({cluster_len}+)'))

    lines, labels = ax1.get_legend_handles_labels()
    lines.extend(legend_patches)
    labels.extend([p.get_label() for p in legend_patches])
    ax1.legend(lines, labels, loc='upper left')

    # Info Box
    stats_text = (f"Stage 1 Candidates: {candidate_count}\n"
                  f"Rejected by Pattern: {rejected}\n"
                  f"FINAL CONFIRMED: {confirmed}")
    
    ax1.text(0.99, 0.99, stats_text, transform=ax1.transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'), zorder=10)

    if has_time:
        fig.autofmt_xdate()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    plot_filename = f"plot_{folder_name}.png"
    plt.savefig(plot_filename)
    print(f"Saved plot to: {plot_filename}")
    
    if config.get('SHOW_PLOTS', True): 
        print(">> Displaying Plot. Close window to continue...")
        plt.show()
    else: 
        plt.close(fig)

# --- WRAPPER ---
def run_analysis_for_folder(folder_name, general_config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    process_anomaly_files(general_config, folder_name, os.path.join(base_dir, folder_name), 
                          os.path.join(base_dir, general_config["TIME_FILE"]), 
                          os.path.join(base_dir, general_config["HOLIDAY_FILE"]))

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    CONFIG = {
        "TIME_FILE": "times_1_hour.csv",
        "HOLIDAY_FILE": "weekends_and_holidays.csv",
        "ANOMALY_SCORE_THRESHOLD": 0.70,
        
        # Stage 1 Requirement
        "MIN_VOTES_FOR_ANOMALY": 2, 
        
        "CLUSTER_RUN_LENGTH": 5,
        "SHOW_PLOTS": True
    }

    FOLDERS = ["n_packets", "n_flows", "n_bytes"]

    for folder in FOLDERS:
        run_analysis_for_folder(folder, CONFIG)