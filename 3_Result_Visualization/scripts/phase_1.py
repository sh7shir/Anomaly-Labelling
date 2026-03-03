import pandas as pd
import glob
import os
import sys
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# --- CORE PROCESSING FUNCTION ---
def process_anomaly_files(config, folder_name, folder_path, time_file_path, holiday_file_path):
    """
    Processes CSV files for a specific folder.
    """
    print(f"\n{'='*60}")
    print(f"STARTING ANALYSIS FOR: {folder_name}")
    print(f"Folder Path: {folder_path}")
    print(f"{'='*60}\n")
    
    # --- 1. Load Time Data ---
    time_df = None
    if os.path.exists(time_file_path):
        try:
            time_df = pd.read_csv(time_file_path)
            if 'id_time' not in time_df.columns or 'time' not in time_df.columns:
                print(f"Warning: {time_file_path} is missing columns. Using integer timesteps.")
                time_df = None
            else:
                print(f"Loaded time data from {os.path.basename(time_file_path)}")
        except Exception as e:
            print(f"Error loading time file: {e}")
    else:
        print(f"Warning: Time file not found at {time_file_path}")

    # --- 2. Process Anomaly CSVs ---
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in '{folder_path}'. Skipping this folder.")
        return

    print(f"Found {len(csv_files)} CSV files.")

    anomaly_dfs = []
    plot_data_df = None
    first_file_processed = False
    
    anomaly_score_threshold = config.get('ANOMALY_SCORE_THRESHOLD', 0.95)

    for f_path in csv_files:
        try:
            filename = os.path.basename(f_path)
            # print(f"Processing {filename}...") # Commented out to reduce clutter in batch runs
            
            df = pd.read_csv(f_path)
            if 'Unnamed: 0' in df.columns:
                 df = pd.read_csv(f_path, index_col=0)

            if 'timestep' not in df.columns:
                if df.index.name == 'timestep':
                    df.reset_index(inplace=True)
                else:
                    continue

            if 'orig_value' not in df.columns or 'anomaly_score' not in df.columns:
                continue

            df = df.set_index('timestep')

            if not first_file_processed:
                plot_data_df = df[['orig_value']].copy()
                first_file_processed = True

            match = re.search(r'_\d{5}_(.*)_origdata\.csv', filename)
            if match:
                name_part = match.group(1)
            else:
                name_part = filename.rsplit('.', 1)[0]
            
            anomaly_col_name = f'is_anomaly_{name_part}'
            df[anomaly_col_name] = (df['anomaly_score'] > anomaly_score_threshold).astype(int)
            anomaly_dfs.append(df[[anomaly_col_name]])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not first_file_processed or not anomaly_dfs:
        print("Error: No valid data processed. Skipping.")
        return

    # Combine Data
    combined_anomalies = pd.concat(anomaly_dfs, axis=1)
    final_df = plot_data_df.join(combined_anomalies, how='left')
    anomaly_cols = [col for col in final_df.columns if col.startswith('is_anomaly')]
    final_df[anomaly_cols] = final_df[anomaly_cols].fillna(0).astype(int)
    final_df['anomaly_count'] = final_df[anomaly_cols].sum(axis=1)

    # --- 3. Anomaly Logic ---
    cols_to_exclude = config.get('DETECTORS_TO_EXCLUDE', [])
    other_cols = []
    for col in anomaly_cols:
        is_excluded = False
        for exclusion in cols_to_exclude:
            if exclusion in col:
                is_excluded = True
                break
        if not is_excluded:
            other_cols.append(col)

    if not other_cols:
        final_df['other_anomaly_count'] = 0
    else:
        final_df['other_anomaly_count'] = final_df[other_cols].sum(axis=1)
    
    anomaly_logic_threshold = config.get('FINAL_ANOMALY_LOGIC_THRESHOLD', 3)
    final_df['final_anomaly_score'] = (final_df['other_anomaly_count'] > anomaly_logic_threshold).astype(int)
    
    if cols_to_exclude:
        plot_label = f"Anomaly (> {anomaly_logic_threshold}, excl. {cols_to_exclude[0]}..)"
    else:
        plot_label = f"Anomaly (Count > {anomaly_logic_threshold})"

    # --- 4. Merge Time & Holiday ---
    x_axis_label = "Timestep"
    if time_df is not None:
        try:
            final_df = final_df.merge(time_df, left_index=True, right_on='id_time', how='left')
            if 'time' in final_df.columns:
                final_df['time'] = pd.to_datetime(final_df['time'])
                final_df = final_df.set_index('time')
                x_axis_label = "Time"
                
                # Holidays
                if os.path.exists(holiday_file_path):
                    try:
                        non_work_df = pd.read_csv(holiday_file_path)
                        if 'Date' in non_work_df.columns and 'Type' in non_work_df.columns:
                            non_work_df['Date'] = pd.to_datetime(non_work_df['Date']).dt.normalize()
                            weekend_dates = set(non_work_df[non_work_df['Type'] == 'Weekend']['Date'])
                            holiday_dates = set(non_work_df[non_work_df['Type'] == 'Holiday']['Date'])
                            
                            final_df['normalized_date'] = final_df.index.tz_convert(None).normalize()
                            final_df['is_weekend'] = final_df['normalized_date'].isin(weekend_dates).astype(int)
                            final_df['is_holiday'] = final_df['normalized_date'].isin(holiday_dates).astype(int)
                    except Exception as e:
                        print(f"Error loading holidays: {e}")
        except Exception as e:
            print(f"Time merge error: {e}")

    # --- Save CSV ---
    # Generate a dynamic filename based on the folder name
    output_csv_name = f"{folder_name}.csv"
    final_df.to_csv(output_csv_name)
    print(f"Saved data to: {output_csv_name}")

    # --- 5. Stats ---
    total_timesteps = len(final_df)
    total_anomalies = final_df['final_anomaly_score'].sum()
    pct = (total_anomalies / total_timesteps) * 100
    stats_text = f"Anomalies: {total_anomalies}/{total_timesteps} ({pct:.2f}%)"

    # --- 6. Plotting ---
    print(f"Generating plot for {folder_name}...")
    fig, ax1 = plt.subplots(figsize=(18, 8))
    fig.suptitle(f'Anomaly Overview: {folder_name}', fontsize=16)

    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel('Original Value', color='tab:blue')
    ax1.plot(final_df.index, final_df['orig_value'], color='tab:blue', label='Value', zorder=2, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Calculate width for bars
    median_diff = None
    if pd.api.types.is_datetime64_any_dtype(final_df.index):
        median_diff = final_df.index.to_series().diff().median()
    if pd.isna(median_diff):
        median_diff = pd.Timedelta(hours=1) if pd.api.types.is_datetime64_any_dtype(final_df.index) else 1

    # Weekends/Holidays
    legend_patches = []
    
    def plot_shaded_regions(col_name, color, label):
        if col_name in final_df.columns and not final_df.empty:
            groups = final_df[final_df[col_name] == 1]
            if not groups.empty:
                final_df[f'{col_name}_grp'] = (final_df[col_name].diff() != 0).cumsum()
                for _, g_df in final_df[final_df[col_name] == 1].groupby(f'{col_name}_grp'):
                    ax1.axvspan(g_df.index.min(), g_df.index.max() + median_diff, color=color, alpha=0.3, zorder=-1, lw=0)
                return True
        return False

    if plot_shaded_regions('is_weekend', 'lightgreen', 'Weekend'):
        legend_patches.append(Rectangle((0,0),1,1, color='lightgreen', alpha=0.3, label='Weekend'))
    
    if plot_shaded_regions('is_holiday', 'lightpink', 'Holiday'):
        legend_patches.append(Rectangle((0,0),1,1, color='lightpink', alpha=0.4, label='Holiday'))

    # Anomalies
    anomalies = final_df[final_df['final_anomaly_score'] == 1]
    if not anomalies.empty:
        ax1.scatter(anomalies.index, anomalies['orig_value'], color='red', label=plot_label, zorder=3, s=15)

    # Clusters
    cluster_run_length = config.get('CLUSTER_RUN_LENGTH', 5)
    final_df['anomaly_group'] = (final_df['final_anomaly_score'].diff() != 0).cumsum()
    anomaly_groups = final_df[final_df['final_anomaly_score'] == 1]
    
    cluster_found = False
    if not anomaly_groups.empty:
        for group_id, group_df in anomaly_groups.groupby('anomaly_group'):
            if len(group_df) >= cluster_run_length:
                cluster_found = True
                
                start_t = group_df.index.min()
                end_t = group_df.index.max() + median_diff
                
                # Determine height dynamically
                min_v, max_v = group_df['orig_value'].min(), group_df['orig_value'].max()
                pad = (max_v - min_v) * 0.1 if (max_v - min_v) > 0 else (max_v*0.1 if max_v!=0 else 1)
                
                rect = Rectangle((start_t, min_v - pad), end_t - start_t, (max_v - min_v) + (2*pad),
                                 color='grey', alpha=0.2, zorder=0)
                ax1.add_patch(rect)

    if cluster_found:
        legend_patches.append(Rectangle((0,0),1,1, color='grey', alpha=0.2, label=f'Cluster ({cluster_run_length}+)'))

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines.extend(legend_patches)
    labels.extend([p.get_label() for p in legend_patches])
    ax1.legend(lines, labels, loc='upper left')

    # Text Box
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'), zorder=10)

    if pd.api.types.is_datetime64_any_dtype(final_df.index):
        fig.autofmt_xdate()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    # --- Save Plot ---
    plot_filename = f"plot_{folder_name}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to: {plot_filename}")
    
    # Optional: Show plot if config says so (default False for batch runs)
    if config.get('SHOW_PLOTS', True):
        plt.show()
    else:
        plt.close(fig) # Close memory to prevent leaks in loops

    print(f"Completed analysis for {folder_name}.\n")


# --- WRAPPER FUNCTION ---
def run_analysis_for_folder(folder_name, general_config):
    """
    Wrapper that resolves paths and calls the processor.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve specific paths
    folder_path = os.path.join(base_dir, folder_name)
    time_csv_path = os.path.join(base_dir, general_config["TIME_FILE"])
    holiday_csv_path = os.path.join(base_dir, general_config["HOLIDAY_FILE"])

    # Check existence
    if not os.path.exists(folder_path):
        print(f"Skipping '{folder_name}': Directory not found.")
        return
    
    if not os.listdir(folder_path):
        print(f"Skipping '{folder_name}': Directory is empty.")
        return

    # Run
    process_anomaly_files(general_config, folder_name, folder_path, time_csv_path, holiday_csv_path)


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    CONFIG = {
        # Files
        "TIME_FILE": "times_1_hour.csv",
        "HOLIDAY_FILE": "weekends_and_holidays.csv",
        
        # Logic
        "ANOMALY_SCORE_THRESHOLD": 0.70,
        "FINAL_ANOMALY_LOGIC_THRESHOLD": 2,
        "DETECTORS_TO_EXCLUDE": [''],
        "CLUSTER_RUN_LENGTH": 5,
        
        # Behavior
        "SHOW_PLOTS": True  # Set to True if you want to manually close each window
    }

    # --- LIST OF FOLDERS TO PROCESS ---
    # You can add as many folders here as you want
    FOLDERS_TO_PROCESS = ["n_packets", "n_flows", "n_bytes"]

    print(f"Starting batch analysis for {len(FOLDERS_TO_PROCESS)} folders...")
    
    for folder in FOLDERS_TO_PROCESS:
        run_analysis_for_folder(folder, CONFIG)

    print("All requested folders processed.")