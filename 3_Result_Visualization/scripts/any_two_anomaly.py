import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

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
                print(f"Loaded time data from {os.path.basename(time_file_path)}")
            else:
                print(f"Warning: {time_file_path} missing columns. Using integer timesteps.")
                time_df = None
        except Exception as e:
            print(f"Error loading time file: {e}")

    # --- 2. Process Anomaly CSVs ---
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{folder_path}'. Skipping.")
        return

    anomaly_dfs = []
    plot_data_df = None
    first_file_processed = False
    
    # Individual detector threshold (probability -> binary)
    anomaly_score_threshold = config.get('ANOMALY_SCORE_THRESHOLD', 0.95)

    print(f"Found {len(csv_files)} detectors. Processing...")

    for f_path in csv_files:
        try:
            filename = os.path.basename(f_path)
            df = pd.read_csv(f_path)
            
            # Handle different CSV formats
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

            # Extract detector name
            match = re.search(r'_\d{5}_(.*)_origdata\.csv', filename)
            name_part = match.group(1) if match else filename.rsplit('.', 1)[0]
            
            col_name = f'is_anomaly_{name_part}'
            
            # Convert score to 1 (Anomaly) or 0 (Normal)
            df[col_name] = (df['anomaly_score'] > anomaly_score_threshold).astype(int)
            anomaly_dfs.append(df[[col_name]])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not first_file_processed or not anomaly_dfs:
        print("Error: No valid data processed. Skipping.")
        return

    # Combine all detectors
    combined_anomalies = pd.concat(anomaly_dfs, axis=1)
    final_df = plot_data_df.join(combined_anomalies, how='left')
    
    anomaly_cols = [col for col in final_df.columns if col.startswith('is_anomaly')]
    final_df[anomaly_cols] = final_df[anomaly_cols].fillna(0).astype(int)

    # --- 3. VOTING LOGIC ---
    min_votes = config.get('MIN_VOTES_FOR_ANOMALY', 1)
    
    final_df['vote_count'] = final_df[anomaly_cols].sum(axis=1)
    final_df['final_anomaly_score'] = (final_df['vote_count'] >= min_votes).astype(int)
    
    print(f"Logic Applied: Mark as anomaly if >= {min_votes} detectors agree.")
    
    # --- 4. Merge Time & Holiday ---
    x_axis_label = "Timestep"
    if time_df is not None:
        try:
            final_df = final_df.merge(time_df, left_index=True, right_on='id_time', how='left')
            if 'time' in final_df.columns:
                final_df['time'] = pd.to_datetime(final_df['time'])
                final_df = final_df.set_index('time')
                x_axis_label = "Time"
                
                if os.path.exists(holiday_file_path):
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
        except Exception: pass

    # --- SAVE RESULTS (New Logic) ---
    # 1. Identify parent directory of the input folder
    parent_dir = os.path.dirname(folder_path) 
    
    # 2. Create 'combined' folder if it doesn't exist
    combined_dir = os.path.join(parent_dir, "combined")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
        print(f"Created directory: {combined_dir}")

    # 3. Save file named as the folder name (e.g., n_packets.csv) inside 'combined'
    output_filename = f"{folder_name}.csv"
    output_path = os.path.join(combined_dir, output_filename)
    
    final_df.to_csv(output_path)
    
    # Stats
    total = len(final_df)
    anomalies = final_df['final_anomaly_score'].sum()
    stats_text = f"Votes >= {min_votes}: {anomalies}/{total} ({(anomalies/total)*100:.2f}%)"
    print(stats_text)
    print(f"Results saved to: {output_path}")

    # --- 5. Plotting ---
    print(f"Generating plot...")
    fig, ax1 = plt.subplots(figsize=(18, 8))
    fig.suptitle(f'Anomaly Overview: {folder_name} (Threshold: {min_votes} votes)', fontsize=16)

    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel('Value', color='tab:blue')
    ax1.plot(final_df.index, final_df['orig_value'], color='tab:blue', alpha=0.8, label='Value')
    
    # Background Shading
    legend_patches = []
    median_diff = pd.Timedelta(hours=1) if pd.api.types.is_datetime64_any_dtype(final_df.index) else 1
    if pd.api.types.is_datetime64_any_dtype(final_df.index):
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

    # Anomalies
    anomalies_df = final_df[final_df['final_anomaly_score'] == 1]
    if not anomalies_df.empty:
        ax1.scatter(anomalies_df.index, anomalies_df['orig_value'], color='red', label=f'Anomaly (Votes>={min_votes})', zorder=3, s=15)

    # Clusters
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

    # Finalize Plot
    lines, labels = ax1.get_legend_handles_labels()
    lines.extend(legend_patches)
    labels.extend([p.get_label() for p in legend_patches])
    ax1.legend(lines, labels, loc='upper left')

    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'), zorder=10)

    if pd.api.types.is_datetime64_any_dtype(final_df.index):
        fig.autofmt_xdate()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # Save Plot (We save this in the combined folder too for cleaner organization, or main folder?)
    # Defaulting to main folder as per typical use, or you can change to combined_dir
    plot_filename = f"plot_{folder_name}.png"
    plt.savefig(plot_filename)
    print(f"Saved plot to: {plot_filename}")
    
    if config.get('SHOW_PLOTS', True): 
        print(">> Displaying Plot. Close the window to process the next folder...")
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
        "ANOMALY_SCORE_THRESHOLD": 0.85,
        "MIN_VOTES_FOR_ANOMALY": 2, 
        "CLUSTER_RUN_LENGTH": 5,
        "SHOW_PLOTS": True
    }

    FOLDERS = ["n_packets", "n_flows", "n_bytes"]

    for folder in FOLDERS:
        run_analysis_for_folder(folder, CONFIG)