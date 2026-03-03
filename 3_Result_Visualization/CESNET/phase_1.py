import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURATION ---
FOLDERS = ["n_packets", "n_flows", "n_bytes"]
TIME_FILE = "times_1_hour.csv"
HOLIDAY_FILE = "weekends_and_holidays.csv"
ANOMALY_SCORE_THRESH = 0.85
MIN_ALARM_COUNT = 2  # How many detectors must agree to call it an anomaly

def analyze_folder(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, folder_name)
    
    print(f"Processing: {folder_name}...")

    # --- STEP 1: Load and Combine Sensor Data ---
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Read the first file to get the 'orig_value' baseline
    df_master = pd.read_csv(all_files[0]).set_index('timestep')[['orig_value']]
    
    # Loop through all files to collect anomaly votes
    for f in all_files:
        detector_name = os.path.basename(f).split('.')[0]
        temp_df = pd.read_csv(f).set_index('timestep')
        
        col_name = f'vote_{detector_name}'
        df_master[col_name] = (temp_df['anomaly_score'] > ANOMALY_SCORE_THRESH).astype(int)

    # Sum the votes
    vote_cols = [c for c in df_master.columns if c.startswith('vote_')]
    df_master['total_votes'] = df_master[vote_cols].sum(axis=1)
    df_master['is_final_anomaly'] = (df_master['total_votes'] > MIN_ALARM_COUNT).astype(int)

    # --- STEP 2: Merge Time and Holiday Data (FIXED) ---
    # 1. Load Time
    time_df = pd.read_csv(os.path.join(base_dir, TIME_FILE))
    df_master = df_master.merge(time_df, left_index=True, right_on='id_time', how='left')
    
    # 2. Fix Timezone Mismatch
    # Convert to datetime
    df_master['time'] = pd.to_datetime(df_master['time'])
    # STRIP TIMEZONE (.tz_localize(None)) so it matches the simple holiday dates
    df_master['time'] = df_master['time'].dt.tz_localize(None)
    df_master = df_master.set_index('time')

    # 3. Prepare 'date_only' for matching
    df_master['date_only'] = df_master.index.normalize() # Now safe to use

    # 4. Load Holidays
    holiday_df = pd.read_csv(os.path.join(base_dir, HOLIDAY_FILE))
    holiday_df['Date'] = pd.to_datetime(holiday_df['Date']).dt.normalize()
    
    # 5. Create Masks
    weekend_dates = set(holiday_df[holiday_df['Type'] == 'Weekend']['Date'])
    holiday_dates = set(holiday_df[holiday_df['Type'] == 'Holiday']['Date'])
    
    df_master['is_weekend'] = df_master['date_only'].isin(weekend_dates)
    df_master['is_holiday'] = df_master['date_only'].isin(holiday_dates)
    
    # Clean up temp column
    df_master.drop(columns=['date_only'], inplace=True)

    # --- STEP 3: Save CSV Data ---
    out_dir_csv = os.path.join(base_dir, "combined")
    os.makedirs(out_dir_csv, exist_ok=True)
    df_master.to_csv(os.path.join(out_dir_csv, f"{folder_name}.csv"))

    # --- STEP 4: Plotting (With Type Fixes) ---
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_title(f"Anomaly Detection: {folder_name}")
    
    ax.plot(df_master.index, df_master['orig_value'], label='Original Value', color='#1f77b4', alpha=0.8)

    anomalies = df_master[df_master['is_final_anomaly'] == 1]
    ax.scatter(anomalies.index, anomalies['orig_value'], color='red', label='Anomaly', zorder=5, s=20)

    # Convert boolean series to numpy array to satisfy matplotlib requirements
    y_min, y_max = ax.get_ylim()
    weekend_mask = df_master['is_weekend'].fillna(False).astype(bool).to_numpy()
    holiday_mask = df_master['is_holiday'].fillna(False).astype(bool).to_numpy()

    ax.fill_between(df_master.index, y_min, y_max, where=weekend_mask, 
                    color='green', alpha=0.1, label='Weekend')
    ax.fill_between(df_master.index, y_min, y_max, where=holiday_mask, 
                    color='red', alpha=0.1, label='Holiday')

    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # --- STEP 5: Save Plot ---
    out_dir_plot = os.path.join(base_dir, "plots")
    os.makedirs(out_dir_plot, exist_ok=True)
    plt.savefig(os.path.join(out_dir_plot, f"plot_{folder_name}.png"))
    plt.show()
    plt.close()

    print(f"Finished {folder_name}. (Anomalies: {len(anomalies)})")

# --- EXECUTION ---
if __name__ == "__main__":
    for folder in FOLDERS:
        if os.path.exists(os.path.join(os.path.dirname(__file__), folder)):
            analyze_folder(folder)
        else:
            print(f"Skipping {folder} (folder not found)")