import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def combine_and_plot_anomalies(csv_files, output_csv):
    """
    Combines anomaly data from multiple CSVs, calculates multi-anomalies,
    saves the combined data, and plots the results.
    """
    print("Starting anomaly combination process...")

    # --- NEW: Load weekends and holidays file ---
    weekends_holidays_file = os.path.join(SCRIPT_DIR, 'weekends_and_holidays.csv')
    wh_df = None
    if os.path.exists(weekends_holidays_file):
        try:
            print(f"Loading weekends and holidays from {weekends_holidays_file}...")
            wh_df = pd.read_csv(weekends_holidays_file)
            # --- FIX: Rename uppercase columns from CSV to lowercase used in code ---
            wh_df = wh_df.rename(columns={'Date': 'date', 'Type': 'type'})
            # Convert to date objects for clean comparison
            wh_df['date'] = pd.to_datetime(wh_df['date']).dt.date
            print("Successfully loaded weekends and holidays.")
        except Exception as e:
            print(f"Warning: Could not load or process {weekends_holidays_file}. Skipping background shading. Error: {e}")
            wh_df = None
    else:
        print(f"Warning: {weekends_holidays_file} not found. Skipping background shading.")
    # --- END NEW ---

    dataframes = []
    orig_data_cols = []
    anomaly_score_cols = []

    for f in csv_files:
        if not os.path.exists(f):
            print(f"Warning: File not found: {f}. Skipping.")
            continue
            
        print(f"Processing file: {f}")
        
        file_name_only = os.path.basename(f)
        base_name = os.path.splitext(file_name_only)[0]
        
        orig_col_name = f"orig_data_{base_name}"
        score_col_name = f"final_anomaly_score_{base_name}"
        
        orig_data_cols.append(orig_col_name)
        anomaly_score_cols.append(score_col_name)
        
        try:
            df = pd.read_csv(
                f, 
                usecols=['time', 'orig_value', 'final_anomaly_score']
            )
            
            df = df.rename(columns={
                'orig_value': orig_col_name,
                'final_anomaly_score': score_col_name
            })
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"Error reading or processing {f}: {e}")
            return

    if not dataframes:
        print("No dataframes were loaded. Exiting.")
        return

    print("Merging dataframes on 'time'...")
    combined_df = dataframes[0]
    
    for df in dataframes[1:]:
        combined_df = pd.merge(combined_df, df, on='time', how='outer')
        
    for col in orig_data_cols:
        # Interpolate missing values and then fill any remaining (e.g., at ends) with 0
        combined_df[col] = combined_df[col].interpolate(method='linear').fillna(0)
    for col in anomaly_score_cols:
        # Fill missing scores with 0
        combined_df[col] = combined_df[col].fillna(0).astype(int)

    print("Calculating 'multi_anomaly' score...")
    # A 'multi_anomaly' is when 2 or more individual anomalies occur at the same time
    combined_df['multi_anomaly'] = (combined_df[anomaly_score_cols].sum(axis=1) >= 2).astype(int)

    # --- Calculate Statistics ---
    total_rows = len(combined_df)
    total_anomalies = combined_df['multi_anomaly'].sum()
    anomaly_percentage = (total_anomalies / total_rows) * 100 if total_rows > 0 else 0
    
    print("\n--- Anomaly Statistics ---")
    print(f"Total data points: {total_rows}")
    print(f"Total 'multi_anomaly' events: {total_anomalies}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
    print("----------------------------\n")

    print("Scaling 'orig_data' columns using Min-Max (0-1)...")
    scaler = MinMaxScaler()
    scaled_data_cols = [f'scaled_{col}' for col in orig_data_cols]
    
    combined_df[scaled_data_cols] = scaler.fit_transform(combined_df[orig_data_cols])
    print("Scaling complete.")

    print(f"Saving combined data to {output_csv}...")
    combined_df.to_csv(output_csv, index=False)
    print("Save complete.")

    print("Generating plot for inspection...")
    
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values(by='time')
    
    # Create 5 subplots, each in its own row
    num_periods = 5
    # Adjust figsize to be wider (18) and taller (4 per plot)
    fig, axes = plt.subplots(num_periods, 1, figsize=(18, num_periods * 4), sharey=True) 
    
    # Ensure axes is an array even if num_periods is 1
    if num_periods == 1:
        axes = [axes]

    colors = plt.cm.get_cmap('tab10', len(scaled_data_cols))
    
    data_chunks = np.array_split(combined_df, num_periods)
    
    for i, df_chunk in enumerate(data_chunks):
        ax = axes[i] 
        
        if df_chunk.empty:
            continue # Skip if chunk is empty

        chunk_start_time = df_chunk['time'].min()
        chunk_end_time = df_chunk['time'].max()

        # --- NEW: Add background shading for weekends and holidays ---
        if wh_df is not None:
            # Get date objects for comparison
            chunk_start_date = chunk_start_time.date()
            chunk_end_date = chunk_end_time.date()

            # --- FIX: Match the capitalized values 'Weekend'/'Holiday' from the CSV ---
            # Find weekends in this chunk's range
            weekends = wh_df[
                (wh_df['type'] == 'Weekend') &
                (wh_df['date'] >= chunk_start_date) &
                (wh_df['date'] <= chunk_end_date)
            ]['date']
            
            # Find holidays in this chunk's range
            holidays = wh_df[
                (wh_df['type'] == 'Holiday') &
                (wh_df['date'] >= chunk_start_date) &
                (wh_df['date'] <= chunk_end_date)
            ]['date']

            # Add shading for each weekend day
            for day in weekends:
                start_of_day = pd.to_datetime(day)
                end_of_day = start_of_day + pd.Timedelta(days=1)
                # Add shading behind plot lines (zorder=0) and hide from legend
                ax.axvspan(start_of_day, end_of_day, color='lightgreen', alpha=0.3, zorder=0, label='_nolegend_')

            # Add shading for each holiday
            for day in holidays:
                start_of_day = pd.to_datetime(day)
                end_of_day = start_of_day + pd.Timedelta(days=1)
                ax.axvspan(start_of_day, end_of_day, color='lightpink', alpha=0.4, zorder=0, label='_nolegend_')
        # --- END NEW ---
        
        # Find anomalies within this specific chunk
        anomaly_times_chunk = df_chunk[df_chunk['multi_anomaly'] == 1]
        
        # Plot each scaled data column
        for j, col in enumerate(scaled_data_cols):
            color = colors(j)
            ax.plot(
                df_chunk['time'], 
                df_chunk[col], 
                label=col.replace("scaled_orig_data_", ""), # Cleaner label
                color=color, 
                alpha=0.8
            )
            
            # Scatter plot for anomalies on this line
            if not anomaly_times_chunk.empty:
                ax.scatter(
                    anomaly_times_chunk['time'], 
                    anomaly_times_chunk[col], 
                    color='red', 
                    s=40,  # Size of the marker
                    zorder=5 # Plot on top
                )

        ax.set_ylabel('Scaled Value (0 to 1)', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', rotation=0) 
        
        # Set title for the subplot
        if not df_chunk.empty:
            start_date = chunk_start_time.strftime('%Y-%m-%d')
            end_date = chunk_end_time.strftime('%Y-%m-%d')
            ax.set_title(f'Period {i+1}: {start_date} to {end_date}', fontsize=12)

    # Set X-label for the last plot only
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # Create a single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Add a proxy artist for the 'Multi-Anomaly Event'
    if not combined_df[combined_df['multi_anomaly'] == 1].empty:
        anomaly_proxy = plt.Line2D([0], [0], marker='o', color='w', 
                                     label='Multi-Anomaly Event',
                                     markerfacecolor='red', markersize=8)
        handles.append(anomaly_proxy)
        labels.append('Multi-Anomaly Event')

    # --- NEW: Add proxies for weekend/holiday to the legend ---
    if wh_df is not None:
        weekend_proxy = plt.Rectangle((0,0), 1, 1, color='lightgreen', alpha=0.3)
        holiday_proxy = plt.Rectangle((0,0), 1, 1, color='lightpink', alpha=0.4)
        handles.extend([weekend_proxy, holiday_proxy])
        labels.extend(['Weekend', 'Holiday'])
    # --- END NEW ---

    # Place legend to the right of the entire figure
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 0.9))
    
    # --- This is the section that adds the statistics to the plot ---
    stats_text = (
        f"--- Statistics ---\n"
        f"Total Data Points: {total_rows}\n"
        f"Multi-Anomalies: {total_anomalies}\n"
        f"Anomaly Rate: {anomaly_percentage:.2f}%"
    )
    fig.text(
        1.01, 0.7, stats_text, 
        transform=fig.transFigure, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.5)
    )
    # --- End of statistics section ---
    
    fig.suptitle('Scaled Data Time Series (Min-Max) with Multi-Anomalies (One Period Per Row)', fontsize=16)
    
    # Adjust layout to make space for legend, stats text, and suptitle
    fig.tight_layout(rect=[0, 0, 0.9, 0.95]) 
    
    print("Displaying plot for inspection...")
    plt.show()
    print("Plot window closed.")
    print("Process finished successfully.")

if __name__ == "__main__":
    # Define file paths relative to the script directory
    input_folder_name = 'combined'
    input_folder_path = os.path.join(SCRIPT_DIR, input_folder_name)
    file_names = ['n_bytes.csv', 'n_flows.csv', 'n_packets.csv']
    files_to_process = [os.path.join(input_folder_path, f) for f in file_names]
    combined_csv_file = os.path.join(SCRIPT_DIR, 'combined.csv')
    
    # Create the input directory if it doesn't exist (for testing)
    if not os.path.exists(input_folder_path):
        os.makedirs(input_folder_path)
        print(f"Created dummy input folder: {input_folder_path}")
        print("Please add your CSV files (n_bytes.csv, etc.) to this folder.")

    combine_and_plot_anomalies(files_to_process, combined_csv_file)


