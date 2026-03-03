import pandas as pd
from datetime import datetime
import os

def process_datasets():
    # 1. Load the data and merge them based on 'id_time'
    df_1367 = pd.read_csv('1367.csv')
    df_times = pd.read_csv('times_1_hour.csv')
    df_merged = pd.merge(df_1367, df_times, on='id_time', how='inner')

    # Format the time column to match "14/02/2014 14:30" (DD/MM/YYYY HH:MM)
    df_merged['time'] = pd.to_datetime(df_merged['time']).dt.strftime('%d/%m/%Y %H:%M')

    # 2. Create the output folders
    os.makedirs('data_offline', exist_ok=True)
    os.makedirs('data_online', exist_ok=True)

    columns_to_extract = ['n_flows', 'n_packets', 'n_bytes']

    # 3. Process data for 'data_offline' folder
    for col in columns_to_extract:
        df_offline = df_merged[['time', col]].copy()
        
        # Rename columns to 'timestamp' and 'value'
        df_offline.rename(columns={'time': 'timestamp', col: 'value'}, inplace=True)
        
        # Save the 3 CSV files
        offline_filename = f'data_offline/1367_{col}.csv'
        df_offline.to_csv(offline_filename, index=False)

    # 4. Process data for 'data_online' folder
    date_id = datetime.now().strftime("%y%m%d%H%M%S")
    
    for col in columns_to_extract:
        df_online = df_merged[['time', col]].copy()
        
        # Keep time column blank (header='') and rename the feature to 'value'
        df_online.rename(columns={'time': '', col: 'value'}, inplace=True)
        
        # Set the dataset name (e.g., '1367_n_flows')
        dataset_name = f'1367_{col}'
        
        # Create and duplicate the data files
        df_online.to_csv(f'data_online/{dataset_name}_origdata.csv', index=False)
        df_online.to_csv(f'data_online/{dataset_name}_adddata-{date_id}.csv', index=False)
        
        # Create the 2 empty label files with the same column names
        df_empty_labels = pd.DataFrame(columns=['', 'value'])
        df_empty_labels.to_csv(f'data_online/{dataset_name}_origlabels.csv', index=False)
        df_empty_labels.to_csv(f'data_online/{dataset_name}_addlabels-{date_id}.csv', index=False)

if __name__ == "__main__":
    process_datasets()
    print("Files successfully created in data_offline and data_online directories.")