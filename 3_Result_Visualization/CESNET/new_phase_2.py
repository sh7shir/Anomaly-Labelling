import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Set directory path to where the input files are located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, 'combined_unified')

# --- STEP 1: Load Columns into 3 Dataframes ---
columns_to_load = [
    'time', 'orig_value', 'detector_votes', 'is_detector_anomaly', 
    'id_time', 'time_of_day', 'expected_pattern', 'pattern_deviation', 
    'smoothed_deviation', 'is_pattern_anomaly', 'is_weekend', 'is_holiday'
]

print("1. Loading Dataframes...")

# Load Bytes
df_bytes = pd.read_csv(os.path.join(INPUT_DIR, 'n_bytes_unified.csv'), usecols=columns_to_load)
df_bytes['time'] = pd.to_datetime(df_bytes['time'])
df_bytes.set_index('time', inplace=True)

# Load Flows
df_flows = pd.read_csv(os.path.join(INPUT_DIR, 'n_flows_unified.csv'), usecols=columns_to_load)
df_flows['time'] = pd.to_datetime(df_flows['time'])
df_flows.set_index('time', inplace=True)

# Load Packets
df_packets = pd.read_csv(os.path.join(INPUT_DIR, 'n_packets_unified.csv'), usecols=columns_to_load)
df_packets['time'] = pd.to_datetime(df_packets['time'])
df_packets.set_index('time', inplace=True)


# --- STEP 2: Check Common Point for is_detector_anomaly ---
print("2. Checking for common anomalies...")

# Create a comparison dataframe aligned by the time index
comparison_df = pd.DataFrame(index=df_bytes.index)

# Pull the 'is_detector_anomaly' column from each dataframe
comparison_df['bytes_flag'] = df_bytes['is_detector_anomaly']
comparison_df['flows_flag'] = df_flows['is_detector_anomaly']
comparison_df['packets_flag'] = df_packets['is_detector_anomaly']

comparison_df.fillna(0, inplace=True)


# --- STEP 3: Mark Point as Anomaly if Match in >= 2 Cases ---
print("3. Calculating consensus...")

comparison_df['total_flags'] = (
    comparison_df['bytes_flag'] + 
    comparison_df['flows_flag'] + 
    comparison_df['packets_flag']
)

# Common anomaly if 2 or more files agree
common_anomaly_mask = comparison_df['total_flags'] >= 2
print(f"   -> Found {common_anomaly_mask.sum()} common anomalies (occurring in at least 2 files).")


# --- STEP 4: Scaling Data for Single Graph (Multivariate) ---
print("4. Scaling data (MinMax) for comparison...")
scaler = MinMaxScaler()

# Scale specific columns so they fit on y-axis [0, 1]
df_bytes['scaled_value'] = scaler.fit_transform(df_bytes[['orig_value']])
df_flows['scaled_value'] = scaler.fit_transform(df_flows[['orig_value']])
df_packets['scaled_value'] = scaler.fit_transform(df_packets[['orig_value']])


# --- STEP 5: Plotting Single Multivariate Graph ---
print("5. Plotting Multivariate Graph...")

fig, ax = plt.subplots(figsize=(18, 8))

# A. Plot the Main Data Lines
ax.plot(df_bytes.index, df_bytes['scaled_value'], label='Bytes (Scaled)', color='tab:blue', alpha=0.6, linewidth=1)
ax.plot(df_flows.index, df_flows['scaled_value'], label='Flows (Scaled)', color='tab:orange', alpha=0.6, linewidth=1)
ax.plot(df_packets.index, df_packets['scaled_value'], label='Packets (Scaled)', color='tab:green', alpha=0.6, linewidth=1)

# B. Plot Common Anomalies ON THE LINES (High Visibility)
# Filter data where the mask is True
anom_bytes = df_bytes[common_anomaly_mask]
anom_flows = df_flows[common_anomaly_mask]
anom_packets = df_packets[common_anomaly_mask]

# Scatter red dots with black edges on the actual values
ax.scatter(anom_bytes.index, anom_bytes['scaled_value'], color='red', edgecolor='black', s=60, zorder=10, label='Anomaly (Bytes)')
ax.scatter(anom_flows.index, anom_flows['scaled_value'], color='red', edgecolor='black', s=60, zorder=10, label='Anomaly (Flows)')
ax.scatter(anom_packets.index, anom_packets['scaled_value'], color='red', edgecolor='black', s=60, zorder=10, label='Anomaly (Packets)')


ax.set_title("Multivariate Anomaly Detection (Scaled 0-1)", fontsize=14)
ax.set_ylabel("Normalized Activity")
ax.set_xlabel("Time")
ax.set_ylim(-0.05, 1.15)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()