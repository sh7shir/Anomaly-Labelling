import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def detect_pattern_anomalies(csv_path):
    print(f"Loading data from {csv_path}...")
    
    # 1. Load and Preprocess
    df = pd.read_csv(csv_path)
    
    # --- FIX: Use 'mixed' format to handle ISO strings with timezones automatically ---
    # utc=True converts everything to UTC, preventing issues with mixed timezones
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # Interpolate missing values to ensure continuous pattern analysis
    df['orig_value'] = df['orig_value'].interpolate(method='time')
    
    print(f"Data loaded: {len(df)} rows. Time range: {df.index.min()} to {df.index.max()}")

    # 2. Define Periodicity (10-minute intervals -> Daily Pattern)
    df['time_of_day'] = df.index.time
    
    # 3. Learn the "Expected Pattern" (The Shape)
    # We use Median instead of Mean to avoid outliers affecting the template
    pattern_template = df.groupby('time_of_day')['orig_value'].median()
    
    # Map the pattern back to the main dataframe
    df['expected_pattern'] = df['time_of_day'].map(pattern_template)
    
    # 4. Calculate Deviation from Pattern
    # We look at the absolute difference between Actual and Expected
    df['pattern_deviation'] = np.abs(df['orig_value'] - df['expected_pattern'])
    
    # 5. Smooth the Deviation (CRITICAL STEP)
    # Window = 24 (approx 24 hours if data is hourly, adjust based on data frequency)
    rolling_window = 24 
    df['smoothed_deviation'] = df['pattern_deviation'].rolling(window=rolling_window, center=True).mean()
    
    # 6. Dynamic Thresholding
    # Anomaly if the smoothed deviation is > 3 Std Devs from the mean deviation
    dev_mean = df['smoothed_deviation'].mean()
    dev_std = df['smoothed_deviation'].std()
    threshold = dev_mean + (.0 * dev_std)
    
    df['is_anomaly'] = (df['smoothed_deviation'] > threshold).astype(int)
    
    # Group anomalies for reporting
    df['anomaly_grp'] = (df['is_anomaly'].diff() != 0).cumsum()
    
    # 7. Visualization
    print("Generating Pattern Analysis Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Actual vs Expected Pattern
    ax1.set_title("Pattern Recognition: Actual vs Expected 'Shape'", fontsize=14)
    ax1.plot(df.index, df['orig_value'], color='tab:blue', alpha=0.7, label='Actual Data (orig_value)')
    ax1.plot(df.index, df['expected_pattern'], color='black', linestyle='--', alpha=0.6, label='Expected Pattern (Median)')
    
    # Highlight Anomalies on Main Plot
    anomalies = df[df['is_anomaly'] == 1]
    if not anomalies.empty:
        ax1.scatter(anomalies.index, anomalies['orig_value'], color='red', s=10, zorder=5, label='Detected Pattern Anomaly')
        
    ax1.set_ylabel("Original Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The "Pattern Deviation" Score
    ax2.set_title("Deviation Score (Difference from Expected Shape)", fontsize=14)
    ax2.plot(df.index, df['smoothed_deviation'], color='tab:orange', label='Smoothed Deviation')
    ax2.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.0f})')
    ax2.fill_between(df.index, 0, df['smoothed_deviation'], where=(df['smoothed_deviation'] > threshold), color='red', alpha=0.3)
    
    ax2.set_ylabel("Deviation Magnitude")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format Dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    output_img = "pattern_anomaly_result.png"
    plt.show()
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")
    
    # Save CSV results
    output_csv = "n_bytes_pattern_anomalies.csv"
    df[['orig_value', 'expected_pattern', 'smoothed_deviation', 'is_anomaly']].to_csv(output_csv)
    print(f"Results saved to {output_csv}")
    
    # Print Stats
    total_anomalies = df['is_anomaly'].sum()
    print(f"\nAnalysis Complete.")
    print(f"Total Anomalies Found: {total_anomalies} points")
    print(f"Percentage: {(total_anomalies/len(df))*100:.2f}%")

if __name__ == "__main__":
    csv_file = './combined/n_packets.csv'
    detect_pattern_anomalies(csv_file)