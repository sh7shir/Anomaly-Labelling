# Step 1 - Processing the CESNET Datasets to use in the set of detectors in step 2

1_data_processing_for_CESNET folder contains a Python script designed to merge, format, and structure network traffic datasets (`1367.csv, 103.csv or other dataset from CESNET` and `times_1_hour.csv`) for both offline and online datasets to be used in the set of detectors in step 2.

* **Data Merging:** Seamlessly joins traffic metrics with their corresponding timestamps using `id_time`.
* **Timestamp Formatting:** Standardizes datetime strings into the `DD/MM/YYYY HH:MM` format.
* **Automated Directory Structuring:** Automatically generates `data_offline` and `data_online` directories.
* **Dataset Splitting:** Isolates specific features (`n_flows`, `n_packets`, `n_bytes`) into individual, pipeline-ready CSV files.
* **Label File Generation:** Prepares blank label files structured to match the online datasets for downstream classification tasks.

## Prerequisites

Ensure you have Python installed on your system along with the following libraries:

* `pandas`

You can install the required dependency via pip:

```bash
pip install pandas

```

##  Usage

1. **Prepare your files:** Ensure that the input datasets (`1367.csv` and `times_1_hour.csv`) are located in the same directory as the script.
2. **Run the script:** Execute the Python script from your terminal:

```bash
python script_for_ces_to_nab_format.py

```

## Output Structure

Running the script will generate two new folders in your current directory with the following structure:

### 1. `data_offline/`

Contains datasets with formatted timestamps and their corresponding values.

* `1367_n_flows.csv`
* `1367_n_packets.csv`
* `1367_n_bytes.csv`

**File format:**
| timestamp | value |
| :--- | :--- |
| 14/02/2014 14:30 | 6701 |

### 2. `data_online/`

Contains duplicated datasets where the time column header is intentionally left blank to match the NAB dataset column name used in the detectors, along with corresponding empty label files. The `<date_id>` is currently configured to `date_id = datetime.now().strftime("%y%m%d%H%M%S")`.

For each feature (e.g., `n_flows`), the following 4 files are generated:

* `1367_n_flows_origdata.csv` *(Original data)*
* `1367_n_flows_adddata-<date_id>.csv` *(Duplicated data)*
* `1367_n_flows_origlabels.csv` *(Empty label file)*
* `1367_n_flows_addlabels-<date_id>.csv` *(Empty label file)*

**Data file format:**
| | value |
| :--- | :--- |
| 14/02/2014 14:30 | 6701 |

**Label file format:**
| | value |
| :--- | :--- |
| | |

## Customization

If you need to change the generated `<date id>` for the `data_online` files, open the script and modify the `date_id` variable:

```python
# Change this value to match your required date ID
date_id = datetime.now().strftime("%y%m%d%H%M%S")

```

```

```

# Step 2 - Processing the Datasets using the set of Detectors


