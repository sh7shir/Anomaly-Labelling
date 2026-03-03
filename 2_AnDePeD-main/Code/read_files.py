import pandas as pd
import json

import main_config as conf

# detectors where a .5 detection threshold is sufficient
HALVER_DETECTORS = ['ReRe', 'rere', 'Alter-ReRe', 'alter-rere', 'AREP', 'arep',
                    'AnDePeD', 'andeped', 'ANDEPED', 'AnDePeDPro', 'andepedpro', 'ANDEPEDPRO']


def read_anomaly_detections(filename: str, detector: str, thds_file: str, data_column: str = 'anomaly_score',
                            thresholds_type: str = 'standard'):
    """
    Reads detection results and thresholds, compares them, and returns a boolean list of anomalies/no anomalies.
    :param filename: The path leading to the detection results.
    :param detector: The name of the anomaly detection algorithm.
    :param thds_file: The path leading to the detection thresholds.
    :param data_column: The name of the column in the data file containing raw detection scores.
    :param thresholds_type: Choose the type of threshold to access.
    :return: Returns a boolean list of anomalies/no anomalies.
    """
    data = pd.read_csv(filename)
    results = data[data_column]
    if detector in HALVER_DETECTORS:
        threshold = .5
    else:
        with open(thds_file) as json_file:
            threshold = float(json.load(json_file)[detector][thresholds_type]['threshold'])
    return [True if score >= threshold else False for score in results]


def read_anomaly_flags(dataset: str, eval_data_dir: str, flags_filename: str = conf.NAB_LABELS_FILE):
    """
    Read anomaly flags and return where they are (indices) in the dataset.
    :param category: Data category.
    :param dataset: Name of the dataset.
    :param eval_data_dir: Folder containing the datasets.
    :param flags_filename: The name of the file containing flags.
    :return: A list of indices that show where flags are in the dataset.
    """
    with open(flags_filename) as json_file:
        all_elements = json.load(json_file)
        for element in all_elements:
            if dataset in element:
                key = element
                break

        labels = all_elements[key]

    data = pd.read_csv(eval_data_dir + '/' + key)
    timestamps = data['timestamp'].tolist()
    indexes = []

    for label in labels:
        indexes.append(timestamps.index(label))

    return indexes


def read_anomaly_detections_andeped(filename: str):
    """
    Reads detection results for AnDePeD, and returns a boolean list of anomalies/no anomalies.
    :param filename: The path leading to the detection results.
    :return: Returns a boolean list of anomalies/no anomalies.
    """
    data = pd.read_csv(filename)
    results = data['anomaly_score']
    return [True if score >= .5 else False for score in results]


def read_file_pandas(filename: str, column=None, to_numpy: bool = False):
    """
    A general purpose function for quickly reading csv files using pandas.
    :param filename: The path leading to the file to be read.
    :param column: The name of the column to return. If none, returns all columns as a pandas dataframe.
    :param to_numpy: Boolean that directs whether to return a numpy array or a pandas dataframe.
    :return: The desired array or dataframe.
    """
    if column is None:
        return pd.read_csv(filename)
    else:
        if to_numpy:
            return pd.read_csv(filename)[column].to_numpy()
        else:
            return pd.read_csv(filename)[column]
