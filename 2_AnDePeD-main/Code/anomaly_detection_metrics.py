import filenames
import read_files

import numpy as np


ANOMALY_WINDOW_SIZE = 'NAB'
NORMALIZE_METRICS = True
ADJUST_ANOMALIES_POSEDGE = True
ROUND = 3


def calculate_anomaly_detection_metrics_main(dataset_filename: str, result_filename: str, detector: str,
                                             data_dir: str = 'data', thresholds_file: str = 'config/thresholds.json',
                                             separator: str = '\\'):
    """
    Calculate the metrics of Precision, Recall and F-score based on the supplied input results.
    :param dataset_filename: Path leading to the original dataset.
    :param result_filename: Path leading to the results' dataset.
    :param detector: name of the anomaly detector algorithm.
    :param data_dir: Folder containing the original datasets.
    :param thresholds_file: Json file for detection thresholds.
    :param separator: Character separator between folder levels (usually '\\' or '/').
    :return: Returns the metrics of Precision, Recall and F-score.
    """
    # get category and dataset
    _, dataset_name = filenames.separate_names(dataset_filename, separator)[-2:]

    # read detections
    values = read_files.read_anomaly_detections(result_filename, detector, thresholds_file)

    # read flags
    flags = read_files.read_anomaly_flags(dataset_name, data_dir)

    # calculate the size of the anomaly window
    anomaly_win_size = calculate_aws(ANOMALY_WINDOW_SIZE, ANOMALY_WINDOW_SIZE == 'NAB', len(values), len(flags))

    # create anomaly windows
    ground_truth_wind = create_anomaly_windows(flags, anomaly_win_size, len(values))

    # adjust detections to only consider no->yes changes (if turned on)
    detections = adjust_anomaly_signals_posedge(values, ADJUST_ANOMALIES_POSEDGE)

    # populate the confusion matrix
    tp, fp, tn, fn = measure_confusion_matrix(detections, ground_truth_wind, anomaly_win_size, NORMALIZE_METRICS)

    # calculate necessary metrics
    return calculate_metrics(tp, fp, tn, fn)


def calculate_aws(orig_aws, nab_mode: bool, dataset_length: int, num_of_flags: int):
    """
    Calculates the size of the anomaly window based on the length of the dataset and the number of anomalies.
    :param orig_aws: Original desired Anomaly Window Size value.
    :param nab_mode: Boolean, whether to use NAB's method.
    :param dataset_length: Length of the original dataset.
    :param num_of_flags: Number of real anomalies in the original dataset.
    :return: Returns the Anomaly Window Size (AWS).
    """
    if orig_aws == 'NAB' or nab_mode:
        # print('data len: ' + str(dataset_length) + ', no of flags: ' + str(num_of_flags))
        if num_of_flags != 0:
            # print('AWS:' + str(int(np.ceil(.1 * dataset_length / num_of_flags))))
            return int(np.ceil(.1 * dataset_length / num_of_flags))
        else:
            # print('AWS:' + str(int(np.ceil(.1 * dataset_length))))
            return int(np.ceil(.1 * dataset_length))
    else:
        return orig_aws


def create_anomaly_windows(ground_truth, aws: int, length: int):
    """
    Returns an array with anomaly windows (0: outside the window, 1: inside the window).
    :param ground_truth: Anomaly flags (a list of indices).
    :param aws: Anomaly Window Size.
    :param length: Length of the original dataset.
    :return: Returns a list of 0/1 values for the anomaly windows.
    """
    anomaly_windows = list([0] * length)
    steps_since_middle = aws + 1
    overwrite_from = 0
    ground_truth_count = 0

    for y in range(length):
        # if we are at a timestep with a flagged anomaly:
        if ground_truth.count(y) > 0:
            steps_since_middle += 1
            ground_truth_count += 1
            # if two windows conflict, overwrite previous window
            if steps_since_middle < aws:
                overwrite_from = int(y - np.floor((steps_since_middle - 1) / 2))
            else:
                overwrite_from = y - aws
            # don't overwrite negative indices
            overwrite_from = np.max([overwrite_from, 0])
            # overwrite array with the anomaly window
            for x in range(max(overwrite_from, 0), min(y + aws + 1, length - 1)):
                anomaly_windows[x] = ground_truth_count
            steps_since_middle = 0
        else:
            if ground_truth_count > 0:
                steps_since_middle += 1

    return anomaly_windows


def adjust_anomaly_signals_posedge(detections, adjust: bool):
    """
    Only take into account no->yes changes in the signal, i.e. disregard the latter part
    of multiple-timestep-long signals.
    :param detections: List of anomaly signals.
    :param adjust: Boolean, whether to make this processing step.
    :return: Processed anomaly detection signals.
    """
    if adjust:
        anomalies_adjusted = list([False] * len(detections))
        # only the beginning of anomalies count as an anomaly signal (no->yes)
        for y in range(len(detections)):
            if y == 0:
                anomalies_adjusted[y] = False
            else:
                anomalies_adjusted[y] = detections[y] and (not detections[y - 1])

        return anomalies_adjusted
    else:
        return detections


def measure_confusion_matrix(detections, anomaly_windows, aws: int, normalize: bool):
    """
    From the list of detections and the anomaly windows, calculate the following metrics: TP, FP, TN, FN.
    :param detections: List of anomaly signals.
    :param anomaly_windows: A list of 0/1 values for the anomaly windows
    :param aws: Anomaly Window Size.
    :param normalize: Boolean, whether to normalize metrics outside the anomaly windows.
    :return: Returns the followinf four metrics: TP, FP, TN, FN.
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    last_detected_anomaly = 0

    for y in range(len(detections)):
        if y == 0:
            continue
        # if we are not in an anomaly window
        if anomaly_windows[y] == 0:
            # if we have just exited an anomaly window
            if anomaly_windows[y - 1] > 0:
                # if the last detected anomaly was not in the window we have just left
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            # if there is an anomaly signal
            if detections[y]:
                false_positives += 1
        # if we are inside an anomaly window
        else:
            # if the current window is different from the previous, and
            # we haven't just entered an anomaly window from 0
            if anomaly_windows[y - 1] != anomaly_windows[y] and anomaly_windows[y - 1] != 0:
                # if the last detected anomaly was not in the window we have just left
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            # if there is an anomaly signal
            if detections[y]:
                # if the last detected anomaly was not in this window
                if last_detected_anomaly != anomaly_windows[y]:
                    true_positives += 1
                    last_detected_anomaly = anomaly_windows[y]
    # normalize false positives and true negatives if selected
    if normalize:
        false_positives = false_positives / (2 * aws + 1)
        true_negatives = len(detections) / (2 * aws + 1) - \
                         true_positives - false_positives - false_negatives
    else:
        true_negatives = len(detections) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, true_negatives, false_negatives


def calculate_metrics(tp, fp, tn, fn):
    """
    Calculates the metrics of Precision, Recall and F-score based on the confusion matrix values (TP, FP, TN, FN).
    :param tp: True Positives.
    :param fp: False Positives.
    :param tn: True Negatives.
    :param fn: False Negatives.
    :return: Returns a list of these three metrics: Precision, Recall and F-score.
    """
    # precision
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # recall
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # F-score
    if precision + recall != 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    return [precision, recall, f_score]