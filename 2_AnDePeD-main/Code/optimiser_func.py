import os
import numpy as np
import pandas as pd
from datetime import datetime

import filenames
import read_files
import write_files
import scaling
import vmd
import AnDePeD.ReRe_main_functions as andeped_func
import anomaly_detection_metrics
import nab_custom_windows as nabwindows

import main_config as conf
import optimiser_config as oconf


def select_params(trial):
    k = trial.suggest_int('k', oconf.K_MIN, oconf.K_MAX)
    alpha = trial.suggest_float('alpha', oconf.ALPHA_MIN, oconf.ALPHA_MAX)
    return k, alpha


def read_preprocess_save_data_get_mse(dataset: str, alpha: float, k: int, omegas: list, modes_paths: list):
    values = read_files.read_file_pandas(dataset, column='value', to_numpy=True)
    values = scaling.preprocess(values)
    values, omega, usum = vmd.decompose(values, alpha, k)

    # save omegas
    latest_omega = omega[-1].tolist()
    omegas.append(latest_omega)

    # save remainder data for anomaly detection
    export_remainder = pd.DataFrame(values, columns=['timestamp', 'value'])
    # replace timestamps with the original values from the NAB corpus
    export_remainder['timestamp'] = read_files.read_file_pandas(dataset, column='timestamp', to_numpy=True)
    export_remainder.to_csv(dataset.replace(conf.OFFLINE_DATA_DIR, conf.PP_OFFL_DATA_DIR), index=False)

    # calculate MSE for later use if needed
    remainder = export_remainder['value'].to_numpy()
    squared = np.square(remainder)  # element-wise square
    sum_sq = np.sum(squared)  # sum of elements
    mse = float(sum_sq / len(remainder))

    # and save usum (sum of modes) for online use
    export_usum = pd.DataFrame(usum, columns=['timestep', 'value'])
    usum_name = 'usum_' + str(datetime.now().strftime('%Y-%m-%d_%H%M%S_%f'))[:-3] + '.csv'
    usum_path = conf.MODE_SUM_DIR + '/' + usum_name
    export_usum.to_csv(usum_path)

    # add usum path to the list of paths
    modes_paths.append(usum_name)

    return mse


def run_anomaly_detector_andeped(source_dataset: str, data_dir: str, preprocessed_dir: str, algorithm: str):
    # create AnDePeD testrun file
    testrun = pd.DataFrame(columns=conf.ANDEPED_TESTRUN_COLUMNS)
    d = source_dataset.replace('\\', '/')
    if algorithm == 'AnDePeDPro':
        testrun.loc[len(testrun)] = conf.ANDEPED_PRO_TESTRUN_PARAMETERS + [d.replace(data_dir, preprocessed_dir)]
    else:
        testrun.loc[len(testrun)] = conf.ANDEPED_TESTRUN_PARAMETERS + [d.replace(data_dir, preprocessed_dir)]
    testrun.to_csv(conf.ANDEPED_TESTRUN_LOCATION, index=False)

    # run AnDePeD using the testrun just created
    andeped_func.run_rere(0, {}, {}, False, conf.ANDEPED_EXPORT_FILE)

    return


def run_anomaly_detector_nab(dataset_path: str, algorithm: str):
    dataset = filenames.separate_names(dataset_path, conf.SEPARATOR, remove_csv=False)[-1]
    custom_windows_path = conf.TEMP_DIR + '/' + conf.TMP_NAB_CUSTOM_WINDOWS_FILE
    nabwindows.create_custom_nab_windows_file(dataset, conf.NAB_WINDOWS_FILE, custom_windows_path)

    datadir = conf.PP_OFFL_DATA_DIR
    os.system('python3 run_nab.py --detect --skipConfirmation --dataDir ' + datadir +
              ' --resultsDir ' + conf.TEMP_DIR + ' --windowsFile ' + custom_windows_path + ' -d ' + algorithm +
              ' --profilesFile ' + conf.NAB_PROFILES_FILE)
    return


def get_fscore_metric_nab(dataset: str, algorithm: str):
    dataset_name = filenames.separate_names(dataset, '/')[-1]
    result_filename = conf.TEMP_DIR + '/' + algorithm + '/' + algorithm + '_' + dataset_name + '.csv'
    f_score = anomaly_detection_metrics.calculate_anomaly_detection_metrics_main(
        dataset, result_filename, algorithm, conf.NAB_ORIG_DATA_DIR, conf.THRESHOLDS_FILE, conf.SEPARATOR)[2]
    return f_score


def save_parameters_f_score(algorithm: str, dataset: str, alpha: float, k: int, f_score: float, test_id: str):
    dataset_name = filenames.separate_names(dataset, '/')[-1]
    write_files.append_file_pandas(conf.OPTUNA_PARAMS_SAVE_FILE.format('f_score', test_id),
                                   [algorithm, dataset_name, alpha, k, f_score])

    return


def save_parameters_mse(algorithm: str, dataset: str, alpha: float, k: int, mse: float, test_id: str):
    dataset_name = filenames.separate_names(dataset, '/')[-1]
    write_files.append_file_pandas(conf.OPTUNA_PARAMS_SAVE_FILE.format('mse', test_id),
                                   [algorithm, dataset_name, alpha, k, mse])
    return