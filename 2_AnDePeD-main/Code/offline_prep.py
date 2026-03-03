import numpy as np

import filenames
import read_files
import write_files
import main_config as conf
import optimiser
import printer


def prepare_procedure(mode: str, algorithm: str, dataset: str, test_id: str):
    """
    Offline preparation for real-time VMD-based mode removal and pre-processing
    :param mode: 'I' or 'II'
    :param algorithm: standard name of the anomaly detection algorithm
    :param dataset: path of dataset to analyse
    :param test_id: any user_specified string to use when saving some files
    :return: alpha_star, k_star, l_vmd, modes_star_path, data_min, data_max
    """

    dataset_name = filenames.separate_names(dataset, '/')[-1]
    printer.begin_offline_preparation(algorithm, dataset_name, mode)

    data_min, data_max, orig_length = get_min_max_len_of_orig_data(dataset)

    if mode == 'I':
        write_files.init_file_pandas(conf.OPTUNA_PARAMS_SAVE_FILE.format('f_score', test_id),
                                     ['algorithm', 'dataset', 'alpha', 'k', 'f_score'])

        # run Optuna, get optimal results
        alpha_star, k_star, omega_star, _ = optimiser.optimise_main(mode, algorithm, dataset, test_id)

        # calculate l_vmd as the smallest
        l_vmd = calculate_l_vmd(omega_star, orig_length)

        # save optimal results
        save_optimal_results(mode, algorithm, dataset, alpha_star, k_star, omega_star, l_vmd, '')

        printer.end_offline_preparation(algorithm, dataset_name, mode, 'alpha_star=' + str(alpha_star) +\
                                        ' , k_star=' + str(k_star) + ' , l_vmd=' + str(l_vmd))
        return alpha_star, k_star, l_vmd, '-1', data_min, data_max

    elif mode == 'II':
        write_files.init_file_pandas(conf.OPTUNA_PARAMS_SAVE_FILE.format('mse', test_id),
                                     ['algorithm', 'dataset', 'alpha', 'k', 'mse'])

        # run Optuna, get optimal results
        _, __, omega_star, modes_star_path = optimiser.optimise_main(mode, algorithm, dataset, test_id)

        # truncate the sum of modes to an integer multiple of the largest period
        if conf.TRUNCATE_MODES_STAR:
            modes_star_path = truncate_modes_star(modes_star_path, omega_star)
        else:
            modes_star_path = conf.MODE_SUM_DIR + '/' + modes_star_path

        # save optimal results
        save_optimal_results(mode, algorithm, dataset, -1.0, -1, [-1], -1, modes_star_path)

        printer.end_offline_preparation(algorithm, dataset_name, mode, 'modes_star_path=' + modes_star_path)
        return -1.0, -1, -1, modes_star_path, data_min, data_max

    return


def calculate_l_vmd(omega_star: list, orig_length: int):
    # if there is not a single other omega_star than DC, we return twice the original length
    if len(omega_star) < 2:
        return 2 * orig_length

    omega_min = min(omega_star[1:])  # the smallest centre frequency <-> the largest period (we exclude omega_0)

    if omega_min > 0:
        largest_period = 1 / omega_min
        return int(np.ceil(2 * largest_period))
    else:
        raise ZeroDivisionError('The smallest centre frequency (omega_min) cannot be zero!')


def save_optimal_results(mode: str, algorithm: str, dataset: str, alpha_star: float, k_star: int,
                         omega_star: list, l_vmd: int, modes_star_path: str):
    dataset_name = filenames.separate_names(dataset, '/')[-1]
    if mode == 'I':
        write_files.append_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('I'),
                                       [algorithm, dataset_name, alpha_star, k_star, omega_star, l_vmd])
    elif mode == 'II':
        write_files.append_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('II'),
                                       [algorithm, dataset, modes_star_path])
    return


def truncate_modes_star(modes_star_path: str, omega_star: list):
    # open full file
    full_modes_star = read_files.read_file_pandas(conf.MODE_SUM_DIR + '/' + modes_star_path)

    # calculate largest period
    omega_min = min(omega_star[1:])  # the smallest centre frequency <-> the largest period (we exclude omega_0)
    if omega_min > 0:
        largest_period = int(np.ceil(1 / omega_min))
    else:
        raise ZeroDivisionError('The smallest centre frequency (omega_min) cannot be zero!')

    # if the largest period exceeds the data length or is non-positive somehow, do not truncate
    if largest_period > len(full_modes_star) or largest_period < 1:
        trunc_modes_star = full_modes_star

    else:
        # calculate the largest integer multiple of the largest period that fits in the length of modes_star
        to_remove_len = len(full_modes_star) % largest_period
        last_remaining_index = len(full_modes_star) - 1 - to_remove_len

        # truncate dataframe
        trunc_modes_star = full_modes_star.truncate(after=last_remaining_index)

    # select new path
    old_usum_name = modes_star_path.split('/')[-1]
    trunc_modes_star_path = conf.TRUNCATED_MODE_SUM_DIR + '/trunc_' + old_usum_name

    # export truncated modes_star to new path
    trunc_modes_star.to_csv(trunc_modes_star_path)

    return trunc_modes_star_path


def get_min_max_len_of_orig_data(path: str):
    orig_data = read_files.read_file_pandas(path, column=conf.OFFLINE_COLUMN_NAME, to_numpy=True)
    data_min = np.min(orig_data)
    data_max = np.max(orig_data)
    orig_length = len(orig_data)

    return data_min, data_max, orig_length
