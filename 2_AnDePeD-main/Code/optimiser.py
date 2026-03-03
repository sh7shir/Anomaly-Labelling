import optuna
import joblib
from datetime import datetime

import filenames
import anomaly_detection_metrics

import optimiser_config as oconf
import optimiser_func as ofunc
import main_config as conf
import printer


def objective_andeped(trial, algorithm, dataset, mode, test_id: str, omegas: list, modes_paths: list):
    # (1) select parameters
    k, alpha = ofunc.select_params(trial)

    # (2) read, preprocess, then save data
    mse = ofunc.read_preprocess_save_data_get_mse(dataset, alpha, k, omegas, modes_paths)

    if mode == 'I':
        # (3) run AnDePeD on the dataset
        ofunc.run_anomaly_detector_andeped(dataset, conf.OFFLINE_DATA_DIR, conf.PP_OFFL_DATA_DIR, algorithm)

        # (4) evaluate results (F-score metric)
        f_score = anomaly_detection_metrics.\
            calculate_anomaly_detection_metrics_main(dataset, conf.ANDEPED_EXPORT_FILE, 'AnDePeD',
                                                     conf.NAB_ORIG_DATA_DIR, conf.THRESHOLDS_FILE, conf.SEPARATOR)[2]
        # (5) save results
        ofunc.save_parameters_f_score(algorithm, dataset, alpha, k, f_score, test_id)

        printer.step_offline_preparation(trial.number, oconf.NUM_TRIALS)
        return f_score

    elif mode == 'II':
        # (3) return MSE of VMD
        printer.step_offline_preparation(trial.number, oconf.NUM_TRIALS)
        return mse

    else:
        return -1.0


def objective_nab(trial, algorithm, dataset, mode, test_id: str, omegas: list, modes_paths: list):
    # (1) select parameters
    k, alpha = ofunc.select_params(trial)

    # (2) read, preprocess, then save data
    mse = ofunc.read_preprocess_save_data_get_mse(dataset, alpha, k, omegas, modes_paths)

    if mode == 'I':
        # (3) run the selected algorithm on the dataset
        ofunc.run_anomaly_detector_nab(dataset, algorithm)

        # (4) evaluate results (average F-score metric)
        f_score = ofunc.get_fscore_metric_nab(dataset, algorithm)

        # (5) calculate objective function value
        ofunc.save_parameters_f_score(algorithm, dataset, alpha, k, f_score, test_id)

        printer.step_offline_preparation(trial.number, oconf.NUM_TRIALS)
        return f_score

    elif mode == 'II':
        # (3) return MSE of VMD
        printer.step_offline_preparation(trial.number, oconf.NUM_TRIALS)
        return mse

    else:
        return -1.0


def optimise_main(mode: str, algorithm: str, dataset: str, test_id: str):
    # select direction based on mode
    if mode == 'I':
        direction = 'maximize'  # F-score as the objective
    elif mode == 'II':
        direction = 'minimize'  # MSE of decomposition as the objective
    else:
        direction = 'maximize'

    # initialise save variables
    omegas = list()
    modes_paths = list()

    # create optuna study
    dataset_name = filenames.separate_names(dataset, '/')[-1]
    study_name = 'online_decomp_param_opt_' + algorithm + '_' + dataset_name + '_' +\
                 str(datetime.now().strftime('%Y%m%d%H%M%S'))
    study = optuna.create_study(study_name=study_name, direction=direction)

    # select appropriate optimisation objective
    if algorithm in conf.ANDEPED_ALGOS:
        wrapped_obj = lambda trial: objective_andeped(trial, algorithm, dataset, mode, test_id, omegas, modes_paths)
    elif algorithm in conf.NAB_ALGOS:
        wrapped_obj = lambda trial: objective_nab(trial, algorithm, dataset, mode, test_id, omegas, modes_paths)
    else:
        wrapped_obj = lambda trial: objective_andeped(trial, algorithm, dataset, mode, test_id, omegas, modes_paths)

    # run optuna study
    study.optimize(wrapped_obj, n_trials=oconf.NUM_TRIALS)
    joblib.dump(study, oconf.STUDY_SAVE_DIR + '/' + study_name + '.pkl')

    # get optimal parameters
    alpha_star = study.best_params['alpha']
    k_star = study.best_params['k']

    best_trial = study.best_trial.number
    omega_star = omegas[best_trial]
    modes_star_path = modes_paths[best_trial]

    return alpha_star, k_star, omega_star, modes_star_path
