TEST_ID = '00001'  # used when saving some files

MODE = 'II'  # 'I' or 'II'
# 'I':  run VMD every timestep, need labelled anomalies
# 'II': run VMD once at the beginning, normal data only

L = 200
TRUNCATE_MODES_STAR = False

OFFLINE_DATA_DIR = 'data_offline'
ONLINE_DATA_DIR = 'data_online'
PP_OFFL_DATA_DIR = 'datapp_offline'
PP_ONL_DATA_DIR = 'datapp_online'
RESULTS_DIR = 'results'
MODE_SUM_DIR = 'modesums'
TRUNCATED_MODE_SUM_DIR = 'modesums/truncated'

TEMP_DIR = 'temp'
OPTUNA_PARAMS_SAVE_FILE = RESULTS_DIR + '/{}_per_params_and_datasets_{}.csv'
OFFLINE_PREP_OPTIMAL_PARAMS_FILE = RESULTS_DIR + '/offline_prep_optimal_params_' + TEST_ID + '_mode_{}.csv'

SEPARATOR = '/'

# AnDePeD related parameters
ANDEPED_MAIN_DIR = 'AnDePeD'
ANDEPED_EXPORT_FILE = 'AnDePeD/eval_results_temp.csv'
ANDEPED_TESTRUN_LOCATION = 'AnDePeD/testRuns/signal_testrun.csv'
ANDEPED_TESTRUN_COLUMNS = ['B', 'THRESHOLD_STRENGTH', 'USE_WINDOW', 'WINDOW_SIZE', 'USE_AGING', 'USE_AARE_AGING',
                        'USE_THD_AGING', 'AGE_POWER', 'USE_AUTOMATIC_WS_AP', 'USE_OFFSET_COMP',
                        'ACCEPTABLE_AVG_DURATION', 'USE_AUTOMATIC_OFFSET', 'OFFSET_WINDOW_SIZE', 'OFFSET_PERCENTAGE',
                        'NUM_EPOCHS', 'NUM_NEURONS', 'FILENAME']
ANDEPED_TESTRUN_PARAMETERS = [30, 3.0, 'T', 1000, 'T', 'T', 'F', 2.0, 'F', 'F', 1, 'F', 0, 0, 30, 30]
ANDEPED_PRO_TESTRUN_PARAMETERS = [30, 3.0, 'T', 800, 'T', 'T', 'F', 2.5, 'T', 'T', 1, 'T', 0, 0, 30, 30]


# NAB related parameters
NAB_MAIN_DIR = 'NAB'
NAB_ORIG_DATA_DIR = 'NAB/evaluation/data'
THRESHOLDS_FILE = 'NAB/evaluation/config/thresholds.json'
NAB_WINDOWS_FILE = 'NAB/evaluation/labels/combined_windows.json'
TMP_NAB_CUSTOM_WINDOWS_FILE = 'tmp_nab_windows.json'
NAB_LABELS_FILE = 'NAB/evaluation/labels/combined_labels.json'
NAB_PROFILES_FILE = 'NAB/evaluation/config/profiles.json'

# complete list of AnDePeD-like and NAB-based detectors
ANDEPED_ALGOS = ['AnDePeD', 'AnDePeDPro']
NAB_ALGOS = ['bayesChangePt', 'windowedGaussian',
             'relativeEntropy', 'earthgeckoSkyline',
             'contextOSE', 'knncad']

OFFLINE_COLUMN_NAME = 'value'
ONLINE_COLUMN_NAME = 'value'  # has to be left on 'value' for compatibility with NAB and AnDePeD

ONLINE_FILE_STRUCTURES = ['{}_origdata.csv',  # 0
                          '{}_origlabels.csv',  # 1
                          '{}_adddata-{}.csv',  # 2
                          '{}_addlabels-{}.csv']  # 3

ALGORITHMS = ANDEPED_ALGOS + NAB_ALGOS
