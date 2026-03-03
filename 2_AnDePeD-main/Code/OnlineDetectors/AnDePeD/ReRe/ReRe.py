"""
INITIALIZATION PARAMETERS

These variables contain the hyperparameters for AnDePeD operation.
"""


import numpy as np
import pandas as pd
import multiprocessing


class ReRe:
    # class variables for multiprocessing
    class_lock = multiprocessing.Lock()
    progress_for_executions = {}
    executions_max = {}
    inst_num = 0

    #########################
    # SET THESE PARAMETERS! #
    #########################

    OPERATION = 'file'
    # 'manual' - set parameters by hand,
    # 'file' - set parameters in a separate file (can be multiple tests)

    if OPERATION == 'file':
        TESTRUN_FOLDER = 'AnDePeD/testRuns'  # set the folder of the test runs here
        TESTRUN_FILE = 'signal_testrun.csv'  # set filename of the test run here

    # set 'manual' parameters here ('file' mode will overwrite these):
    # RePAD parameters
    B = 30  # look-back param.
    F = 1  # predict-forward param.
    THRESHOLD_STRENGTH = 3  # this is the coefficient of sigma (std. dev.), def.=3
    USE_WINDOW = True  # whether to use windowed-mode execution
    WINDOW_SIZE = 800  # the size of the window when calculating AARE and thd values

    USE_AGING = True  # whether to turn on aging of values when calculating AARE and thd
    USE_AARE_AGING = True
    USE_THD_AGING = False
    AGE_POWER = 2.5

    USE_AUTOMATIC_WS_AP = True

    USE_OFFSET_COMP = True  # whether to use the implemented offset correction segment that triggers a model retrain
    ACCEPTABLE_AVG_DURATION = 1  # [seconds]: the maximum allowed duration of one timestep on average
    USE_AUTOMATIC_OFFSET = True
    OFFSET_WINDOW_SIZE = 30  # the number of datapoints to use with offset compensation
    OFFSET_PERCENTAGE = .9  # the ratio of datapoints that need to be above threshold to trigger LSTM retrain

    # LSTM parameters
    NUM_EPOCHS = 30  # number of epochs
    NUM_NEURONS = 30  # number of neurons

    # implementation parameters
    FILENAME = 'ec2_cpu_utilization_ac20cd.csv'  # the name of the imported file
    TO_CSV = True  # whether to dump data to a .csv file
    EVAL_EXPORT = True  # whether to export data for evaluation use
    EVAL_FOLDER = 'eval_results'
    DO_DIV = False  # whether to divide data by the largest number in the dataset times 1,1
    USE_LESS = False  # whether to use only the beginning of the data
    LESS = 100  # the number of data points to use
    DEBUG = False  # whether to print verbose information to the console while operating
    NOTES = 'AnDePeD test'  # type notes here to be saved with the hyperparameters
    STATUS_BAR = True  # replaces the #/# lines showing the algorithm operation with a status bar if DEBUG == False
    BATCH_STATUS_BAR = True

    ###############################
    # END OF USER SET PARAMETERS! #
    ###############################

    run_for = 1
    data = pd.DataFrame()
    length = 0

    # initialize parameters for automatic window and ageing
    if USE_AUTOMATIC_WS_AP:
        WS_AP_COEFF = 2  # the number to multiply the WS and AP parameters by
        SIGNAL_DATABASE_LEN = B ** 2  # signal window size
        DECISION_FREQ = B  # the minimum number of timesteps between two tuning events
        FLAPPING_LENGTH_COEFF = 1.5  # the maximum number of times no_length can be larger than anom_length
        SIGNAL_THRESHOLD_COEFF = B  # the coefficient when calculating percentage threshold of 'freq_retrain'
        TOO_LONG_ANOM_COEFF = 2.5  # this times B is the maximum allowed length of an anomaly

    def __init__(self):
        from OnlineDetectors.AnDePeD.ReRe.lstm_func import Lstm

        self.values = list()

        self.b_s = None
        self.thr_s = None
        self.usw_s = None
        self.wis_s = None
        self.usa_s = None
        self.uaa_s = None
        self.uta_s = None
        self.agp_s = None
        self.uap_s = None
        self.uoc_s = None
        self.aad_s = None
        self.uao_s = None
        self.owi_s = None
        self.ope_s = None
        self.nep_s = None
        self.nne_s = None
        self.fil_s = None

        # create LSTM models
        self.lstm_model_1 = Lstm(self.B, self.NUM_NEURONS)
        self.lstm_model_2 = Lstm(self.B, self.NUM_NEURONS)
        self.tmp_lstm_model_1 = Lstm(self.B, self.NUM_NEURONS)
        self.tmp_lstm_model_2 = Lstm(self.B, self.NUM_NEURONS)

        # create lists
        self.values = np.empty(self.length)
        self.predicted_1 = list()
        self.predicted_2 = list()
        self.AARE_1 = list()
        self.AARE_2 = list()
        self.threshold_1 = list()
        self.threshold_2 = list()
        self.anomaly_1 = list()
        self.anomaly_2 = list()
        self.anomaly_aggr = list()
        self.pattern_change_1 = list()
        self.pattern_change_2 = list()

        # aging
        self.window_beginning = 0

        self.DO_DIFF = None
        self.DO_SCAL = None

    def read_testrun_data(self):
        # reading parameters from the specified file
        if self.OPERATION == 'file':
            runs_data = pd.read_csv(self.TESTRUN_FOLDER + '/' + self.TESTRUN_FILE)
            self.b_s = runs_data['B']
            self.thr_s = runs_data['THRESHOLD_STRENGTH']
            self.usw_s = runs_data['USE_WINDOW']
            self.wis_s = runs_data['WINDOW_SIZE']
            self.usa_s = runs_data['USE_AGING']
            self.uaa_s = runs_data['USE_AARE_AGING']
            self.uta_s = runs_data['USE_THD_AGING']
            self.agp_s = runs_data['AGE_POWER']
            self.uap_s = runs_data['USE_AUTOMATIC_WS_AP']
            self.uoc_s = runs_data['USE_OFFSET_COMP']
            self.aad_s = runs_data['ACCEPTABLE_AVG_DURATION']
            self.uao_s = runs_data['USE_AUTOMATIC_OFFSET']
            self.owi_s = runs_data['OFFSET_WINDOW_SIZE']
            self.ope_s = runs_data['OFFSET_PERCENTAGE']
            self.nep_s = runs_data['NUM_EPOCHS']
            self.nne_s = runs_data['NUM_NEURONS']
            self.fil_s = runs_data['FILENAME']

            self.run_for = len(self.b_s)

            self.TO_CSV = True
            self.DO_DIFF = False
            self.DO_SCAL = False
            self.DO_DIV = False
            self.USE_LESS = False
            self.DEBUG = False

    def ingest_testrun_data(self, testrun_data: list):
        # reading parameters from the input
        self.b_s = [testrun_data[0]]
        self.thr_s = [testrun_data[1]]
        self.usw_s = [testrun_data[2]]
        self.wis_s = [testrun_data[3]]
        self.usa_s = [testrun_data[4]]
        self.uaa_s = [testrun_data[5]]
        self.uta_s = [testrun_data[6]]
        self.agp_s = [testrun_data[7]]
        self.uap_s = [testrun_data[8]]
        self.uoc_s = [testrun_data[9]]
        self.aad_s = [testrun_data[10]]
        self.uao_s = [testrun_data[11]]
        self.owi_s = [testrun_data[12]]
        self.ope_s = [testrun_data[13]]
        self.nep_s = [testrun_data[14]]
        self.nne_s = [testrun_data[15]]
        self.fil_s = ''

        self.run_for = 1

        self.OPERATION = 'file'
        self.TO_CSV = False
        self.DO_DIFF = False
        self.DO_SCAL = False
        self.DO_DIV = False
        self.USE_LESS = False
        self.DEBUG = False

        return

    from OnlineDetectors.AnDePeD.ReRe.initial import param_refresh, load, initialize_cons
    from OnlineDetectors.AnDePeD.ReRe.main_algo import initialize_rere, set_values, next_timestep, get_latest_anomaly
    from OnlineDetectors.AnDePeD.ReRe.offset_comp import init_offset_compensation, compensate_offset
    from OnlineDetectors.AnDePeD.ReRe.auto_offset_comp import init_auto_offset_compensation, auto_tune_offset
    from OnlineDetectors.AnDePeD.ReRe.timer import init_timer, start_timestep, end_timestep
    from OnlineDetectors.AnDePeD.ReRe.preprocess import preprocess
    from OnlineDetectors.AnDePeD.ReRe.auto_ws_ap import init_auto_ws_ap, auto_tune_ws_ap
    from OnlineDetectors.AnDePeD.ReRe.to_csv import init_to_csv, dump_hyperparameters, dump_results, write_time
    from OnlineDetectors.AnDePeD.ReRe.window_ageing import update_window_beginning, ageing_coefficient
    from OnlineDetectors.AnDePeD.ReRe.thd_aare_func import aare, thd_1, thd_2
