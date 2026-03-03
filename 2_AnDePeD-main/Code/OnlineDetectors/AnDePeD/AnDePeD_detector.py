import gc

import main_config as conf
import online_buffer as obuff
from OnlineDetectors.online_nab_detector import OnlineAnomalyDetector
from OnlineDetectors.AnDePeD.ReRe import ReRe


class ANDEPEDDetector(OnlineAnomalyDetector):

    def __init__(self, buffer_size: int, algorithm: str, *args, **kwargs):
        super(ANDEPEDDetector, self).__init__(*args, **kwargs)

        self.algorithm = algorithm

        self.buffer = obuff.CircularBuffer(buffer_size)
        self.rere = ReRe.ReRe()

    def initialize(self):
        if self.algorithm == 'AnDePeDPro':
            self.rere.ingest_testrun_data(conf.ANDEPED_PRO_TESTRUN_PARAMETERS)
        else:
            self.rere.ingest_testrun_data(conf.ANDEPED_TESTRUN_PARAMETERS)

        self.rere.init_timer()
        self.rere.init_offset_compensation()
        self.rere.init_auto_offset_compensation()
        self.rere.init_auto_ws_ap()

        self.rere.param_refresh(0)
        self.rere.inst_num = 0
        self.rere.progress_for_executions = {}
        self.rere.executions_max = {}
        self.rere.BATCH_STATUS_BAR = False
        self.rere.DEBUG = False

        self.rere.initialize_cons()
        self.rere.preprocess()

        self.rere.initialize_rere()

        return

    def handleRecord(self, inputData):
        # add new element to buffer, and import it into ReRe
        new_value = inputData['value']
        self.buffer.add_item(new_value)
        self.rere.set_values(self.buffer.get_all_items())

        # start time measurement
        self.rere.start_timestep()

        # update the beginning of the sliding window
        self.rere.update_window_beginning(self.time)

        # perform one timestep of the original ReRe algorithm
        self.rere.next_timestep(self.time)

        # perform offset compensation
        if self.rere.USE_OFFSET_COMP:
            self.rere.compensate_offset(self.time)

        # perform automatic tuning of offset compensation
        if self.rere.USE_AUTOMATIC_OFFSET:
            self.rere.auto_tune_offset(self.time)

        # perform automatic tuning of WINDOW_SIZE and AGE_POWER
        if self.rere.USE_AUTOMATIC_WS_AP:
            self.rere.auto_tune_ws_ap(self.time)

        # stop time measurement and update averages
        self.rere.end_timestep(self.time)

        # get anomaly signal just detected
        anomaly_now = self.rere.get_latest_anomaly()
        anomaly_score = 1.0 if anomaly_now else 0.0

        gc.collect()

        return [anomaly_score]
