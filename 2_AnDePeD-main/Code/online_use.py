import numpy as np
import pandas as pd

import read_files

import online_buffer as obuff
import scaling
import vmd

from OnlineDetectors.bayesChangePt.bayes_changept_detector import BayesChangePtDetector
from OnlineDetectors.windowedGaussian.windowedGaussian_detector import WindowedGaussianDetector
from OnlineDetectors.relativeEntropy.relative_entropy_detector import RelativeEntropyDetector
from OnlineDetectors.earthgeckoSkyline.earthgecko_skyline_detector import EarthgeckoSkylineDetector
from OnlineDetectors.contextOSE.context_ose_detector import ContextOSEDetector
from OnlineDetectors.knncad.knncad_detector import KnncadDetector
from OnlineDetectors.AnDePeD.AnDePeD_detector import ANDEPEDDetector


class OnlineProcedure:
    """
    This class is responsible for executing online pre-processing and anomaly detection using a circular buffer.
    """

    def __init__(self, mode: str, data_parameters: list, algorithm: str, dataset: str, initial_data: list):
        """
        Initialise online pre-processing and anomaly detection.
        :param mode: 'I' or 'II'
        :param data_parameters: [L(length of anom.det. array), alpha_star, k_star, l_vmd, modes_star_path, min, max]
        :param algorithm: name of anomaly detector
        :param dataset: any string to reference the data by
        :param initial_data: to load the buffer with
        """

        self.mode = mode

        (self.l, self.alpha_star, self.k_star, self.l_vmd, self.modes_star_path,
         self.data_min, self.data_max) = data_parameters

        self.buffer = obuff.CircularBuffer(self._get_buffer_size())
        self.buffer.load(initial_data)

        self.algorithm = algorithm
        self.dataset = dataset
        self.detector = self.initialise_online_detector()

        self.save_data = pd.DataFrame(columns=['algorithm', 'dataset', 'timestep', 'orig_value',
                                               'remainder_value', 'anomaly_score'])

        self.time = 0

        return

    def next_timestep(self, new_value: float):
        # (1) add data to buffer
        self.buffer.add_item(new_value)

        # (2) scale data
        values = self.buffer.get_all_items()
        scaled_values = scaling.preprocess(values)

        # (3) ...
        if self.mode == 'I':
            remainder_values = self._mode_i_next_timestep(scaled_values)
        elif self.mode == 'II':
            remainder_values = self._mode_ii_next_timestep(scaled_values)
        else:
            remainder_values = [-1]

        # (4) feed the new value to the anomaly detector
        new_remainder = remainder_values[-1]
        anom_score = self.detector.next_timestep(new_remainder)

        # (5) save results for later analysis
        self.save_data.loc[len(self.save_data)] = [self.algorithm, self.dataset, self.time,
                                                   new_value, new_remainder, anom_score]

        self.time += 1

        return

    def _mode_i_next_timestep(self, scaled_values: np.ndarray):
        # (3.1) VMD + mode removal using (alpha_star, K_star)
        remainder_values, _, _ = vmd.decompose(scaled_values, self.alpha_star, self.k_star)

        # vmd.decompose returns a 2D numpy array (time, data), where the second column is needed
        remainder_values = remainder_values[:, 1]

        # (3.2) cut to newest L values for anomaly detector
        remainder_values = remainder_values[-self.l:]
        return remainder_values

    def _mode_ii_next_timestep(self, scaled_values: np.ndarray):
        # (3) only mode removal using pre-computed modes
        modes_star = self.read_and_extend_modes_star_to_given_length(len(scaled_values))
        remainder_values = scaled_values - modes_star
        return remainder_values

    def _get_buffer_size(self):
        to_ret = -1
        if self.mode == 'I':
            to_ret = self.l_vmd
        if self.mode == 'II':
            to_ret = self.l
        if self.mode in ['I', 'II']:
            return to_ret + to_ret % 2
        else:
            return -1

    def initialise_online_detector(self):
        if self.algorithm == 'bayesChangePt':
            det = BayesChangePtDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'windowedGaussian':
            det = WindowedGaussianDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'relativeEntropy':
            det = RelativeEntropyDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'earthgeckoSkyline':
            det = EarthgeckoSkylineDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'contextOSE':
            det = ContextOSEDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'knncad':
            det = KnncadDetector(input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'AnDePeDPro':
            det = ANDEPEDDetector(buffer_size=100000, algorithm='AnDePeDPro',
                                  input_min=self.data_min, input_max=self.data_max)

        elif self.algorithm == 'AnDePeD':
            det = ANDEPEDDetector(buffer_size=100000, algorithm='AnDePeD',
                                  input_min=self.data_min, input_max=self.data_max)

        else:
            return -1

        # call the children's initialisation functions
        det.initialize()

        return det

    def export_saved_data(self, filepath: str):
        self.save_data.to_csv(filepath)
        return

    def read_and_extend_modes_star_to_given_length(self, desired_length: int):
        ms_orig = read_files.read_file_pandas(self.modes_star_path, column='value', to_numpy=True)
        ms_new = np.empty(shape=0)

        while len(ms_new) + len(ms_orig) < desired_length:
            ms_new = np.append(ms_new, ms_orig)

        ms_new = np.append(ms_new, ms_orig[ : desired_length - len(ms_new)])

        return ms_new

