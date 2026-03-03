# ----------------------------------------------------------------------
# Originally by NAB, modified to fit online operation.
# ----------------------------------------------------------------------

import abc
from datetime import datetime, timedelta


class OnlineAnomalyDetector(object, metaclass=abc.ABCMeta):
    """
    Base class for all anomaly detectors. When inheriting from this class please
    take note of which methods MUST be overridden, as documented below.
    """

    def __init__(self, input_min, input_max):

        # self.dataSet = dataSet
        self.probationaryPeriod = 750  # 0.15 * 5000 is the hard programmed limit by NAB (750 points have to be enough)

        self.inputMin = input_min
        self.inputMax = input_max

        self.time = 0  # number of timesteps since initialisation

    def initialize(self):
        """
        Do anything to initialize your detector in before calling run.
        Pooling across cores forces a pickling operation when moving objects from
        the main core to the pool and this may not always be possible. This function
        allows you to create objects within the pool itself to avoid this issue.
        """
        pass

    def getAdditionalHeaders(self):
        """
        Returns a list of strings. Subclasses can add in additional columns per record.
        This method MAY be overridden to provide the names for those columns.
        """
        return []

    @abc.abstractmethod
    def handleRecord(self, inputData):
        """
        Returns a list [anomalyScore, *]. It is required that the first element of the list is the anomalyScore.
        The other elements may be anything, but should correspond to the names returned by getAdditionalHeaders().
        This method MUST be overridden by subclasses
        """
        raise NotImplementedError

    def getHeader(self):
        """
        Gets the outputPath and all the headers needed to write the results files.
        """
        headers = ['timestamp', 'value', 'anomaly_score']
        headers.extend(self.getAdditionalHeaders())
        return headers

    def next_timestep(self, new_value: float):
        """
        Main function that is called to collect anomaly scores for a given file.
        """

        # timestamps begin from 2000.01.01. and are incremented every 5 minutes
        begin_dt = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        increment_dt = timedelta(minutes=5)
        current_dt = begin_dt + self.time * increment_dt

        inputData = {  # detectors expect a dictionary with the key 'value' containing the data
            'timestamp': current_dt,
            'value': new_value
        }

        detectorValues = self.handleRecord(inputData)  # call the individual function of the detector

        # Make sure anomalyScore is between 0 and 1
        if not 0 <= detectorValues[0] <= 1:
            raise ValueError(
                f"anomalyScore must be a number between 0 and 1. "
                f"Please verify if '{self.handleRecord.__qualname__}' method is "
                f"returning a value between 0 and 1")

        self.time += 1

        return detectorValues[0]
