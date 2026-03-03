from OnlineDetectors.online_nab_detector import OnlineAnomalyDetector
from OnlineDetectors.contextOSE.cad_ose import ContextualAnomalyDetectorOSE


class ContextOSEDetector(OnlineAnomalyDetector):
    """
    This detector uses Contextual Anomaly Detector - Open Source Edition
    2016, Mikhail Smirnov   smirmik@gmail.com
    https://github.com/smirmik/CAD
    """

    def __init__(self, *args, **kwargs):
        super(ContextOSEDetector, self).__init__(*args, **kwargs)

        self.cadose = None

    def handleRecord(self, inputData):
        anomalyScore = self.cadose.getAnomalyScore(inputData)
        return (anomalyScore,)

    def initialize(self):
        self.cadose = ContextualAnomalyDetectorOSE(
            minValue=self.inputMin,
            maxValue=self.inputMax,
            restPeriod=self.probationaryPeriod / 5.0,
        )
