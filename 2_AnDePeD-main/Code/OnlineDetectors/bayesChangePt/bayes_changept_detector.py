import numpy
from scipy import stats

from OnlineDetectors.online_nab_detector import OnlineAnomalyDetector


class BayesChangePtDetector(OnlineAnomalyDetector):

    def __init__(self, *args, **kwargs):

        super(BayesChangePtDetector, self).__init__(*args, **kwargs)

        # Set up the matrix that will hold our beliefs about the current
        # run lengths. We'll initialize it all to zero at first. For efficiency,
        # we preallocate a data structure to hold only the info we need to detect
        # change points: columns for the current and next recordNumber, and a
        # sufficient number of rows (where each row represents probabilites of a
        # run of that length).
        self.maxRunLength = 500
        self.runLengthProbs = numpy.zeros((self.maxRunLength + 2, 2))

        # Record 0 is a boundary condition, where we know the run length is 0.
        self.runLengthProbs[0, 0] = 1.0

        # Init variables for state.
        self.recordNumber = 0
        self.previousMaxRun = 1

        # Define algorithm's helpers.
        self.observationLikelihoood = StudentTDistribution(alpha=0.1, beta=0.001, kappa=1.0, mu=0.0)
        self.lambdaConst = 250
        self.hazardFunction = constantHazard

    def handleRecord(self, inputData):
        """
        Returns a list [anomalyScore]. Algorithm details are in the comments.
        """
        # To accommodate this next record, shift the columns of the run length probabilities matrix.
        if self.recordNumber > 0:
            self.runLengthProbs[:, 0] = self.runLengthProbs[:, 1]
            self.runLengthProbs[:, 1] = 0

        # Evaluate the predictive distribution for the new datum under each of the parameters.
        # This is standard Bayesian inference.
        predProbs = self.observationLikelihoood.pdf(inputData["value"])

        # Evaluate the hazard function for this interval
        hazard = self.hazardFunction(self.recordNumber + 1, self.lambdaConst)

        # We only care about the probabilites up to maxRunLength.
        runLengthIndex = min(self.recordNumber, self.maxRunLength)

        # Evaluate the growth probabilities -- shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive probabilities.
        self.runLengthProbs[1:runLengthIndex + 2, 1] = (
                self.runLengthProbs[:runLengthIndex + 1, 0] *
                predProbs[:runLengthIndex + 1] *
                (1 - hazard)[:runLengthIndex + 1]
        )

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the probability mass back down at run length = 0.
        self.runLengthProbs[0, 1] = numpy.sum(
            self.runLengthProbs[:runLengthIndex + 1, 0] *
            predProbs[:runLengthIndex + 1] *
            hazard[:runLengthIndex + 1]
        )

        # Renormalize the run length probabilities for improved numerical stability.
        self.runLengthProbs[:, 1] = (self.runLengthProbs[:, 1] /
                                     self.runLengthProbs[:, 1].sum())

        # Update the parameter sets for each possible run length.
        self.observationLikelihoood.updateTheta(inputData["value"])

        # Get the current run length with the highest probability.
        maxRecursiveRunLength = self.runLengthProbs[:, 1].argmax()

        # To calculate anomaly scores from run length probabilites we have several
        # options, implemented below:
        #   1. If the max probability for any run length is the run length of 0, we
        #   have a changepoint, thus anomaly score = 1.0.
        #   2. The anomaly score is the probability of run length 0.
        #   3. Compute a score by assuming a change in sequence from a previously
        #   long run is more anomalous than a change from a short run.
        # Option 3 results in the best anomaly detections (by far):
        if maxRecursiveRunLength < self.previousMaxRun:
            anomalyScore = 1 - (float(maxRecursiveRunLength) / self.previousMaxRun)
        else:
            anomalyScore = 0.0

        # Update state vars.
        self.recordNumber += 1
        self.previousMaxRun = maxRecursiveRunLength

        return [anomalyScore]


def constantHazard(arraySize, lambdaConst):
    """ The hazard function helps estimate the changepoint prior. Parameter
  lambdaConst is the timescale on the prior distribution of the changepoint.
  """
    return numpy.ones(arraySize) / float(lambdaConst)


class StudentTDistribution:

    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = numpy.array([alpha])
        self.beta0 = self.beta = numpy.array([beta])
        self.kappa0 = self.kappa = numpy.array([kappa])
        self.mu0 = self.mu = numpy.array([mu])

    def pdf(self, data):
        """ Probability density function for the Student's T continuous random
    variable. More details here:
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """
        return stats.t.pdf(x=data,
                           df=2 * self.alpha,
                           loc=self.mu,
                           scale=numpy.sqrt((self.beta * (self.kappa + 1)) /
                                            (self.alpha * self.kappa))
                           )

    def updateTheta(self, data):
        """ Update parameters of the distribution."""
        muT0 = numpy.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = numpy.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = numpy.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = numpy.concatenate((self.beta0, self.beta + (self.kappa * (data -
                                                                           self.mu) ** 2) / (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
