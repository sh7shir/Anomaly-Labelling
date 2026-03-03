# AnDePeD

Accompanying repository for our paper on real-time anomaly detection on periodic
server farm telemetry data. We introduce our new algorithms; AnDePeD and AnDePeD Pro.

## Code

This folder contains the following real-time, unidimensional anomaly detection
algorithm implementations.

#### Our novel methods

- AnDePeD;
- AnDePeD Pro.

#### Our pre-existing methods

- Alter-Re²;
- AREP.

#### Sourced from [NAB](https://github.com/numenta/NAB) for comparison
 
- Bayesian Changepoint;
- Windowed Gaussian;
- Relative Entropy;
- Earthgecko Skyline;
- KNN CAD.

We also include the data files present in the **Data** folder here for ease of use.

#### Installation

We used `Python 3.9.2` for running our experiments with package versions included
in `requirements.txt`. Packages can be installed system-wide or in a virtual
environment using the `pip install -r requirements.txt`command.
For more specific details on setting up individual algorithms from NAB,
refer to the respective README files therein:
[.../nab/detectors](https://github.com/numenta/NAB/tree/master/nab/detectors).


#### Running algorithms

Parameters of the experiment can be set in `main_config.py`, while
[Optuna](https://optuna.org/) optimisation parameters can be set in
`optimiser_config.py`. Run `main.py` once parameters are set.

## Data

### Offline

This folder contains seven datasets from the [NAB data corpus](https://github.com/numenta/NAB)
and corresponding anomaly labels in the file `combined_labels.json`. They are all periodic
datasets from the server farm telemetry domain. We chose these seven datasets as the
pre-collected offline set that our pre-processing procedure is trained on.

### Online

To simulate pre-existing offline data and streaming online data from network devices, we first
carefully removed all anomalies from the original seven periodic datasets, created a copy of
them three times, then reintroduced the anomalies we had removed into each of the three copies
at randomised positions. This had to be done differently for each dataset, as the varied nature
of anomalies offered limited generalisation. As a result, we achieved a dataset three times
longer, with similar anomalies at different, random places. In the case where multiple different
normal sections had been present in the original data, we also randomised the order of those
segments.

This folder contains the results of this randomisation via four files for each of the seven datasets:
- `<dataset name>_origdata.csv`: original NAB dataset;
- `<dataset name>_origlabels.csv`: anomaly labels of original NAB dataset;
- `<dataset name>_adddata-<date id>.csv`: randomised, three times longer dataset we generated;
- `<dataset name>_addlabels-<date id>.csv`: anomaly labels for the randomised,
three times longer dataset we generated.

We chose the original NAB datasets as the offline collected portion used for preparation,
and utilised the newly created, randomised, triple-length dataset for simulating
online streaming data.

## Results

This folder contains the numerical results that are shown in the paper as figures. 

- `anomaly_detection_performance_results.csv`: average results of the five detectors from 
NAB, Alter-Re², AREP, AnDePeD and AnDePeD Pro (latter two in both online
operational modes: Mode-I and Mode-II), using the metrics of
Precision, Recall, F-score and MCC on the datasets in the **Data** folder  
(higher values are better, value range is $[0,1]$ for all four metrics);

- `initialisation_delays.csv`: results of our initialisation delay calculations for the
five detectors from NAB, Alter-Re², AREP, AnDePeD and AnDePeD Pro,
based on their respective publications or implementations  
(lower values are better, value range is $[0, \infty)$ in theory and $[0, 4621]$ in practice);

- `detection_delay_results_Mode-I.csv`: average, minimum, maximum and standard deviation results of detection delays by
the five detectors from NAB, Alter-Re², AREP, AnDePeD and AnDePeD Pro (latter two in Mode-I)  
(lower values are better, value range is $[0, \infty)$ in theory and $[0, 4621]$ in practice).
