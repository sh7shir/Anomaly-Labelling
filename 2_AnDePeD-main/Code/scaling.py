import numpy as np
from sklearn.preprocessing import MinMaxScaler

# scaling parameters
SCALE_MIN = -1
SCALE_MAX = 1
MODE = 'minmax'


def preprocess(data, mode=MODE, out_range=(SCALE_MIN, SCALE_MAX)):
    data = np.asarray(data)

    if mode == 'minmax':
        scaler = MinMaxScaler(feature_range=out_range)
        data = data.reshape(-1, 1)

    scaler.fit(data)
    new_data = scaler.transform(data)
    new_data = new_data.flatten()

    return new_data
