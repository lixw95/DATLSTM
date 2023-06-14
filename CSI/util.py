import pandas as pd
import os
import numpy as np
import numpy
from sklearn.preprocessing import MinMaxScaler
def load_train_dataset(year):
    main_path = r"/home/lxw/TCN -attn/train_set220_knn"
    file_name = '{}.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values
    print(data.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    x1 = data[:, 0:11]
    x2 = data[:, 12:]
    x = np.hstack((x1, x2))
    y = data[:, 11]
    y = y.reshape(-1, 1)
    input_x = np.hstack((x, y))
    return numpy.array(input_x, dtype=np.float32), numpy.array(y, dtype=np.float32)
def load_test_dataset(year):
    main_path = r"/home/lxw/TCN -attn/test_dataset"
    file_name = '{}.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path)
    data = df.iloc[:, 1:].values
    x1 = data[:, 0:11]
    x2 = data[:, 12:]
    x = np.hstack((x1, x2))
    y = data[:, 11]
    y = y.reshape(-1, 1)
    input_x = np.hstack((x, y))
    return numpy.array(input_x, dtype=np.float32), numpy.array(y, dtype=np.float32)

if __name__ == '__main__':
    load_train_dataset(1)