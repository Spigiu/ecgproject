import csv
import numpy as np
import os.path
import torch
from scipy.io import loadmat, savemat


def assemble_data(labels_to_pick = {'NSR': 1, 'FA': 2, 'TACHY': 3, 'BRADY': 4}):
    datamat = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if os.path.isfile(datamat):
        print(f"Dataset already assembled in {os.path.dirname(__file__)}.")
    else:
       
        data = []
        labels = []

        with open(os.path.join(os.path.dirname(__file__), 'Data', 'FILEE.txt'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            for row in csv_reader:
                #print(row)
                if row[1] in labels_to_pick.keys():
                    with open(os.path.join(os.path.dirname(__file__), 'Data', row[0]), 'r') as datafile:
                        data_reader=csv.reader(datafile)
                        d = []
                        for val in data_reader:
                            d.append(float(val[0])) 
                    data.append(d)
                    labels.append(labels_to_pick[row[1]])
                    
        data = np.array(data, dtype=object)
        labels = np.array(labels)
        label_names = list(labels_to_pick.keys())
        savemat(datamat, {'data': data, 'labels': labels, 'label_names': label_names})


def load_data(t_len=15000):
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if ~os.path.isfile(filename):
        assemble_data()
    struct = loadmat(filename)
    data = struct['data'][0]
    samples = len(data)
    n_series = 1
    ts = np.zeros((samples, n_series, t_len), dtype=np.float32)
    for n in range(samples):
        if len(data[n][0]) >= t_len:
            ts[n, 0, :] = data[n][0][:t_len]
        else:
            ts[n, 0, :len(data[n][0])] = data[n][0]
    labels = struct['labels'][0]
    return torch.from_numpy(ts), torch.from_numpy(labels)


def get_label_names():
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if ~os.path.isfile(filename):
        assemble_data()
    struct = loadmat(filename)
    names = struct['label_names']
    return names
