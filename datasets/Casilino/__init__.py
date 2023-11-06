import csv
import os
from scipy.io import savemat, loadmat
import numpy as np
import torch
from util import MultiNotch_50Hz_60Hz,BandPassFilter1_3_30Hz

def assemble_data(labels_to_pick={'NSR': 0}, k=15000):
    datamat = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if os.path.isfile(datamat):
        print(f"Dataset already assembled in {os.path.dirname(__file__)}.")
    else:
        data = []
        labels = []

        with open(os.path.join(os.path.dirname(__file__), 'Data', 'FILEE.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if row[1] in labels_to_pick.keys():
                    with open(os.path.join(os.path.dirname(__file__), 'Data', row[0]), 'r') as datafile:
                        data_reader = csv.reader(datafile)
                        d = []
                        for val in data_reader:
                            d.append(float(val[0]))
                        
                        # Modifica:
                           # for i in range(0, len(d) -k  + 1, k):
                               # segment = d[i:i + k]
                             #   data.append(segment)
                             #   labels.append(labels_to_pick[row[1]])
                           # print(data[3])
                    data.append(d)
                    labels.append(labels_to_pick[row[1]])
        data = np.array(data, dtype=object)
        
        for i in range(len(data)):
            data[i] = MultiNotch_50Hz_60Hz(data[i])
            data[i] = BandPassFilter1_3_30Hz(data[i])
    
    
    
        # Vado a fare il filtraggio
      #  data=float(data[0])
       # data = MultiNotch_50Hz_60Hz(data)
       # data = BandPassFilter1_3_30Hz(data)
        
        
        labels = np.array(labels)
        label_names = list(labels_to_pick.keys())
        savemat(datamat, {'data': data, 'labels': labels, 'label_names': label_names})
        print(labels)



def load_data(t_len=15000, labels_to_pick={'NSR': 0}, k=15000): #(beat_len)
    
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if ~os.path.isfile(filename):
        assemble_data(k=15000)
    struct = loadmat(filename)
    data = struct['data'][0]
    samples = len(data)
    n_series = 1
    
    
    labels = struct['labels'][0]
   # selected_indices = [i for i, label in enumerate(labels) if label in labels_to_pick.values()]
   # data = [data[i] for i in selected_indices]
    #labels = [labels[i] for i in selected_indices]
    samples = len(data)
    
    
  
    ts = np.zeros((samples, n_series,t_len), dtype=np.float32)
    for n in range(samples):  
        if len(data[n][0])>=t_len:
            ts[n,0, :t_len] = data[n][0][:t_len]
                         
        else:
            ts[n, 0, :len(data[n][0])] = data[n][0]
     
    
    labels = struct['labels'][0]
                         
    return torch.from_numpy(ts), torch.from_numpy(labels).long()






def get_label_names():
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if not os.path.isfile(filename):
        assemble_data(labels)
    struct = loadmat(filename)
    names = struct['label_names']
    return names


def assemble_data1(labels_to_pick={'FA': 1}, k=250):
    datamat = os.path.join(os.path.dirname(__file__), 'dataset1.mat')
    if os.path.isfile(datamat):
        print(f"Dataset already assembled in {os.path.dirname(__file__)}.")
    else:
        data = []
        labels = []

        with open(os.path.join(os.path.dirname(__file__), 'Data', 'FILEE.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if row[1] in labels_to_pick.keys():
                    with open(os.path.join(os.path.dirname(__file__), 'Data', row[0]), 'r') as datafile:
                        data_reader = csv.reader(datafile)
                        d = []
                        for val in data_reader:
                            d.append(float(val[0]))
                            #d = [float(val[0]) for val in data_reader]

                        ## Modifica:
                       # for i in range(0, len(d) - k + 1, k):
                        #    segment = d[i:i + k]
                         #   data.append(segment)
                          #  labels.append(labels_to_pick[row[1]])
                               # print(data)
                    data.append(d)
                    labels.append(labels_to_pick[row[1]])

        data = np.array(data, dtype=object)
        for i in range(len(data)):
            data[i] = MultiNotch_50Hz_60Hz(data[i])
            data[i] = BandPassFilter1_3_30Hz(data[i])
        
        # Vado a fare il filtraggio
      #  data = float(data[0])
       # data = MultiNotch_50Hz_60Hz(data)
       # data = BandPassFilter1_3_30Hz(data)
        
        
        
        labels = np.array(labels)
        label_names = list(labels_to_pick.keys())
        savemat(datamat, {'data': data, 'labels': labels, 'label_names': label_names})
        print(labels)
        



def load_data1(t_len=15000, labels_to_pick={'FA': 1}, k=15000): #(beat_len)
    
    filename = os.path.join(os.path.dirname(__file__), 'dataset1.mat')
    if ~os.path.isfile(filename):
        assemble_data1(k=15000)
    struct = loadmat(filename)
    data = struct['data'][0]
    samples = len(data)
    n_series = 1
    
    
    labels = struct['labels'][0]
   # selected_indices = [i for i, label in enumerate(labels) if label in labels_to_pick.values()]
    #data = [data[i] for i in selected_indices]
   # labels = [labels[i] for i in selected_indices]
    samples = len(data)
    
    
  
    ts = np.zeros((samples, n_series,t_len), dtype=np.float32)
    for n in range(samples):  
        if len(data[n][0])>=t_len:
            ts[n,0, :t_len] = data[n][0][:t_len]
                         
        else:
            ts[n, 0, :len(data[n][0])] = data[n][0]
     
    
    labels = struct['labels'][0]
                         
    return torch.from_numpy(ts), torch.from_numpy(labels).long()






def get_label_names1():
    filename = os.path.join(os.path.dirname(__file__), 'dataset.mat')
    if not os.path.isfile(filename):
        assemble_data1(labels)
    struct = loadmat(filename)
    names = struct['label_names']
    return names




