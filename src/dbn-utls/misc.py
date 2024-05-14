import numpy as np
import pandas as pd
import torch
from google.colab import files
from torch.utils.data import Dataset
import math

#function for computing SEM
def SEM(measure):
    nr_of_measures = len(measure)
    if not(isinstance(measure, np.ndarray)):
        measure = np.asarray(measure)
    sem = np.std(measure)/math.sqrt(nr_of_measures)
    return sem

def save_mat_xlsx(my_array, filename='my_res.xlsx'):
    # create a pandas dataframe from the numpy array
    my_dataframe = pd.DataFrame(my_array)
    # save the dataframe as an excel file
    my_dataframe.to_excel(filename, index=False)
    # download the file
    files.download(filename)


def reshape_data(train_ds, batch_sz_real, batch_sz):
    n = batch_sz//batch_sz_real # Designed for 128 and 64 only
    #if the nr of rows of train_ds['data'] is not divisible by n...
    if not(train_ds['data'].shape[0]%n==0): 
        #remove a row from the data and label arrays
        train_ds['data'] = train_ds['data'][:-1,:,:]
        train_ds['labels'] = train_ds['labels'][:-1,:,:]
    #reshape both the data and labels array so that the batch size is now the desired one (i.e. BATCH_SIZE)
    train_ds['data'] = train_ds['data'].view(train_ds['data'].shape[0]//n, batch_sz, train_ds['data'].shape[2])
    train_ds['labels'] = train_ds['labels'].view(train_ds['labels'].shape[0]//n, batch_sz, train_ds['labels'].shape[2])
    return train_ds


class MyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        # Initialize the dataset with features, labels, and an optional transform function
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Define the behavior when len() is called on an instance of MyDataset
        return len(self.features)

    def __getitem__(self, idx):
        # Define the behavior when an item is accessed using indexing
        feature = self.features[idx]
        label = self.labels[idx]

        # apply transformations if specified
        if self.transform:
            feature = self.transform(feature)

        return feature, label
    
def relabel_09(data, old_labels):
    old_labels = sorted(old_labels)
    #relabels the data from the indices sorted_list to 0-9
    image, label = data
    return image, old_labels.index(label)

def get_relative_freq(valore, hist, bin_edges,numero_bin=20):
    # Trova il bin in cui si trova il valore
    indice_bin = np.digitize(valore, bin_edges)

    # Controlla se l'indice è fuori dai limiti
    if 1 <= indice_bin <= numero_bin:
        frequenza_relativa = hist[indice_bin - 1]
        return frequenza_relativa
    else:
        return 0.0  # Il valore è al di fuori dei bin

def raddrizza_lettere(data_train_retraining_ds,data_test_retraining_ds):
    #questi loop sono per raddrizzare le lettere
    data_train_retraining_L = []
    for item in data_train_retraining_ds:
        image = item[0].view(28, 28)
        image = torch.rot90(image, k=-1)
        image = torch.flip(image, [1])
        data_train_retraining_L.append((image,item[1]))
    data_test_retraining_L = []
    for item in data_test_retraining_ds:
        image= item[0].view(28, 28)
        image = torch.rot90(image, k=-1)
        image = torch.flip(image, [1])
        data_test_retraining_L.append((image,item[1]))
    data_test_retraining_ds = data_test_retraining_L
    data_train_retraining_ds = data_train_retraining_L