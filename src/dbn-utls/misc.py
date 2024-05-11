from torch.utils.data import Dataset
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
    
def decrease_labels_by_10(data):
    image, label = data
    label -= 10  # Sottrai 10 da ciascuna etichetta
    return image, label