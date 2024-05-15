
from copy import deepcopy
import os
from typing import cast
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from skimage.filters import threshold_sauvola
from tqdm import tqdm 
from os import path 
import json
from pathlib import Path


from misc import relabel_09, reshape_data, MyDataset
from dbns import *
from Classifiers import *

def data_and_labels(data_train: Dataset, BATCH_SIZE: int,
                    NUM_FEAT: int,DATASET_ID: str,n_cols_labels: int):
    """
    Prepare the data for processing by machine learning models.
    (NOTE FOR THE WRITER: Be more specific)
    Args:
        data_train (Dataset): The training dataset.
        BATCH_SIZE (int): The batch size for the DataLoader.
        NUM_FEAT (int): The number of features in the data.
        DATASET_ID (str): The identifier for the dataset.
        n_cols_labels (int): The number of columns in the labels.

    Returns:
        tuple: A tuple containing the training data and labels.
            - train_data (Tensor): The training data.
            - train_labels (Tensor): The training labels.
    """
    
    d = 'cuda' if torch.cuda.is_available() else 'cpu' 
    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, generator=torch.Generator(device=d)) #create a dataloader with the data shuffled
    #shuffle=False impedisce al DataLoader di mescolare casualmente i dati durante l'addestramento, 
    #consentendo di iterare attraverso i dati nell'ordine originale in cui sono stati forniti.
    #Questo è stato fatto perchè alternativamente da un Type error che non sono riuscito a risolvere
    # Calculate the total number of batches
    num_batches = data_train.__len__() // BATCH_SIZE 
    # Create empty tensors to store the training data and labels
    train_data = torch.empty(num_batches, BATCH_SIZE, NUM_FEAT)
    train_labels = torch.empty(num_batches, BATCH_SIZE, n_cols_labels)
    with tqdm(train_loader, unit = 'Batch') as tdata: 
        #unit='Batch': Specifies the unit of measurement displayed by the progress bar
        #Inside the with block, you typically have a loop that iterates over train_loader, 
        #and tqdm will automatically update and display the progress bar as the loop progresses.
        for idx, (batch, labels) in enumerate(tdata):
            tdata.set_description(f'Train Batch {idx}\t') # set a description for the progress bar.
            if idx<num_batches: # Check if the current batch index is within the number of batches
                bsize = batch.shape[0] # Get the batch size of the current batch
                if DATASET_ID =='MNIST' or DATASET_ID =='EMNIST':
                    #reshape images into vectors and change the type of the elements to torch.float32
                    train_data[idx,:,:] = batch.reshape(bsize, -1).type(torch.float32) 
                else:
                    gray_batch = batch.mean(dim=1, keepdim=True) # convert to grayscale
                    for i in range(bsize): #for every element in the batch...
                        #convert the image to numpy and eliminate dimensions = 1
                        img_to_store = gray_batch[i].numpy().squeeze() 
                        if 'BW' in DATASET_ID: #if the DATASET_ID includes the BW letters (that stand for 'Black and White')...
                            #...apply the Sauvola-Pietikainen algorithm for binarization. 
                            #Parameters follow the paper https://link.aps.org/doi/10.1103/PhysRevX.13.021003
                            threshold = threshold_sauvola(img_to_store,window_size=7, k=0.05) 
                            img_to_store = img_to_store > threshold
                        #convert the np array into torch tensor (as a 1d vector (-> reshape))
                        train_data[idx, i, :] = torch.from_numpy(img_to_store.reshape(-1).astype(np.float32)) 
                if len(labels.shape)==1: #if your labels have just 1 dimension...
                    labels = labels.unsqueeze(1) #... then add 1 dimension
                train_labels[idx, :, :] = labels.type(torch.float32) #store also labels as torch.float32
    return train_data, train_labels 

def load_data_ZAMBRA(ds_id: str,batch_sz: int ,Zambra_folder_drive: str):
    """
    This function summarizes the loading data into the Zambra repository, 
    avoiding issues and errors.

    Args:
        ds_id (str): The dataset ID.
        batch_sz (int): The desired batch size.
        Zambra_folder_drive (str): The path to the Zambra folder drive.

    Returns:
        tuple: A tuple containing the train dataset and test dataset.

    Raises:
        ValueError: If the dataset ID is not recognized.
    """
    n_cols_labels = 1 # Initialize the number of label columns (e.g. 1 for MNIST)
    # Determine the number of features based on DATASET_ID
    
    if ds_id =='MNIST' or ds_id =='EMNIST':
        NUM_FEAT= np.int32(28*28)
    elif ds_id =='CIFAR10':
        NUM_FEAT= np.int32(32*32)
    elif 'CelebA' in ds_id:
        NUM_FEAT= np.int32(64*64)
        n_cols_labels = 40
    else:
        raise ValueError(f'Dataset {ds_id} not recognized')
    
    # Create names for test and training data files
    train_path= path.join(Zambra_folder_drive,'dataset_dicts',f'train_dataset_{ds_id}.npz')
    test_path = path.join(Zambra_folder_drive,'dataset_dicts',f'test_dataset_{ds_id}.npz')

    # If the training file exists, load the data (both train and test)
    if path.exists(train_path):
        train_dataset = dict(np.load(train_path))
        test_dataset = dict(np.load(test_path))
        # Convert the numpy arrays to torch tensors
        for key in train_dataset:
            train_dataset[key] = torch.from_numpy(train_dataset[key])
            test_dataset[key]= torch.from_numpy(test_dataset[key])
        # Calculate the actual BATCH_SIZE of the training data  
        batch_sz_real = train_dataset['data'].shape[1] 
        #if there is a mismatch between the actual batch size and the desired batch size (i.e. BATCH_SIZE)...
        if not(batch_sz_real ==batch_sz):
            #...then ask the user if he wants to reshape the data to the desired batch size
            reshape_yn = int(input('data found with batchsize '+str(batch_sz_real)+ 
                        '.Reshape it to the desired batchsize '+str(batch_sz)+'? (1=y,0=n)'))
            if reshape_yn==1: # if the user asks for reshape...
                #...reshape both the train and test dataset
                train_dataset = reshape_data(train_dataset, batch_sz_real, batch_sz)
                test_dataset = reshape_data(test_dataset, batch_sz_real, batch_sz)
    # If the training file does not exist, load the data from scratch based on DATASET_ID
    else:
        if ds_id =='MNIST':
            transform =transforms.Compose([transforms.ToTensor()])
            data_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
            data_test = datasets.MNIST('../data', train=False, download=True, transform=transform)

        elif ds_id =='EMNIST':
            transform =transforms.Compose([transforms.ToTensor()])
            data_train = datasets.EMNIST('../data', train=True,split = 'byclass', download=True, transform=transform)
            data_test = datasets.EMNIST('../data', train=False,split = 'byclass', download=True, transform=transform)
            #NOTA: il labelling dell'EMNIST by class ha 62 labels: le cifre (0-9), lettere MAUSCOLE (10-36), lettere MINUSCOLE(38-62)
            #In the Zambra paper they use 20 uppercase letters from the first 10 EMNIST classes.
            target_classes = list(range(10, 20))
            data_train = [relabel_09(item, target_classes) for item in data_train if item[1] in target_classes]
            data_test = [relabel_09(item, target_classes) for item in data_test if item[1] in target_classes]  
        elif ds_id=='CIFAR10':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()])
            data_train = datasets.CIFAR10(root='../data', split='train', download=True, transform=transform)
            data_test = datasets.CIFAR10(root='../data', split='test', download=True, transform=transform)   

        elif 'CelebA' in ds_id:
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                transforms.ToTensor()])
            data_train = datasets.CelebA(root='../data', split='train', download=True, transform=transform)
            data_test = datasets.CelebA(root='../data', split='test', download=True, transform=transform)    
        train_data, train_labels = data_and_labels(data_train, batch_sz,NUM_FEAT,ds_id,n_cols_labels)
        test_data, test_labels = data_and_labels(data_test, batch_sz,NUM_FEAT,ds_id,n_cols_labels)  
        train_dataset = {'data': train_data, 'labels': train_labels}
        test_dataset = {'data': test_data, 'labels': test_labels}
        #If the directory already exists, it does nothing (thanks to "exist_ok=True")
        Path(path.join(Zambra_folder_drive,'dataset_dicts')).mkdir(exist_ok=True)  
        #This is a common way to save multiple arrays or data structures into a single archive file for easy storage and later retrieval.
        #'**' unpacks the dictionary and passes its contents to the function.
        np.savez(train_path, **train_dataset)
        np.savez(test_path, **test_dataset)

    return train_dataset, test_dataset

def Multiclass_dataset(train_ds, selected_idx = [20,31], for_classifier = False, Old_rbm=False, DEVICE ='cuda'):
    batch_sz = train_ds['data'].shape[1]
    #Deep copy and move training data to the specified device (e.g., 'cuda')
    Train_data = deepcopy(train_ds['data']).to(DEVICE) 
    # If selected indices are provided, deep copy and move the corresponding labels to the device
    Train_lbls = deepcopy(train_ds['labels'][:,:,selected_idx]).to(DEVICE) if not(selected_idx==[]) else deepcopy(train_ds['labels']).to(DEVICE)
    #Calculate the side length of the image (NOTE: the images are squares, and are originally stored as vectors)
    side_img = int(np.sqrt(Train_data.shape[2])) 
    Train_data = Train_data.view(Train_data.shape[0]*Train_data.shape[1], side_img, side_img).unsqueeze(1) #here i am storing the data without batching them, and as images (i.e. matrices instead of vectors)
    Train_lbls = Train_lbls.view(Train_lbls.shape[0]*Train_lbls.shape[1], Train_lbls.shape[2]) # i store also the labels without batching   
    if not(selected_idx==[]): # If selected indices are provided
        #Here i transform multilabels (e.g. blonde and with sunglasses) into one-hot encoded labels, like in the MNIST. This part of the code is used for CelebA labels in particular
        powers_of_10 = torch.pow(10, torch.arange(len(selected_idx), dtype=torch.float)).to(DEVICE) # this generates a tensor of powers of 10, where each element corresponds to a different power of 10
        Cat_labels = torch.matmul(Train_lbls,powers_of_10) #for each multilabel (e.g. 1 0 1 1) i compute the matmul with powers_of_10, so to obtain a single number (e.g. 1101)
        # now in Cat_labels each each element is labelled with a single category, like in MNIST
        cats, f_cat = torch.unique(Cat_labels, return_counts= True) #i compute the frequency of each of the categories in Cat_labels
        lowest_freq_idx = torch.argmin(f_cat)#i find the index of the lower frequency category...
        lowest_freq = f_cat[lowest_freq_idx]#...and using it the frequency of that category...
        lowest_freq_cat = cats[lowest_freq_idx]#...and its identity (lowest_freq_cat) 
        for category in cats: #for every category present in Cat_labels...
            if not(category==lowest_freq_cat): #if that category is not the one with the lowest frequency...
                cat_freq = f_cat[cats==category] #find the frequency of that category
                cat_indexes = torch.where(Cat_labels == category)[0] #...and the indexes of elements identified by that category
                #below i select a random subpopulation of 'category' (of size = to the frequency difference with the lowest freq category) that i will later delete from the dataset
                #in order to have all the labels balanced (i.e. all with the same number of elements)
                if category==0: #if the category is the first one to be iterated (i.e. 0)
                    indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
                else:
                    new_indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
                    indexes_to_delete = torch.cat((indexes_to_delete, new_indexes_to_delete))   
        Idxs_to_keep = torch.tensor([i for i in range(len(Cat_labels)) if i not in indexes_to_delete], device = DEVICE) #the elements i will keep are the ones not present in indexes_to_delete
        # use torch.index_select() to select the elements to keep
        new_Cat_labels = torch.index_select(Cat_labels, 0, Idxs_to_keep)
        #Here below i will re-label the Cat_labels with more manageable names (i.e. progressives from 0 to nr categories -1, as happens for MNIST)
        proxy_cat = 2 #i begin from 2, given that labels 0 and 1 already exist in the old labelling system
        for category in cats:
            if category>=10: #i will change the name to all categories that are not 0 and 1
                new_Cat_labels = torch.where(new_Cat_labels == category, proxy_cat, new_Cat_labels) #this updates new_Cat_labels by replacing certain category labels (category) with a different value (proxy_cat) based on the specified condition (new_Cat_labels == category)
                proxy_cat = proxy_cat + 1 
        new_Train_data = torch.index_select(Train_data, 0, Idxs_to_keep) #i select just the training examples corresponding to Idxs_to_keep
    else:
        new_Train_data = Train_data
        new_Cat_labels = Train_lbls   
    if for_classifier: # if you need your data (CelebA) to be preprocessed to be inputted to a classifier
        #then i apply the following transformation to make the data suitable to ResNet
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224),
            transforms.ToTensor(),])
        dataset = MyDataset(features = new_Train_data, labels= new_Cat_labels, transform=transform)
        #dataset = MyDataset(Train_lbls, Train_data)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        return train_loader #i return the loader ready to be used for classifier training or testing

    elif Old_rbm==False: 
        #This should be the classic preprocessing for the current code DBN
        new_Train_data = torch.squeeze(new_Train_data,1)
        new_Train_data = new_Train_data.view(new_Train_data.shape[0],new_Train_data.shape[1]*new_Train_data.shape[2])
        num_batches = new_Train_data.__len__() // batch_sz
        new_Train_data = new_Train_data[:(num_batches*batch_sz),:]
        new_Train_data = new_Train_data.view(num_batches,batch_sz,new_Train_data.shape[1])
        new_Cat_labels = new_Cat_labels[:(num_batches*batch_sz)]
        new_Cat_labels = new_Cat_labels.view(num_batches,batch_sz,1)
        train_ds = {'data': new_Train_data, 'labels': new_Cat_labels}
        return train_ds
    else: #if you are processing the data to be used by the old monolayer RBM (i.e. the one used in BI23), then...
        new_Train_data = new_Train_data.squeeze(1)
        new_Cat_labels = new_Cat_labels
        return new_Train_data, new_Cat_labels
    
    
def tool_loader_ZAMBRA(DEVICE,  selected_idx = [], only_data = True,classifier_yn = True, last_layer_sz = 1000, Load_DBN_yn = 3):
    from google.colab import drive
    drive.mount('/content/gdrive')
    Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'    
    #load the various files necessary
    with open(path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
        CPAR = json.load(filestream)
    filestream.close()
    #and extract the relevant parameters
    DATASET_ID = CPAR['DATASET_ID'] 
    with open(path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
        LPAR = json.load(filestream)
    filestream.close()
    
    train_dataset_original, test_dataset_original = load_data_ZAMBRA(DATASET_ID,LPAR['BATCH_SIZE'],Zambra_folder_drive) #load the dataset of interest
    batch_sz = train_dataset_original['data'].shape[1]
    if 'CelebA' in DATASET_ID:
        if selected_idx == []:
            nrEx = train_dataset_original['labels'].shape[0] #usare
            train_dataset = deepcopy(train_dataset_original)
            train_dataset['data'] = train_dataset['data'][:nrEx//2,:,:]  #for issues of processing on Colab, i will train using only half of the CelebA training data
            '''
            #Selection of a single label (UNUSED)
            cat_id = 20 #male
            L_all = deepcopy(train_dataset['labels'][:nrEx//2,:,:])
            train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,cat_id]
            test_dataset['labels'] = test_dataset['labels'][:,:,cat_id]
            '''
            train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,:]  # i downsample also the labels
            test_dataset = deepcopy(test_dataset_original)
        else: 
            #i preprocess the data considering only the selected idxs
            train_dataset = Multiclass_dataset(train_dataset_original, selected_idx= selected_idx)
            test_dataset = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx)
    else:
        train_dataset = train_dataset_original
        test_dataset = test_dataset_original
    
    if only_data: #if only the processed data are needed...
        return train_dataset_original, test_dataset_original 
    #PyTorch will use GPU (CUDA) tensors as the default tensor type.
    if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not(Load_DBN_yn == 0 or Load_DBN_yn == 1):
        Load_DBN_yn = int(input('Do you want to load a iDBN (Zambra 22 style) or do you want to train it? (1=yes, 0=no)'))    
    if Load_DBN_yn == 0: #if the user chose to train a iDBN from scratch...
        #i divide the labels (Y) from the training examples (X), both for train and test
        Xtrain = train_dataset['data'].to(DEVICE)
        Xtest  = test_dataset['data'].to(DEVICE)
        Ytrain = train_dataset['labels'].to(DEVICE)
        Ytest  = test_dataset['labels'].to(DEVICE)    
        # -----------------------------------------------------
        # Initialize performance metrics data structures
        loss_metrics = np.zeros((CPAR['RUNS'], LPAR['EPOCHS'], CPAR['LAYERS']))
        acc_metrics  = np.zeros((CPAR['RUNS'], LPAR['EPOCHS'], CPAR['LAYERS']))
        test_repr    = np.zeros((CPAR['RUNS']))
        PATH_MODEL = os.getcwd()  
        # Train the DBN
        for run in range(CPAR['RUNS']):
            print(f'\n\n---Run {run}\n')    
            if CPAR['ALG_NAME'] == 'g':
                dbn = gDBN(CPAR['ALG_NAME'], DATASET_ID, CPAR['INIT_SCHEME'], PATH_MODEL, LPAR['EPOCHS'], DEVICE = DEVICE)
                dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPAR, readout = CPAR['READOUT'])
            elif CPAR['ALG_NAME'] == 'i':
                dbn = iDBN(CPAR['ALG_NAME'], DATASET_ID, CPAR['INIT_SCHEME'], PATH_MODEL, LPAR['EPOCHS'], DEVICE = DEVICE, last_layer_sz =1000)
                dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPAR, readout = CPAR['READOUT'], num_discr = CPAR['NUM_DISCR'])
            elif CPAR['ALG_NAME'] == 'fs':
                dbn = fsDBN(CPAR['ALG_NAME'], DATASET_ID, CPAR['INIT_SCHEME'], PATH_MODEL, LPAR['EPOCHS'], DEVICE = DEVICE)
                dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPAR)
                
            for layer_id, rbm in enumerate(dbn.rbm_layers):
                loss_metrics[run, :, layer_id] = rbm.loss_profile
                acc_metrics[run, :, layer_id] = rbm.acc_profile
            #end    
            test_repr[run] = dbn.test(Xtest, Ytest)[0]
            name = dbn.get_name() 
            #if you use CelebA with one-hot labels (i.e. 4 labels usually)
            if 'CelebA' in DATASET_ID and not(selected_idx == []):
                #the number of classes is 2 to the power of the selected classes 
                dbn.Num_classes = 2**len(selected_idx)
            #i compute the inverse weight matrix that i will use for label biasing on the top layer of the DBN
            dbn.invW4LB(train_dataset)   
                
            fname = f"{name}_{dbn.Num_classes}classes_nEp{LPAR['EPOCHS']}_nL{len(dbn.rbm_layers)}_lastL{dbn.top_layer_size}_bsz{batch_sz}"
            dbn.fname = fname
            #i save the trained DBN
            torch.save(dbn.to(torch.device('cpu')),
                        open(path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
    else: #if you want to load an existing DBN...
        if not('CelebA' in DATASET_ID):
            Num_classes = 10
            nL = 3 #i.e. nr of layers
            last_layer_sz = str(last_layer_sz) #da cambiare in 1000 se si decide
        elif not(selected_idx == []):
            Num_classes = 2**len(selected_idx)
            nL = 3
            last_layer_sz = input('dimensione dell ultimo layer?') #usually 1000 o 5250
        else:
            Num_classes = 40
        fname = f"dbn_iterative_normal_{DATASET_ID}_{Num_classes}classes_nEp{LPAR['EPOCHS']}_nL{nL}_lastL{last_layer_sz}_bsz{batch_sz}"
        dbn = torch.load(path.join(Zambra_folder_drive, fname+'.pkl'))
        if not(hasattr(dbn, 'fname')):
            dbn.fname = fname
            torch.save(dbn.to(torch.device('cpu')), open(path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
    #load also the appropriate classifier to identify the samples generated by the DBN
    if not hasattr(dbn, 'depth'): #temporary
        dbn.depth = len(dbn.rbm_layers)
        dbn.idx_max_depth = dbn.depth - 1
    if classifier_yn == False:
        return dbn,train_dataset_original, test_dataset_original,classifier
    classifier = classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = selected_idx, DEVICE = 'cuda')
    return dbn,train_dataset_original, test_dataset_original,classifier 

def classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = [], DEVICE = 'cuda'):
    if dbn.dataset_id == 'MNIST':
        #I create an instance of the classifier in which I will later load the saved parameters.
        classifier = VGG16((1,32,32), batch_norm=True).to(DEVICE) 
        PATH = '/content/gdrive/My Drive/ZAMBRA_DBN/VGG16_MNIST/VGG16_MNIST_best_val.pth'
        classifier.load_state_dict(torch.load(PATH))
    else: #CelebA
        Load_classifier = int(input('do you want to load a classifier or train it from scratch? (1=load, 0=train)'))
        num_classes = 2**len(selected_idx)
        fname = 'resnet_'+str(num_classes)+'classes.pt'  
        if Load_classifier ==0: #if i want to train the classifier from scratch...
            #i adapt the dataloaders to be suitable for the classifier
            train_dataloader = Multiclass_dataset(train_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
            test_dataloader = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
            #i create the insance of the classifier and train it (all inside the CelebA_ResNet_classifier)
            classifier = CelebA_ResNet_classifier(ds_loaders = [train_dataloader, test_dataloader], num_classes = num_classes,  num_epochs = 20, learning_rate = 0.001, filename=fname)
        else: #if i want to load the classifier 
            classifier = CelebA_ResNet_classifier(ds_loaders = [],  num_classes = num_classes, filename=fname)   
    classifier.eval() #i put the classifier in evaluation mode
    return classifier


def load_NPZ_dataset(ds_filepath, nr_batches_retraining: None|int = None):
    dataset = dict(np.load(ds_filepath))
    # Convert the numpy arrays to torch tensors
    dataset = {k: torch.from_numpy(v) for k, v in dataset.items()}
    if nr_batches_retraining is not None:
        dataset = {'data': dataset['data'][:nr_batches_retraining, :, :], 'labels': dataset['labels'][:nr_batches_retraining, :, :]}
    return dataset
