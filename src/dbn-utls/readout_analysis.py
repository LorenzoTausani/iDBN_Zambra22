import random
from tqdm import tqdm
import torch
from torchvision import datasets,transforms
import pickle
import os
import json
from sklearn.metrics import accuracy_score
from data_load import data_and_labels, load_NPZ_dataset, tool_loader_ZAMBRA
from misc import get_relative_freq, relabel_09, sampling_gen_examples
from plots import hist_pixel_act, plot_relearning
import Study_generativity
from Study_generativity import *
from matplotlib.ticker import StrMethodFormatter
from typing import Literal
MixingType = Literal['origMNIST', 'lbl_bias', 'chimeras', '[]']

# -- RIDGE CLASSIFIER --

def ridge_readout(classifier, Xtrain, Xtest, Ytest):
  n_feat = Xtrain.shape[-1]
  x_test  = Xtest.cpu().numpy().reshape(-1, n_feat)
  y_test  = Ytest.cpu().numpy().flatten()
  y_pred = classifier.predict(x_test)
  readout_acc = accuracy_score(y_test, y_pred)
  return readout_acc

def readout_V_to_Hlast(dbn,train_dataset,test_dataset, DEVICE='cuda', existing_classifier_list = [], fraction_train = 1):
  
  classifier_list = []

  #Load data and labels
  if 'CelebA' in dbn.dataset_id:
    train_dataset = Multiclass_dataset(train_dataset, selected_idx= [20,31])
    test_dataset = Multiclass_dataset(test_dataset, selected_idx = [20,31])
  Xtrain = train_dataset['data'].to(DEVICE)
  Xtest  = test_dataset['data'].to(DEVICE)
  Ytrain = train_dataset['labels'].to(DEVICE)
  Ytest  = test_dataset['labels'].to(DEVICE)

  n_train_batches, batch_sz, _ = Xtrain.shape
  batch_indices = list(range(n_train_batches))
  n_test_batches = Xtest.shape[0]
  upper = int(n_train_batches*fraction_train) #also 4/5
  
  #get readout of the visible layer (i.e. from actual data)
  readout_acc_V =[]
  if len(existing_classifier_list) == 0:
    readout_acc,classifier = dbn.rbm_layers[1].get_readout(Xtrain[:upper, :, :], Xtest, Ytrain[:upper, :, :], Ytest)
    classifier_list.append(classifier)
  else:
    readout_acc = ridge_readout(existing_classifier_list[0], Xtrain, Xtest, Ytest)
  print(f'Readout accuracy = {readout_acc*100:.2f}')
  readout_acc_V.append(readout_acc)

  #get readout of the hidden layers
  for rbm_idx,rbm in enumerate(dbn.rbm_layers):
      #initialize the hidden representations
      _Xtrain = torch.zeros((n_train_batches, batch_sz, rbm.Nout))
      _Xtest = torch.zeros((n_test_batches, batch_sz, rbm.Nout))
      #get the hidden representations of both the train and test sets
      _Xtest, _ = rbm(Xtest)
      random.shuffle(batch_indices)
      with tqdm(batch_indices, unit = 'Batch') as tlayer:
          for n in tlayer:
              tlayer.set_description(f'Layer {rbm.layer_id}')
              _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])
              
      #get readout of the hidden layer
      if len(existing_classifier_list) == 0:
        readout_acc, classifier = rbm.get_readout(_Xtrain[:upper, :, :], _Xtest, Ytrain[:upper, :, :], Ytest)
        classifier_list.append(classifier)
      else:
        readout_acc = ridge_readout(existing_classifier_list[rbm_idx+1], _Xtrain, _Xtest, Ytest)
      print(f'Readout accuracy = {readout_acc*100:.2f}')
      readout_acc_V.append(readout_acc)
      #the hidden representations of the current layer will be used as 
      #the input for the next layer
      Xtrain = _Xtrain.clone()
      Xtest  = _Xtest.clone()
      
  return readout_acc_V, classifier_list

def get_ridge_classifiers(Force_relearning = True, last_layer_sz=1000):
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  MNIST_rc_file= os.path.join(Zambra_folder_drive,'MNIST_ridge_classifiers'+str(last_layer_sz)+'.pkl')
  print("\033[1m Make sure that your iDBN was trained only with MNIST for 100 epochs \033[0m")
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if not(os.path.exists(MNIST_rc_file)) or Force_relearning:
    dbn,MNIST_Train_DS, MNIST_Test_DS,_= tool_loader_ZAMBRA(DEVICE, only_data = False,last_layer_sz=last_layer_sz, Load_DBN_yn = 1)
    _, MNIST_classifier_list = readout_V_to_Hlast(dbn,MNIST_Train_DS, MNIST_Test_DS)
    # Save the list of classifiers to a file
    with open(MNIST_rc_file, 'wb') as file:
        pickle.dump(MNIST_classifier_list, file)
  else:
    with open(MNIST_rc_file, 'rb') as file:
      MNIST_classifier_list = pickle.load(file)# Load the list of classifiers from the file
      
  return MNIST_classifier_list

# -- RETRAINING DATA --

def random_selection_withinBatch(batch_size=128):
  # Creazione di una lista di tutti gli indici possibili da 0 a 127
  all_indices = list(range(batch_size))
  # Estrai 64 indici casuali dalla lista
  random.shuffle(all_indices)
  selected_indices = all_indices[:batch_size//2] 
  # Gli altri 64 indici sono quelli rimanenti
  remaining_indices = all_indices[batch_size//2:]
  return selected_indices, remaining_indices

def mixing_data_within_batch(half_MNIST,half_retraining_ds):
  nr_batches, batch_size, imgvec_len = half_MNIST.shape
  mix_retraining_ds_MNIST = torch.zeros((nr_batches*2,batch_size,imgvec_len))
  #the random selection of batch index is separated for the half MNIST and the half retraining datasets
  rand_select_MNIST = list(range(nr_batches)); random.shuffle(rand_select_MNIST)
  rand_select_retrainDS = list(range(nr_batches)); random.shuffle(rand_select_retrainDS)

  for idx in range(nr_batches): #for every random batch index
      #the random selection is separated also for indices within the batch
      MNIST_idxs_selected, MNIST_idxs_remained = random_selection_withinBatch(batch_size)
      selected_indices_retrainDS, remaining_indices_retrainDS = random_selection_withinBatch(batch_size)
      
      MNIST_examples = half_MNIST[rand_select_MNIST[idx],MNIST_idxs_selected,:]
      retrainDS_examples = half_retraining_ds[rand_select_retrainDS[idx],selected_indices_retrainDS,:]
      mix_batch1 = torch.cat((MNIST_examples, retrainDS_examples), dim=0)
      # Use the permutation to shuffle the examples of the dataset
      mix_retraining_ds_MNIST[idx,:,:] = mix_batch1[torch.randperm(batch_size)]

      MNIST_examples2 = half_MNIST[rand_select_MNIST[idx],MNIST_idxs_remained,:]
      retrainDS_examples2 = half_retraining_ds[rand_select_retrainDS[idx],remaining_indices_retrainDS,:]
      mix_batch2 = torch.cat((MNIST_examples2, retrainDS_examples2), dim=0)
      # Use the permutation to shuffle the examples of the dataset
      mix_retraining_ds_MNIST[idx+(nr_batches//2),:,:] = mix_batch2[torch.randperm(batch_size)]

  return mix_retraining_ds_MNIST
  

def get_retraining_data(MNIST_train_dataset, train_dataset_retraining_ds = {}, dbn=[], classifier=[], 
                        n_steps_generation = 10, ds_type = 'EMNIST', Type_gen = None,
                        H_type = 'det', correction_type = None):
  #Type_gen = 'chimeras'/'lbl_bias'/None
  #correction_type = 'frequency'/'sampling'/'rand'/None
  #NOTA: il labelling dell'EMNIST by class ha 62 labels: le cifre (0-9), lettere MAUSCOLE (10-36), lettere MINUSCOLE(38-62)
  #20,000 uppercase letters from the first 10 EMNIST classes.
  coeff = 1; batch_sz = 128; sample_sz = 20000
  nr_batches_retraining = round(sample_sz/batch_sz) 
  half_batches = round(nr_batches_retraining/2)
  half_ds_size = half_batches*batch_sz #i.e. 9984
  retraining_ds = {'train':train_dataset_retraining_ds}
  if correction_type is not None and Type_gen in ['chimeras','lbl_bias']:
    coeff = 2 #moltiplicatore. Un tempo stava a 2
    #This is the distribution of avg pixels active in the MNIST train dataset
    avg_pixels_active_TrainMNIST = torch.cat([torch.mean(batch, axis=1) for batch in MNIST_train_dataset['data']])

  root = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  #load EMNIST byclass data
  if not(bool(train_dataset_retraining_ds)):
    try:
      train_dataset_retraining_ds = load_NPZ_dataset(os.path.join(root,'dataset_dicts',f'train_dataset_{ds_type}.npz'), nr_batches_retraining)
      test_dataset_retraining_ds = load_NPZ_dataset(os.path.join(root,'dataset_dicts',f'test_dataset_{ds_type}.npz'))
    except:
      transform =transforms.Compose([transforms.ToTensor()])
      if ds_type == 'EMNIST':
        data_train_retraining_ds = datasets.EMNIST('../data', train=True,split = 'byclass', download=True, transform=transform)
        data_test_retraining_ds = datasets.EMNIST('../data', train=False,split = 'byclass', download=True, transform=transform)
        #target_classes = list(range(10, 20)) #i.e. the first 10 capital letter classes
        #data are relabelled from target_classes to 0-9
        target_classes = [17,18,19,20,21,22,23,24,25,26] #migliori dritte: [22,32,26,16,30,11,20,10,23,25], medi[17,18,19,20,21,22,23,24,25,26]
        data_train_retraining_ds = [relabel_09(item,target_classes) for item in data_train_retraining_ds if item[1] in target_classes]
        data_test_retraining_ds = [relabel_09(item,target_classes) for item in data_test_retraining_ds if item[1] in target_classes]
        #raddrizza lettere andava qua
      elif ds_type == 'fMNIST':
          data_train_retraining_ds = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
          data_test_retraining_ds = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

      train_data_retraining_ds, train_labels_retraining_ds = data_and_labels(data_train_retraining_ds, BATCH_SIZE=batch_sz,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
      test_data_retraining_ds, test_labels_retraining_ds = data_and_labels(data_test_retraining_ds, BATCH_SIZE=batch_sz,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
      #i select just 20000 examples (19968 per l'esattezza)
      train_dataset_retraining_ds = {'data': train_data_retraining_ds[:nr_batches_retraining,:,:], 'labels': train_labels_retraining_ds[:nr_batches_retraining,:,:]}
      test_dataset_retraining_ds = {'data': test_data_retraining_ds, 'labels': test_labels_retraining_ds}

    retraining_ds = {'train':train_dataset_retraining_ds, 'test':test_dataset_retraining_ds}
    if dbn==[]:
        return retraining_ds
    
  if not(Type_gen in ['chimeras','lbl_bias','rand']):
    half_MNIST = MNIST_train_dataset['data'][:half_batches,:,:].to('cuda')
  else:
    dbn.invW4LB(MNIST_train_dataset)
    n_samples = math.ceil((sample_sz/2)*coeff/(10*n_steps_generation))
    #NOTA: QUA NON VIENE GESTITO IL CODICE DEI VALORI RANDOM, come veniva fatto prima
    noisy_W = False if H_type == 'det' else True
    LB_hidden, LB_labels = dbn.label_biasing(topk = -1,n_reps= n_samples, noisy_W = noisy_W) 
    dict_DBN_lBias_classic = dbn.generate_from_hidden(LB_hidden, nr_gen_steps=n_steps_generation)

    if Type_gen == 'lbl_bias':
      visible_states = dict_DBN_lBias_classic['vis_states']
    elif Type_gen == 'chimeras':
      Mean, _ = Perc_H_act(dbn, LB_labels, gen_data_dictionary=dict_DBN_lBias_classic, 
                          layer_of_interest=2, plot = False)
      k = int((torch.mean(Mean, axis=0)[0]*dbn.top_layer_size)/100)
      Ian = Intersection_analysis(dbn, top_k_Hidden=k,nr_steps=n_steps_generation)
      Ian.do_intersection_analysis()
      n_samples = math.ceil((sample_sz/2)*coeff/(len(Ian.intersections.keys())*n_steps_generation))
      _, visible_states = multiple_chimeras(Ian.model, classifier, sample_nr = n_samples, Ian = Ian, 
                  nr_gen_steps = n_steps_generation, topk = k, gather_visits = False, gather_visible = True)
    
    elif Type_gen == 'rand':
      Mean, _ = Perc_H_act(dbn, LB_labels, gen_data_dictionary=dict_DBN_lBias_classic, 
                          layer_of_interest=2, plot = False)
      k = int((torch.mean(Mean, axis=0)[0]*dbn.top_layer_size)/100)
      R_hidden = dbn.random_hidden_bias(n = k, size= LB_hidden.shape, discrete = True)
      dict_DBN_R = dbn.generate_from_hidden(R_hidden, nr_gen_steps=n_steps_generation)
      visible_states = dict_DBN_R['vis_states']
      
    # Adapt the shape of the visible states and sample the required amount
    # Vis_states : numbers of samples to generate x number of generation steps x size of the visible layer 
    Vis_states = visible_states.permute(0, 2, 1)
    sample_nr,_, imgvec_len = Vis_states.shape
    Vis_states = Vis_states.reshape(sample_nr*n_steps_generation,imgvec_len) #Vis_states.shape[2]=784
    indices = torch.randperm(sample_nr*n_steps_generation)[:math.ceil(half_ds_size*coeff)]
    # Sample the rows using the generated indices
    sampled_data = Vis_states[indices]    
    #TODO: GUARDARE E MIGLIORARE LA SELECTION SESSSION
    if correction_type is not None and Type_gen in ['chimeras','lbl_bias']:
      avg_activity_sampled_data =  torch.mean(sampled_data,axis = 1)
      hist, bin_edges = np.histogram(avg_pixels_active_TrainMNIST, bins=20, density=True)
      sum_hist = np.sum(hist); prob_distr = hist/sum_hist
      results = torch.tensor([get_relative_freq(av.item(), hist, bin_edges) for av in avg_activity_sampled_data], dtype = torch.float32)
      results = results / sum_hist if correction_type == 'sampling' else results
      if correction_type == 'frequency':
        top_indices = torch.topk(results, k=half_ds_size).indices
      elif correction_type == 'sampling':
        top_indices = torch.tensor(sampling_gen_examples(results, prob_distr,desired_len_array = half_ds_size + 1000)) [:half_ds_size] #1000 è per evitare di andare sotto 9984
        print(f"\033[1mNumber of unique elements: {len(torch.unique(top_indices))}\033[0m")
      elif correction_type == 'rand':
        top_indices = torch.tensor(np.where(results.cpu() != 0)[0])
        random_indices = torch.randperm(top_indices.size(0))
        top_indices = top_indices[random_indices[:half_ds_size]]
      sampled_data = sampled_data[top_indices]
      avg_activity_sampled_data_topK =  torch.mean(sampled_data,axis = 1)
      hist_pixel_act(avg_pixels_active_TrainMNIST,avg_activity_sampled_data,avg_activity_sampled_data_topK, n_bins = 20)

    half_MNIST = sampled_data.view(half_batches, batch_sz, 784)

  half_retraining_ds = train_dataset_retraining_ds['data'][:half_batches,:,:].to('cuda')
  mix_retraining_ds_MNIST=mixing_data_within_batch(half_MNIST,half_retraining_ds)
  # Use the permutation to shuffle the examples of the dataset
  mix_retraining_ds_MNIST = mix_retraining_ds_MNIST[torch.randperm(nr_batches_retraining)]
  retraining_ds['mixed'] = mix_retraining_ds_MNIST
  return retraining_ds

# -- RELEARNING --

def relearning(retrain_ds_type = 'EMNIST', mixing_type: MixingType = '[]', n_steps_generation=10, 
              correction_type = 'frequency', relearning_epochs = 50, readout_interleave = 5,
              new_rdata_epochs: int|None = None, last_layer_sz = 1000, H_type='det'):
    #retrain_ds_type = 'EMNIST'/'fMNIST'
    #load necessary items
    DEVICE='cuda'; DATASET_ID='MNIST'; Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
    dbn,MNISTtrain_ds, MNISTtest_ds,classifier= tool_loader_ZAMBRA(DEVICE, only_data = False,Load_DBN_yn = 1, 
                                                                  last_layer_sz=last_layer_sz)
    type_retrain = 'sequential' if mixing_type=='[]' else 'interleaved'
    type_mix = 'mix_'+mixing_type
    retraining_ds = get_retraining_data(MNISTtrain_ds,{},dbn, classifier,
                    n_steps_generation = n_steps_generation,  ds_type = retrain_ds_type,
                    Type_gen = mixing_type,H_type = H_type, correction_type = correction_type)
    
    MNIST_classifier_list= get_ridge_classifiers(Force_relearning = False, last_layer_sz=last_layer_sz)
    
    with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
        LPARAMS = json.load(filestream)
    with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
        CPARAMS = json.load(filestream)
        
    n_readouts = math.ceil(relearning_epochs/readout_interleave)
    if new_rdata_epochs is not None:
      LPARAMS['EPOCHS']=new_rdata_epochs
      train_per_readout = math.ceil(readout_interleave/new_rdata_epochs)
    else:
      LPARAMS['EPOCHS']=readout_interleave
      train_per_readout = 1 
  
    Xtrain = retraining_ds['train']['data'].to(DEVICE) if mixing_type=='[]' else retraining_ds['mixed'].to(DEVICE)
    Ytrain = retraining_ds['train']['labels'].to(DEVICE) #TO UNDERSTAND: is this just a placeholder?
    Xtest = retraining_ds['test']['data'].to(DEVICE); Ytest = retraining_ds['test']['labels'].to(DEVICE)
    #store the readouts before retraining (i.e. trained only with MNIST)
    readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,MNISTtrain_ds,MNISTtest_ds,existing_classifier_list = MNIST_classifier_list)
    readout_acc_V_RETRAIN_DS,_ = readout_V_to_Hlast(dbn,retraining_ds['train'],retraining_ds['test']) 
    readout = {0: {'digits':readout_acc_V_DIGITS, 'retrain_ds':readout_acc_V_RETRAIN_DS}}
    
    for r_idx in range(n_readouts):
      for _ in range(train_per_readout):
        if new_rdata_epochs is not None:
          r_ds = get_retraining_data(MNISTtrain_ds,retraining_ds['train'],dbn, classifier,
                                    n_steps_generation = n_steps_generation,  ds_type = retrain_ds_type, 
                                    Type_gen = mixing_type,H_type =H_type, correction_type = correction_type)
          Xtrain = r_ds['mixed'].to(DEVICE)
        dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = CPARAMS['READOUT'], 
                  num_discr = CPARAMS['NUM_DISCR'])
      readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,MNISTtrain_ds,MNISTtest_ds,existing_classifier_list = MNIST_classifier_list)
      readout_acc_V_RETRAIN_DS,_ = readout_V_to_Hlast(dbn,retraining_ds['train'],retraining_ds['test'])
      current_retraining_epoch = (r_idx+1)*train_per_readout*LPARAMS['EPOCHS']
      readout[current_retraining_epoch] = {'digits':readout_acc_V_DIGITS, 'retrain_ds':readout_acc_V_RETRAIN_DS}

    return readout, dbn


def get_prototypes(Train_dataset,nr_categories=26):
  Prototypes = torch.zeros(nr_categories,Train_dataset['data'].shape[2])
  beg_range=0
  if nr_categories == 26:
    beg_range=10
  for l_idx in range(beg_range,beg_range+nr_categories):
    indices = torch.nonzero(Train_dataset['labels']==l_idx)

    sel_imgs = torch.zeros(indices.shape[0],Train_dataset['data'].shape[2])

    for c,idx in enumerate(indices):
      nB = int(idx[0])
      nwB = int(idx[1])
      sel_imgs[c,:] = Train_dataset['data'][nB,nwB,:]

    Avg_cat = torch.mean(sel_imgs,axis = 0)
    Prototypes[l_idx-10,:] = Avg_cat

  return Prototypes

'''
analysis prototypes:
MNIST_prototypes = get_prototypes(train_dataset,nr_categories=10)
EMNIST_prototypes = get_prototypes(train_dataset_retraining_ds,nr_categories=26)

Euclidean_dist_MNIST_EMNIST = torch.zeros(MNIST_prototypes.shape[0],EMNIST_prototypes.shape[0])

for i_MNIST, MNIST_prot in enumerate(MNIST_prototypes):
  for i_EMNIST, EMNIST_prot in enumerate(EMNIST_prototypes):
    Euclidean_dist_MNIST_EMNIST[i_MNIST, i_EMNIST] = torch.norm(MNIST_prot - EMNIST_prot)

Mins, _ =torch.min(Euclidean_dist_MNIST_EMNIST,axis=0)
topk_values, topk_indices = torch.topk(Mins, k=10)
alphabet = "abcdefghijklmnopqrstuvwxyz"

# Map indices to letters
letters = [alphabet[i] for i in topk_indices]

# Join the letters to form a string
result = ''.join(letters)

print("Letters with the specified indices:", result)

#vedere i prototipi
Letter_prototypes_28x28 = torch.zeros(26,28,28)
c=0
for lp in EMNIST_prototypes:
  image = lp.view(28, 28)
  # 1. Rotate the tensor 90 degrees in the opposite direction
  #image = torch.rot90(image, k=-1)

  # 2. Flip the tensor horizontally to restore it to its original orientation
  #image = torch.flip(image, [1])
  #Letter_prototypes_28x28[c,:,:] = image

  # Display the image using Matplotlib
  plt.imshow(image.cpu(), cmap='gray')
  plt.show()
  c=c+1
'''

def readout_epoch0_1andHalfBatches(dbn,train_dataset_retraining_ds, test_dataset_retraining_ds,train_dataset,test_dataset, mix_ds = []):
  DEVICE = 'cuda'
  DATASET_ID = 'MNIST'
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
    LPARAMS = json.load(filestream)
  Xtest  = test_dataset_retraining_ds['data'].to(DEVICE)
  Ytest  = test_dataset_retraining_ds['labels'].to(DEVICE)
  MNIST_classifier_list= get_ridge_classifiers(Force_relearning = False)

  R_list=[]
  if mix_ds == []:
    n_in = [1,train_dataset_retraining_ds['data'].shape[0]//2,train_dataset_retraining_ds['data'].shape[0]]
  else:
    n_in = [1,mix_ds.shape[0]//2,mix_ds.shape[0]]
  LPARAMS['EPOCHS'] = 1
  for i in n_in:
    dbn,_, _,_= tool_loader_ZAMBRA(DEVICE, only_data = False,last_layer_sz=1000, Load_DBN_yn = 1)

    if mix_ds ==[]:
      mb_data = train_dataset_retraining_ds['data'][:i,:,:]
      mb_lbls = train_dataset_retraining_ds['labels'][:i,:,:]
      mb_dataset = {'data':mb_data,'labels':mb_lbls}

    else:
      mb_data = mix_ds[:i,:,:]
      mb_lbls = mix_ds[:i,:,:]


    dbn.train(mb_data, Xtest, mb_lbls, Ytest, LPARAMS, readout = False, num_discr = False)

    if not(mix_ds ==[]):
      mb_data = train_dataset_retraining_ds['data'][:i,:,:]
      mb_lbls = train_dataset_retraining_ds['labels'][:i,:,:]
      mb_dataset = {'data':mb_data,'labels':mb_lbls}
    readout_acc_V, classifier_list = readout_V_to_Hlast(dbn,mb_dataset,test_dataset_retraining_ds)

    readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,train_dataset,test_dataset,existing_classifier_list = MNIST_classifier_list)
    print(readout_acc_V)
    R_list.append(readout_acc_V[-1])
    R_list.append(readout_acc_V_DIGITS[-1])
  return R_list

def readout_comparison(dbn, classifier,MNIST_train_dataset,MNIST_test_dataset,
                      mixing_type_options = ['[]','origMNIST', 'chimeras'], retr_DS = 'EMNIST', 
                      H_type = ['det', 'det', 'det'], new_retrain_dataV = [False, False, False]):
    
  if not isinstance(H_type, list): H_type = [H_type] * len(mixing_type_options)
  if not isinstance(new_retrain_dataV, list): new_retrain_dataV = [new_retrain_dataV] * len(mixing_type_options)
  new_retrain_dataV_list = ['1g4e' if nR else '' for nR in new_retrain_dataV]

  Readouts = np.zeros((11+3,len(mixing_type_options)*2))
  
  for id_mix,mix_type in enumerate(mixing_type_options):
    H_type_it = H_type[id_mix]
    if mix_type=='[]': mix_type=[]
    #you will have the half_MNIST_gen_option active (True)
    #only if your mixing type is not origMNIST
    half_MNIST_gen_option = mix_type != 'origMNIST'

    Retrain_ds,Retrain_test_ds,mix_retrain_ds = get_retraining_data(MNIST_train_dataset,{},dbn, classifier,100,  ds_type = retr_DS, Type_gen = mix_type, correction_type = 'frequency')

    if mix_type=='[]':
      R = readout_epoch0_1andHalfBatches(dbn,Retrain_ds,Retrain_test_ds,MNIST_train_dataset,MNIST_test_dataset, mix_ds = [])
    else:
      R = readout_epoch0_1andHalfBatches(dbn,Retrain_ds,Retrain_test_ds,MNIST_train_dataset,MNIST_test_dataset, mix_ds = mix_retrain_ds)
    i_Retr = [0,2,4]
    i_MNIST = [1,3,5]
    Readouts[:3,id_mix+len(mixing_type_options)] = [R[i] for i in i_Retr]
    Readouts[:3,id_mix] = [R[i] for i in i_MNIST]
    if mix_type==[]:
      mix_type='[]'

    Readout_last_layer_MNIST, Readout_last_layer_RETRAINING_DS,_ = relearning(retrain_ds_type = retr_DS, mixing_type =mix_type, n_steps_generation=100, new_rdata_epochs = new_retrain_dataV[id_mix], correction_type = 'other', l_par = 1, last_layer_sz=1000, H_type = H_type_it)
    Readouts[3:,id_mix] = Readout_last_layer_MNIST
    Readouts[3:,id_mix+len(mixing_type_options)] = Readout_last_layer_RETRAINING_DS

  D_names = {'[]':'seq', 'origMNIST': 'int_orig', 'chimeras':'int_chim', 'lbl_bias': 'int_LB'}

  # Define column names
  columns = ['MNIST ' + D_names[m] + '_H' + h +'_'+ nR for m, h,nR in zip(mixing_type_options, H_type,new_retrain_dataV_list)] + [retr_DS +' '+ D_names[m] + '_H' + h +'_'+ nR for m, h,nR in zip(mixing_type_options, H_type,new_retrain_dataV_list)]

  # Convert NumPy array to Pandas DataFrame
  df = pd.DataFrame(Readouts, columns=columns)
  
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  # Save DataFrame to Excel file
  tipo = ['M'+D_names[m] + '_H' + h +'_'+ nR for m, h,nR in zip(mixing_type_options, H_type,new_retrain_dataV_list)]
  tipo = '_'.join(tipo)
  file_path = os.path.join(Zambra_folder_drive, "Readouts_" + retr_DS + tipo + ".xlsx")

  df.to_excel(file_path, index=False)
  

  ix = [0,1,2,4,5,6,7,8,9,10,11,12,13]
  plot_relearning(Readouts[ix,:], yl = [0.75, 1], legend_labels = [])

  return Readouts
