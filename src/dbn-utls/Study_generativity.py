from tqdm import tqdm #tqdm is a Python library that provides a way to create progress bars for loops and iterators, 
#making it easier to track the progress of lengthy operations.
import numpy as np
import torch
from dbns import *
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import Classifiers
import methods
from Classifiers import *
from methods import *
from itertools import combinations
from misc import save_mat_xlsx
from data_load import Multiclass_dataset
import seaborn as sns

class Intersection_analysis:
    def __init__(self, model, top_k_Hidden=100, nr_steps=100):
        self.model = model #the DBN model
        self.top_k_Hidden = top_k_Hidden #nr of hidden units with highest activity, which will then be binarized to 1
        self.nr_steps = nr_steps #nr steps of generation
        #label biasing for the intersection method
        LB_hidden, LB_labels = self.model.label_biasing(topk=-1,n_reps = 1)
        self.LB_hidden = LB_hidden; self.LB_labels = LB_labels
        
    def do_intersection_analysis(self):
      #Find top k active indices for each category
      top_k_idxs_LB = {i: torch.topk(self.LB_hidden[:, i], self.top_k_Hidden)[1] for i in range(self.model.Num_classes)}

      #Iterate over each binary combination of categories
      intersections = {}
      for cat1, cat2 in combinations(range(self.model.Num_classes), 2):
          #Find the intersection of the top k active indices betw the 2 cats
          intersections[f"{cat1},{cat2}"] = torch.tensor(sorted(list(set(top_k_idxs_LB[cat1].tolist()).intersection(top_k_idxs_LB[cat2].tolist()))))
      self.intersections = intersections
      
    def do_intersection_analysis_ZAMBRA(self):
      #for the intersection method
      for dig in range(self.model.Num_classes): #for each class...
        g_H = self.model.getH_label_biasing( on_digits=dig, topk = -1) #...do label biasing activating just that digit
        if dig == 0:
            hid_bias = g_H
        else:
            hid_bias = torch.hstack((hid_bias,g_H)) #stack together the label biasing vector for each digit

      vettore_indici_allDigits_biasing = torch.empty((0),device= self.model.DEVICE)

      for digit in range(self.model.Num_classes): #for each digit
        hid_vec_B = hid_bias[:,digit] #get the hidden state obtained by label biasing with the specific class 'digit'
        #in the next two lines i find the top p indexes in terms of activation
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) 
        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #I concatenate the top p indexes for all digits in this vector

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) # Of the indexes found i take just the ones that are not repeated      

      digit_digit_common_elements_count_biasing = torch.zeros((self.model.Num_classes,self.model.Num_classes)) #in here i will count the number of common elements in each intersection
      self.unique_H_idxs_biasing = unique_idxs_biasing

      result_dict_biasing ={} #here i will store, for each combination of classes (keys), the units in intersection between them
      #for each category i iterate to compute the entries of the nr.classes x nr.classes matrices
      #itero per ogni digit per calcolare le entrate delle matrici 10 x 10
      for row in range(self.model.Num_classes): 
        for col in range(self.model.Num_classes):

          common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

          counter_biasing = 0
          for id in unique_idxs_biasing: #for each of the top indices
            digits_found = torch.floor(torch.nonzero(vettore_indici_allDigits_biasing==id)/self.top_k_Hidden)
            #torch.nonzero(vettore_indici_allDigits_biasing==id) finds the positions in the array vettore_indici_allDigits_biasing  where there is the value id is present
            #indeed, given that the vector vettore_indici_allDigits_biasing contains the top 100 most active units for each digit, if i divide the indexes by 100 (i.e. top_k_Hidden)
            #then i will find for which digit the unit id was active.

            if torch.any(digits_found==row) and torch.any(digits_found==col): #if the digits found present both the row and the col digits...
                common_el_idxs_biasing = torch.hstack((common_el_idxs_biasing,id)) #add the id to the vector of ids that will be used for intersection method biasing
                counter_biasing += 1 # i count the number of intersection elements to fill in the digit_digit_common_elements_count_biasing matrix

          result_dict_biasing[str(row)+','+str(col)] = common_el_idxs_biasing #store the units in the intersection
          digit_digit_common_elements_count_biasing[row,col] = counter_biasing

      self.intersections = result_dict_biasing 

      print(digit_digit_common_elements_count_biasing)
      #lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))

      return digit_digit_common_elements_count_biasing
    
    def generate_chimera(self,classifier, cats2intersect = [8,2], sample_nr = 1000, plot=0):
      #this function does generation from chimeras obtained with the intersection method
      cats2intersect= sorted(cats2intersect)
      biasing_vecs =torch.zeros(sample_nr,self.model.top_layer_size) 
      if not(cats2intersect =='rand'): #if you don't want to generate from random chimeras
        #activate the entries corresponding the intersection units of interest
        biasing_vecs[:,self.intersections[f"{cats2intersect[0]},{cats2intersect[1]}"]]=1
      else: #cats2intersect = 'rand'
        for i in range(sample_nr): #for every sample you want to generate
          #select two random classes
          n1, n2 = sorted(random.sample(range(self.model.Num_classes), 2))
          #activate the entries corresponding the intersection units of interest
          biasing_vecs[i,self.intersections[f"{n1},{n2}"]]=1

      biasing_vecs = torch.transpose(biasing_vecs,0,1)
      #generate from the hidden vectors produced
      d = self.model.generate_from_hidden(biasing_vecs, nr_gen_steps=self.nr_steps)  
      d = Classifier_accuracy(d, classifier, self.model, plot=0) #compute the accuracy of the classifier over the generation period
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,self.model, Plot=plot, Ian=1)
      
      return d, df_average,df_sem, Transition_matrix_rowNorm

    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2], temperature=1, nr_of_examples = 1000, plot=0, entropy_correction=[]):
      #this function does generation from chimeras obtained with the intersection method
      b_vec =torch.zeros(nr_of_examples,self.model.top_layer_size) 
      if not(elements_of_interest =='rand'): #if you don't want to generate from random chimeras
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1]) #entry of interest in the intersection dictionary
        b_vec[:,self.intersections[dictionary_key].long()]=1#activate the entries corresponding the intersection units of interest

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples): #for every sample you want to generate
          #select two random classes
          n1 = random.randint(0, self.model.Num_classes-1) 
          n2 = random.randint(0, self.model.Num_classes-1)
          #activate the entries corresponding the intersection units of interest
          dictionary_key = str(n1)+','+str(n2) 
          b_vec[i,self.intersections[dictionary_key].long()]=1

      b_vec = torch.transpose(b_vec,0,1)
      #b_vec = torch.unsqueeze(b_vec,0) #NOT USED
      d = self.model.generate_from_hidden(b_vec, nr_gen_steps=self.nr_steps) #generate from the hidden vectors produced
      
      d = Classifier_accuracy(d, VGG_cl, self.model, plot=0) #compute the accuracy of the classifier over the generation period
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,self.model, Plot=plot, Ian=1)
      
      # if nr_of_examples < 16:
      #   Plot_example_generated(d, self.model ,row_step = 10, dS=20, custom_steps = True, Show_classification = False)

      
      return d, df_average,df_sem, Transition_matrix_rowNorm

def Chimeras_nr_visited_states(model, classifier, Ian =[], topk=149, apprx=1,plot=1,compute_new=1,
                                      nr_sample_generated =100,cl_labels=[], lS=25):
    c_Tmat = 0
    n_digits = model.Num_classes
    combinations_of_two = list(combinations(range(n_digits), 2))
    
    if Ian!=[]:
      fN='Visited_digits_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr='Visited_digits_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fN_NDST='Nondigit_stateTime_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
    else:
      fN='Visited_digits_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr='Visited_digits_Lbiasing_error_k' + str(topk)+'.xlsx'
      fN_NDST='Nondigit_stateTime_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_Lbiasing_error_k' + str(topk)+'.xlsx'

    if compute_new==1:
      #both
      Vis_states_mat = np.zeros((n_digits, n_digits))
      Vis_states_err = np.zeros((n_digits, n_digits))
      if n_digits==10:
        Non_digit_mat  = np.zeros((n_digits, n_digits))
        Non_digit_err  = np.zeros((n_digits, n_digits))

      for row, col in combinations_of_two:
        if Ian!=[]: #intersection method
          d, df_average,df_sem, _ = Ian.generate_chimera(classifier,cats2intersect = [row,col],
                                        sample_nr = nr_sample_generated,plot=0)
        else: #double label biasing
          LB2_hidden = model.getH_label_biasing(on_digits=[row,col], topk=topk)
          LB2_hidden = LB2_hidden.repeat(1,nr_sample_generated)
          d = model.generate_from_hidden(LB2_hidden, nr_gen_steps=100)
          d = Classifier_accuracy(d, classifier,model,labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
          df_average,df_sem, _ = classification_metrics(d,model,Plot=0,dS=50,Ian=1)

        c_Tmat = c_Tmat+1
        Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
        Vis_states_err[row,col]=df_sem.Nr_visited_states[0]
        if n_digits==10:
          Non_digit_mat[row,col] = df_average['Non-digit'][0]
          Non_digit_err[row,col] = df_sem['Non-digit'][0]

      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)
      if n_digits==10:
        save_mat_xlsx(Non_digit_mat, filename=fN_NDST)
        save_mat_xlsx(Non_digit_err, filename=fNerr_NDST)

    else: #load already computed Vis_states_mat
      if n_digits==10:
        Non_digit_mat = pd.read_excel(fN_NDST)
        Non_digit_err = pd.read_excel(fNerr_NDST)
        # Convert the DataFrame to a NumPy array
        Non_digit_mat = Non_digit_mat.values
        Non_digit_err = Non_digit_err.values
      Vis_states_mat = pd.read_excel(fN)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values

      Vis_states_err = pd.read_excel(fNerr)
      # Convert the DataFrame to a NumPy array
      Vis_states_err = Vis_states_err.values

    if plot==1:

      Vis_states_mat = Vis_states_mat.round(apprx)
      Vis_states_err = Vis_states_err.round(apprx)

      plt.figure(figsize=(15, 15))
      mask = np.triu(np.ones_like(Vis_states_mat),k=+1) # k=+1 per rimuovere la diagonale
      # Set the lower triangle to NaN
      Vis_states_mat = np.where(mask==0, np.nan, Vis_states_mat)
      Vis_states_mat = Vis_states_mat.T
      #ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=False,square=True, cbar=False)
      ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=True, annot_kws={"size": lS},square=True,cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
      if not(cl_labels==[]):
        ax.set_xticklabels(cl_labels)
        ax.set_yticklabels(cl_labels)
      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)

      plt.xlabel('Class', fontsize = lS) # x-axis label with fontsize 15
      plt.ylabel('Class', fontsize = lS) # y-axis label with fontsize 15
      #cbar = plt.gcf().colorbar(ax.collections[0], location='left', shrink=0.82)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)
      plt.show()

    if n_digits==10:
      #print('final c_Tmat',c_Tmat) #NOT USED
      return Vis_states_mat, Vis_states_err,Non_digit_mat,Non_digit_err #,Transition_matrix_tensor
    else:
      return Vis_states_mat, Vis_states_err
    


def Chimeras_nr_visited_states_ZAMBRA(model, VGG_cl, Ian =[], topk=149, apprx=1,plot=1,compute_new=1, nr_sample_generated =100, entropy_correction=[],cl_labels=[], lS=25):
    #Transition_matrix_tensor = torch.zeros((11, 11, 55), device='cuda') #NOT USED
    c_Tmat = 0
    n_digits = model.Num_classes
    if Ian!=[]:
      fN='Visited_digits_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr='Visited_digits_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fN_NDST='Nondigit_stateTime_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
    else:
      fN='Visited_digits_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr='Visited_digits_Lbiasing_error_k' + str(topk)+'.xlsx'
      fN_NDST='Nondigit_stateTime_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_Lbiasing_error_k' + str(topk)+'.xlsx'

    if compute_new==1:
      #both
      Vis_states_mat = np.zeros((n_digits, n_digits))
      Vis_states_err = np.zeros((n_digits, n_digits))
      if n_digits==10:
        Non_digit_mat  = np.zeros((n_digits, n_digits))
        Non_digit_err  = np.zeros((n_digits, n_digits))

      if Ian!=[]:
        for row in range(n_digits):
          for col in range(row,n_digits):
            d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = [row,col], nr_of_examples = nr_sample_generated, temperature = 1, plot=0, entropy_correction= entropy_correction)
            if not(row==col):
              #Transition_matrix_tensor[:,:, c_Tmat] = Transition_matrix_rowNorm #NOT USED
              c_Tmat = c_Tmat+1
            Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
            Vis_states_err[row,col]=df_sem.Nr_visited_states[0]
            if n_digits==10:
              Non_digit_mat[row,col] = df_average['Non-digit'][0]
              Non_digit_err[row,col] = df_sem['Non-digit'][0]
      else:
        numbers = list(range(n_digits))
        combinations_of_two = list(combinations(numbers, 2))

        for idx, combination in enumerate(combinations_of_two):
          gen_hidden = model.label_biasing(on_digits=  list(combination), topk = topk)
          gen_hidden_rep = gen_hidden.repeat(1,nr_sample_generated)
          d = model.generate_from_hidden(gen_hidden_rep , nr_gen_steps=100)
          d = Classifier_accuracy(d, VGG_cl,model, labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
          df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,model,Plot=0,dS=50,Ian=1)
          if not(combination[0]==combination[1]):
            #Transition_matrix_tensor[:,:, c_Tmat] = Transition_matrix_rowNorm #NOT USED
            c_Tmat = c_Tmat+1
          Vis_states_mat[combination[0],combination[1]]=df_average.Nr_visited_states[0]
          Vis_states_err[combination[0],combination[1]]=df_sem.Nr_visited_states[0]
          if n_digits==10:
            Non_digit_mat[combination[0],combination[1]] = df_average['Non-digit'][0]
            Non_digit_err[combination[0],combination[1]] = df_sem['Non-digit'][0]


      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)
      if n_digits==10:
        save_mat_xlsx(Non_digit_mat, filename=fN_NDST)
        save_mat_xlsx(Non_digit_err, filename=fNerr_NDST)

    else: #load already computed Vis_states_mat
      if n_digits==10:
        Non_digit_mat = pd.read_excel(fN_NDST)
        Non_digit_err = pd.read_excel(fNerr_NDST)
        # Convert the DataFrame to a NumPy array
        Non_digit_mat = Non_digit_mat.values
        Non_digit_err = Non_digit_err.values
      Vis_states_mat = pd.read_excel(fN)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values

      Vis_states_err = pd.read_excel(fNerr)
      # Convert the DataFrame to a NumPy array
      Vis_states_err = Vis_states_err.values

    if plot==1:

      Vis_states_mat = Vis_states_mat.round(apprx)
      Vis_states_err = Vis_states_err.round(apprx)

      plt.figure(figsize=(15, 15))
      mask = np.triu(np.ones_like(Vis_states_mat),k=+1) # k=+1 per rimuovere la diagonale
      # Set the lower triangle to NaN
      Vis_states_mat = np.where(mask==0, np.nan, Vis_states_mat)
      Vis_states_mat = Vis_states_mat.T
      #ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=False,square=True, cbar=False)
      ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=True, annot_kws={"size": lS},square=True,cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
      if not(cl_labels==[]):
        ax.set_xticklabels(cl_labels)
        ax.set_yticklabels(cl_labels)
      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)

      plt.xlabel('Class', fontsize = lS) # x-axis label with fontsize 15
      plt.ylabel('Class', fontsize = lS) # y-axis label with fontsize 15
      #cbar = plt.gcf().colorbar(ax.collections[0], location='left', shrink=0.82)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)
      plt.show()

    if n_digits==10:
      #print('final c_Tmat',c_Tmat) #NOT USED
      return Vis_states_mat, Vis_states_err,Non_digit_mat,Non_digit_err #,Transition_matrix_tensor
    else:
      return Vis_states_mat, Vis_states_err
    
    
def Perc_H_act(model, sample_labels, gen_data_dictionary=[], dS = 50, l_sz = 5, layer_of_interest=2):

    c=0 #inizializzo il counter per cambiamento colore
    cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
    figure, axis = plt.subplots(1, 1, figsize=(15,15)) #setto le dimensioni della figura
    lbls = [] # qui storo le labels x legenda

    for digit in range(model.Num_classes): # per ogni digit...
        
        Color = cmap(c/256) #setto il colore di quel determinato digit
        l = torch.where(sample_labels == digit) #trovo gli indici dei test data che contengono quel determinato digit
        nr_examples= len(l[0]) #nr degli esempi di quel digit (i.e. n)

        gen_H_digit = gen_data_dictionary['hid_states'][layer_of_interest,l[0],:,:]
        nr_steps = gen_H_digit.size()[2]
        if digit == 0:
            Mean_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
            Sem_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
        SEM = torch.std(torch.mean(gen_H_digit,1)*100,0)/math.sqrt(nr_examples)
        MEAN = torch.mean(torch.mean(gen_H_digit,1)*100,0).cpu()
        Mean_storing[digit, : ] = MEAN.cuda()
        Sem_storing[digit, : ] = SEM

        if digit==0: #evito di fare sta operazione più volte
          y_lbl = '% active H units'

        SEM = SEM.cpu() #sposto la SEM su CPU x plotting
        x = range(1,nr_steps+1) #asse delle x, rappresentante il nr di step di ricostruzione svolti
        plt.plot(x, MEAN, c = Color, linewidth=l_sz) #plotto la media
        plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color, alpha=0.3) # e le barre di errore
        
        c = c+25
        lbls.append(digit)

    axis.legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) # legenda
    #ridimensiono etichette assi e setto le labels
    axis.tick_params(axis='x', labelsize= dS) 
    axis.tick_params(axis='y', labelsize= dS)
    axis.set_ylabel(y_lbl,fontsize=dS)
    axis.set_xlabel('Generation step',fontsize=dS)
    axis.set_title(y_lbl+' - classwise',fontsize=dS)


    axis.set_ylim([0,100])
    return Mean_storing, Sem_storing


def readout_V_to_Hlast(dbn,train_dataset,test_dataset, DEVICE='cuda'):
  if 'CelebA' in dbn.dataset_id:
    train_dataset = Multiclass_dataset(train_dataset, selected_idx= [20,31])
    test_dataset = Multiclass_dataset(test_dataset, selected_idx = [20,31])

  Xtrain = train_dataset['data'].to(DEVICE)
  Xtest  = test_dataset['data'].to(DEVICE)
  Ytrain = train_dataset['labels'].to(DEVICE)
  Ytest  = test_dataset['labels'].to(DEVICE)
  readout_acc_V =[]

  n_train_batches = Xtrain.shape[0]
  n_test_batches = Xtest.shape[0]
  batch_size = Xtrain.shape[1]

  readout_acc = dbn.rbm_layers[1].get_readout(Xtrain, Xtest, Ytrain, Ytest)
  print(f'Readout accuracy = {readout_acc*100:.2f}')
  readout_acc_V.append(readout_acc)
  for rbm in dbn.rbm_layers:

      _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
      _Xtest = torch.zeros((n_test_batches, batch_size, rbm.Nout))

      _Xtest, _ = rbm(Xtest)

      batch_indices = list(range(n_train_batches))
      random.shuffle(batch_indices)
      with tqdm(batch_indices, unit = 'Batch') as tlayer:
          for idx, n in enumerate(tlayer):

              tlayer.set_description(f'Layer {rbm.layer_id}')
              _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])

          #end BATCHES
      #end WITH

      readout_acc = rbm.get_readout(_Xtrain, _Xtest, Ytrain, Ytest)
      print(f'Readout accuracy = {readout_acc*100:.2f}')
      #end
      readout_acc_V.append(readout_acc)

      Xtrain = _Xtrain.clone()
      Xtest  = _Xtest.clone()
  return readout_acc_V


def comparisons_plot(Results_dict, sel_key='Nr_visited_states_MEAN'):
  # parametri grafici
  def Nr_visited_states_MEAN_SEM(Results_dict, MNIST_fname):
    def SEM(measure):
      import math
      nr_of_measures = len(measure)
      if not(isinstance(measure, np.ndarray)):
        measure = np.asarray(measure)
      sem = np.std(measure)/math.sqrt(nr_of_measures)
      return sem

    MNIST = Results_dict[MNIST_fname]
    if not('Nr_visited_states_MEAN' in MNIST):
      if len(MNIST.keys()) == 3:
        sz_mean = len(MNIST.keys())
      else:
        sz_mean = len(MNIST.keys())-1
      Nr_visited_states_MEAN = np.zeros(sz_mean)
      Nr_visited_states_SEM = np.zeros(sz_mean)
      c=0
      keys = ['Nr_visited_states_LB', 'Nr_visited_states_C2lb', 'Nr_visited_states_Cint']
      for k in keys:
        print()
        MNIST[k] = np.array(MNIST[k])
        print(MNIST[k])
        Nr_visited_states_MEAN[c]=np.mean(MNIST[k])
        Nr_visited_states_SEM[c]=SEM(MNIST[k])
        c=c+1
      Results_dict[MNIST_fname]['Nr_visited_states_MEAN']=Nr_visited_states_MEAN
      Results_dict[MNIST_fname]['Nr_visited_states_SEM']=Nr_visited_states_SEM
    return Results_dict

  LineW = 4
  Mk = 's'
  Mk_sz = 12
  Cp_sz = 12
  Err_bar_sz = 4
  Scritte_sz = 50

  if sel_key=='Nr_visited_states_MEAN':
    x_labels = ['LB', 'C_2LB', 'C_int']
    x_lab = 'Generation method'
    y_lab = 'Number of states'
    y_r = [1,4]
  else:
    # Create a list of the x-axis labels
    x_labels = ['V', 'H1', 'H2', 'H3']
    x_lab = 'Layer'
    y_lab = 'Accuracy'
    y_r = [0.85,1]


  # ottieni le chiavi del dizionario e calcola il loro numero
  keys = list(Results_dict.keys())
  n = len(keys)

  # costruisci la stringa per il prompt di input
  input_prompt = "Which keys do you want to select (indicate in [] as a list)?\n"
  for i in range(n):
      input_prompt += f"{i}: {keys[i]}\n"

  # richiedi all'utente di selezionare una chiave
  selected_key_index = input(input_prompt)
  selected_key_index = eval(selected_key_index)

  fnames = [keys[idx] for idx in selected_key_index]
  def custom_sort(elem):
    if 'dbn' in elem:
        return (0, -1 * elem.count('MNIST'))
    elif 'MNIST' in elem:
        return (1, -1 * elem.count('MNIST'))
    else:
        return (2, 0)

  fnames = sorted(fnames, key=custom_sort)

  line_list = []
  fig, ax = plt.subplots(figsize=(15, 15))
  for fname in fnames:
    Results_dict = Nr_visited_states_MEAN_SEM(Results_dict, fname)
    Dati = np.array(Results_dict[fname][sel_key])
    if 'RBM' in fname:
      model_type = 'RBM'
      L_style='--'
    else:
      model_type = 'iDBN'
      L_style='-'
    if 'MNIST' in fname:
      L_col = 'blue'
      ds_type = 'MNIST'
    else:
      L_col = 'red'
      ds_type = 'CelebA'
      if sel_key=='readout':
        y_r = [0.75,0.8]
    print(Dati)
    model_type = model_type+' '+ds_type
    linei, = ax.plot(x_labels, Dati, color=L_col, label=model_type, linewidth=LineW, marker=Mk, markersize=Mk_sz, linestyle=L_style)
    line_list.append(linei)
    if sel_key=='Nr_visited_states_MEAN':
      Dati_SEM = np.array(Results_dict[fname]['Nr_visited_states_SEM'])
      # Add error bars to the second line
      ax.errorbar(x_labels, Dati, yerr=Dati_SEM, fmt='none', ecolor=L_col, capsize=Cp_sz,  elinewidth=Err_bar_sz)
  if len(fnames)>1:
    ax.legend(handles=line_list, loc='upper center', bbox_to_anchor=(1.35, 0.7), fontsize=Scritte_sz)


    
  # Set the x-axis label
  ax.set_xlabel(x_lab, fontsize=Scritte_sz)
  # Set the y-axis label
  ax.set_ylabel(y_lab, fontsize=Scritte_sz)
  # Set the font size of all the text in the plot
  plt.rc('font', size=Scritte_sz)
  # Set the y-axis range
  ax.set_ylim(y_r)
  # Set the x-axis tick font size
  ax.tick_params(axis='x', labelsize=Scritte_sz)
  # Set the y-axis tick font size
  ax.tick_params(axis='y', labelsize=Scritte_sz)
  # Display the plot
  plt.show()

