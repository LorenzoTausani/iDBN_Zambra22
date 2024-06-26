from collections import defaultdict
from tqdm import tqdm #tqdm is a Python library that provides a way to create progress bars for loops and iterators, 
#making it easier to track the progress of lengthy operations.
import numpy as np
import torch
from src.dbn_utls.dbns import *
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from src.dbn_utls.Classifiers import *
from src.dbn_utls.methods import *
from itertools import combinations
from src.dbn_utls.misc import SEM, save_mat_xlsx
from src.dbn_utls.data_load import Multiclass_dataset
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
    
    def generate_chimera(self,classifier, cats2intersect = [8,2], sample_nr = 1000, plot=0):
      #this function does generation from chimeras obtained with the intersection method
      biasing_vecs =torch.zeros(sample_nr,self.model.top_layer_size) 
      if not(cats2intersect =='rand'): #if you don't want to generate from random chimeras
        #activate the entries corresponding the intersection units of interest
        cats2intersect= sorted(cats2intersect)
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


def multiple_chimeras(model, classifier, sample_nr, Ian: Intersection_analysis| None = None, 
                      nr_gen_steps = 100, topk =-1, gather_visits = False, gather_visible = False):
  
  n_digits = model.Num_classes
  if Ian: Ian.nr_steps = nr_gen_steps; Ian.top_k_Hidden = topk
  combinations_of_two = list(combinations(range(n_digits), 2))
  
  #initialize the outputs
  states_stats = {'Vis_states_mat': np.zeros((n_digits, n_digits)),
            'Vis_states_err': np.zeros((n_digits, n_digits)),
            'Non_digit_mat': np.zeros((n_digits, n_digits)),
            'Non_digit_err': np.zeros((n_digits, n_digits))}
  visible_states = []
  generated_data_avg = defaultdict(dict)
  #loop for every combination of classes
  for row, col in combinations_of_two:
    if Ian is not None: #intersection method
      d, df_average,df_sem, _ = Ian.generate_chimera(classifier,cats2intersect = [row,col],
                                    sample_nr = sample_nr,plot=0)
      for k in ['hid_prob','hid_states']:
          generated_data_avg[f"{row},{col}"][k]= torch.mean(d[k][-1,:,:,:],axis=0)
    else: #double label biasing
      LB2_hidden = model.getH_label_biasing(on_digits=[row,col], topk=topk)
      LB2_hidden = LB2_hidden.repeat(1,sample_nr/n_digits)
      d = model.generate_from_hidden(LB2_hidden, nr_gen_steps=nr_gen_steps)
      d = Classifier_accuracy(d, classifier,model,labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
      df_average,df_sem, _ = classification_metrics(d,model,Plot=0,dS=50,Ian=1)
    #gather the outputs of interest  
    if gather_visits:
      states_stats['Vis_states_mat'][row,col]=df_average.Nr_visited_states[0]
      states_stats['Vis_states_err'][row,col]=df_sem.Nr_visited_states[0]
      if 'Non-digit' in df_average.keys():
        states_stats['Non_digit_mat'][row,col] = df_average['Non-digit'][0]
        states_stats['Non_digit_err'][row,col] = df_sem['Non-digit'][0]
        
    if gather_visible:
      visible_states.append(d['vis_states'][:,:,:nr_gen_steps])
  
  if gather_visible:
    visible_states = torch.cat(visible_states, dim=0)
    
  return states_stats, visible_states,generated_data_avg


def Chimeras_nr_visited_states(model, classifier, Ian = None, topk=-1, apprx=1,plot=1,compute_new=1,
                                      nr_sample_generated =100, nr_gen_steps=100, cl_labels=[], lS=25):
    n_digits = model.Num_classes
    Chim_type = '2LB' if Ian is None else 'Int'
    
    if compute_new==1:
      states_stats, _ ,_= multiple_chimeras(model, classifier, sample_nr = nr_sample_generated, Ian = Ian, 
                        nr_gen_steps = nr_gen_steps, topk = topk, gather_visits = True, gather_visible = False)
      
      for k, v in states_stats.items():
        fN = f"{k}_topk{topk}_{Chim_type}.xlsx"
        save_mat_xlsx(v, filename=fN)

    else: #load already computed Vis_states_mat
      states_stats = {'Vis_states_mat': np.zeros((n_digits, n_digits)),
                'Vis_states_err': np.zeros((n_digits, n_digits)),
                'Non_digit_mat': np.zeros((n_digits, n_digits)),
                'Non_digit_err': np.zeros((n_digits, n_digits))}
      for k, v in states_stats.items():
        fN = f"{k}_topk{topk}_{Chim_type}.xlsx"
        try:
          states_stats[k] = pd.read_excel(fN).values
        except:
          print(f"File {fN} not found")

    if plot==1:
      Vis_states_mat = states_stats['Vis_states_mat'].round(apprx)
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

    return states_stats
    
        
def Perc_H_act(model, sample_labels, gen_data_dictionary=[], dS = 50, l_sz = 5, layer_of_interest=2, plot = False):

    c=0 #inizializzo il counter per cambiamento colore
    cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
    if plot: _, axis = plt.subplots(1, 1, figsize=(15,15))
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
        if plot:
          x = range(1,nr_steps+1) #asse delle x, rappresentante il nr di step di ricostruzione svolti
          plt.plot(x, MEAN, c = Color, linewidth=l_sz) #plotto la media
          plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color, alpha=0.3) # e le barre di errore
        
        c = c+25
        lbls.append(digit)
    if plot:
      axis.legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) # legenda
      #ridimensiono etichette assi e setto le labels
      axis.tick_params(axis='x', labelsize= dS) 
      axis.tick_params(axis='y', labelsize= dS)
      axis.set_ylabel(y_lbl,fontsize=dS)
      axis.set_xlabel('Generation step',fontsize=dS)
      axis.set_title(y_lbl+' - classwise',fontsize=dS)
      axis.set_ylim([0,100])
    
    return Mean_storing, Sem_storing

def comparisons_plot(Results_dict, sel_key='Nr_visited_states_MEAN'):
  # parametri grafici
  def Nr_visited_states_MEAN_SEM(Results_dict, MNIST_fname):
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

