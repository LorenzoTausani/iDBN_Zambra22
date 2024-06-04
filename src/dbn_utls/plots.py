import re
from matplotlib.lines import Line2D
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from sklearn.decomposition import PCA
import torch

from src.dbn_utls.Study_generativity import Intersection_analysis, multiple_chimeras
from src.dbn_utls.misc import make_dir


def Plot_example_generated(input_dict,num_classes = 10,row_step = 10, dS=50, lblpad = 110, custom_steps = True, Show_classification = False, not_random_idxs = True):
    
    Generated_samples=input_dict['vis_states']
    nr_steps = Generated_samples.shape[2]
    img_side = int(np.sqrt(Generated_samples.shape[1]))

    if Show_classification ==True:
        Classifications = input_dict['Cl_pred_matrix']
    
    if custom_steps == True:
        steps=[3,5,10,25,50,100]
        rows=len(steps)
    else:
        steps = range(row_step,nr_steps+1,row_step) #controlla che funzioni
        rows = math.floor(nr_steps/row_step) 

    cols = Generated_samples.shape[0]
    fig_side = 25/num_classes
    if cols>num_classes:
        _, axis = plt.subplots(rows+1,num_classes, figsize=(25*(num_classes/num_classes),fig_side*(1+rows)))
    elif cols>1:
        _, axis = plt.subplots(rows+1,cols, figsize=(25*(cols/num_classes),fig_side*(1+rows)))
    else:
        _, axis = plt.subplots(rows+1,cols+1, figsize=(25*(cols/num_classes),fig_side*(1+rows)))

    if cols >= 10:
        if not_random_idxs ==True:
            random_numbers = range(num_classes)
        else:
            random_numbers = random.sample(range(cols), num_classes) # 10 random samples are selected
    else:
        random_numbers = random.sample(range(cols), cols) # 10 random samples are selected

    c=0
    for sample_idx in random_numbers: #per ogni sample selezionato
        # plotto la ricostruzione dopo uno step
        reconstructed_img= Generated_samples[sample_idx,:,0] #estraggo la prima immagine ricostruita per il particolare esempio (lbl può essere un nome un po fuorviante)
        reconstructed_img = reconstructed_img.view((img_side,img_side)).cpu() #ridimensiono l'immagine e muovo su CPU
        axis[0, c].tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
        axis[0, c].imshow(reconstructed_img , cmap = 'gray')
        if Show_classification==True:
            axis[0, c].set_title("Class {}".format(Classifications[sample_idx,0]), fontsize=dS)
        if c==0:
            ylabel = axis[0, c].set_ylabel("Step {}".format(1), fontsize=dS,rotation=0, labelpad=lblpad)

        axis[0, c].set_xticklabels([])
        axis[0, c].set_yticklabels([])
        axis[0, c].set_aspect('equal')

        #for idx,step in enumerate(range(row_step,nr_steps+1,row_step)): # idx = riga dove plotterò, step è il recostruction step che ci plotto
        for idx,step in enumerate(steps): # idx = riga dove plotterò, step è il recostruction step che ci plotto
            idx = idx+1 #sempre +1 perchè c'è sempre 1 step reconstruction

            #plotto la ricostruzione

            reconstructed_img= Generated_samples[sample_idx,:,step-1] #step-1 perchè 0 è la prima ricostruzione
            reconstructed_img = reconstructed_img.view((img_side,img_side)).cpu()
            axis[idx, c].tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
            axis[idx, c].imshow(reconstructed_img , cmap = 'gray')
            if Show_classification==True:
                axis[idx, c].set_title("Class {}".format(Classifications[sample_idx,step-1]), fontsize=dS)
            #axis[idx, lbl].set_title("Step {}".format(step) , fontsize=dS)
            if c==0:
                ylabel = axis[idx, c].set_ylabel("Step {}".format(step), fontsize=dS, rotation=0, labelpad=lblpad)

            axis[idx, c].set_xticklabels([])
            axis[idx, c].set_yticklabels([])
            axis[idx, c].set_aspect('equal')
        c=c+1
    #aggiusto gli spazi tra le immagini
    plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.2) 
    
    #plt.savefig("Reconstuct_plot.jpg") #il salvataggio è disabilitato

    plt.show()
    
    
def StateTimePlot(Trans_nr, T_mat_labels, lS=25):
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(Trans_nr, linewidth=0.5, annot=True, annot_kws={"size": lS}, 
                        square=True, cbar_kws={"shrink": .82},fmt='.1f', cmap='jet')
        if T_mat_labels==[]:
            T_mat_labels = [str(i) for i in range(len(Trans_nr))]
            ax.set_yticklabels(T_mat_labels)
            if len(Trans_nr) == 10:
                T_mat_labels.append('Non\ndigit')
        ax.set_xticklabels(T_mat_labels)
        ax.tick_params(axis='both', labelsize=lS)

        plt.xlabel('Class', fontsize = 25) # x-axis label with fontsize 15
        plt.ylabel('Biasing Class', fontsize = 25) # y-axis label with fontsize 15
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=lS)
        plt.show()


def Transition_mat_plot(Transition_matrix_rowNorm,T_mat_labels=[], lS=25):
    plt.figure(figsize=(15, 15))
    Transition_matrix=Transition_matrix_rowNorm*100
    ax = sns.heatmap(torch.round(Transition_matrix.cpu(), decimals=2), linewidth=0.5, 
                    annot=True, annot_kws={"size": lS},square=True,
                    cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
    plt.xlabel('To', fontsize = 25) # x-axis label with fontsize 15
    plt.ylabel('From', fontsize = 25) # y-axis label with fontsize 15
    if T_mat_labels==[]:
        T_mat_labels = [str(i) for i in range(len(Transition_matrix_rowNorm)-1)]
        if len(Transition_matrix_rowNorm) == 11:
            T_mat_labels.append('Non\ndigit')
    ax.set_xticklabels(T_mat_labels)
    ax.set_yticklabels(T_mat_labels)
    ax.tick_params(axis='both', labelsize=lS)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lS)
    plt.show()
    
def Cl_plot(axis,x,y,y_err=[],x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],
            Title = 'Classifier accuracy',l_sz=3, dS=30, color='g'):
    y=y.cpu()

    axis.plot(x, y, c = color, linewidth=l_sz)
    if y_err != []:
        y_err = y_err.cpu()
        axis.fill_between(x,y-y_err, y+y_err, color=color,
                alpha=0.3)
    axis.tick_params(axis='x', labelsize= dS)
    axis.tick_params(axis='y', labelsize= dS)
    axis.set_ylabel(y_lab,fontsize=dS)
    axis.set_ylim(lim_y)
    axis.set_xlabel(x_lab,fontsize=dS)
    axis.set_title(Title,fontsize=dS)

def Cl_plot_classwise(axis,cl_lbls,x,classwise_y,classwise_y_err=[], Num_classes=10,
            x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy - classwise',
            l_sz=3, dS= 30, cmap=cm.get_cmap('hsv')):
    c=0
    for digit in range(Num_classes):
        Color = cmap(c/256) 
        MEAN = classwise_y[digit,:].cpu()
        axis.plot(x, MEAN, c = Color, linewidth=l_sz)
        if classwise_y_err!=[]:
            SEM = classwise_y_err[digit,:].cpu()
            axis.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
                    alpha=0.3)
        c = c+25
    axis.legend(cl_lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
    axis.tick_params(axis='x', labelsize= dS)
    axis.tick_params(axis='y', labelsize= dS)
    axis.set_ylabel(y_lab,fontsize=dS)
    axis.set_ylim(lim_y)
    axis.set_xlabel(x_lab,fontsize=dS)
    #axis.set_title(Title,fontsize=dS)
    
def hist_pixel_act(avg_pixels_active_TrainMNIST,avg_activity_sampled_data,avg_activity_sampled_data_topK, n_bins = 20):
    plt.figure()
    plt.hist(avg_pixels_active_TrainMNIST.cpu(), bins=n_bins, color='blue', alpha=0.7,density=True, label='MNIST train set')  # You can adjust the number of bins as needed
    plt.hist(avg_activity_sampled_data.cpu(), bins=n_bins, color='red', alpha=0.7,density=True, label='Generated data - no correction')
    plt.hist(avg_activity_sampled_data_topK.cpu(), bins=n_bins, color='orange', alpha=0.7,density=True, label='Generated data - corrected')
    # Add labels and a title
    plt.xlabel('Average pixel activation')
    plt.ylabel('Relative frequency (%)')
    plt.legend()
    plt.show()

def plot_relearning(Readouts, yl = [0.1, 1],lab_sz = 50, leg_on =1, linewidth = 4, 
                    legend_dict = {'seq':'Sequential learning', 'int_origMNIST': 'Interleaved learning - experience replay', 
                    'int_chimeras': 'Interleaved learning - generative replay', 'int_rand': 'Interleaved learning - random'},
                    outpath = None):
    dotting = {'seq':'-', 'int_origMNIST': '--', 'int_chimeras': ':', 'int_rand': '-.'}
    fig, ax = plt.subplots(figsize = (20,20))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
    
    vars = Readouts.columns.tolist()
    for v in vars:
        if not(v=='retrain epoch'):
            lbl_dot = [[l, dotting[k]] for k,l in legend_dict.items() if k in v]
            lbl_dot = lbl_dot[0] if len(lbl_dot)>0 else [v,'-']
            c= 'r' if any(p == 'MNIST' for p in v.split('_')) else 'b'
            ax.plot(Readouts['retrain epoch'], Readouts[v], label=lbl_dot[0], 
                    color = c, linestyle = lbl_dot[1], lw=linewidth)
    ax.set_xlabel('Retraining epoch', fontsize=lab_sz)
    ax.set_ylabel('Readout', fontsize=lab_sz)
    ax.set_ylim(yl) 
    
    legend_dotting = {v:dotting[k] for k,v in legend_dict.items()}
    legend_labels = list(legend_dotting.keys())
    legend_handles = [plt.Line2D([], [], color='black', linestyle=style, lw=linewidth) for style in legend_dotting.values()]
    if leg_on ==1:
        ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=lab_sz, ncol=1)         
    ax.tick_params(labelsize=lab_sz)   
    if outpath:
        fig.savefig(outpath)  
    plt.show()
    
    
def plot_end_readout(Readout, mix_types=['rand, chimeras']):
    R_end = Readout.iloc[-1]
    result_dict = {}
    for column_name in R_end.index:
        parts = column_name.split('_')
        if 'rand' in column_name:
            return column_name
        if parts[0] == 'MNIST' and any(mt in column_name for mt in mix_types):
            k = parts[2] if len(parts)<5 else parts[2]+'_'+parts[3] #difference betw rand_k and others
            result_dict[k] = R_end[column_name]
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: int(re.search(r'\d+', item[0]).group()) if re.search(r'\d+', item[0]) else 0))

    plt.plot(list(sorted_dict.keys()), list(sorted_dict.values()))
    plt.xticks(rotation=45)
    plt.show()
    
    
def PCA_average(time_neuron_avg, n_components=2, title = None, out_dir = None):
    pca = PCA(n_components=n_components)
    pca.fit(time_neuron_avg)
    pca_timeseries = pca.transform(time_neuron_avg)
    explained_variance = pca.explained_variance_ratio_
    if n_components == 2:
        colors = np.linspace(1, pca_timeseries.shape[0]+1, pca_timeseries.shape[0])
        scatter = plt.scatter(pca_timeseries[:,0], pca_timeseries[:,1], c=colors, cmap='gray_r', edgecolors='black')
        plt.xlabel(f'PC1 - explained var: {explained_variance[0]*100:.2f} %')
        plt.ylabel(f'PC2 - explained var: {explained_variance[1]*100:.2f} %')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Gen steps')
        if title:
            plt.title(title)
        if out_dir:
            plt.savefig(out_dir)
        plt.show()
    return pca_timeseries, explained_variance


def plot_pca_between(avg_trials_pca,explained_variance, keys_of_interest = ['digits', '4,9', 'R']):
    
    if 'digits' in keys_of_interest:
        keys_of_interest.remove('digits')
        keys_of_interest.extend([str(j) for j in range(10)])
            
    cmap = plt.cm.get_cmap('jet', len(keys_of_interest))
    nr_gens = avg_trials_pca['R'].shape[0]
    legend_elements = []
    for i, key in enumerate(keys_of_interest):
        alphas = np.linspace(0.25, 1.0, nr_gens)
        trial = avg_trials_pca[key]
        for j in range(nr_gens):
            if j == 0:
                plt.text(trial[j,0], trial[j,1], key, color=cmap(i), fontsize=12)
            else:
                plt.scatter(trial[j,0], trial[j,1], c=[cmap(i)], alpha=alphas[j])  # inner color with varying alpha
                plt.scatter(trial[j,0], trial[j,1], c='none', edgecolors='black')  # edge color with no alpha
        

        # Create a Line2D object for the legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10, label=key))

    plt.xlabel(f'PC1 - explained var: {explained_variance[0]*100:.2f} %')
    plt.ylabel(f'PC2 - explained var: {explained_variance[1]*100:.2f} %')
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
    plt.show()       
    
def study_trajectories_version2(dbn, classifier, nr_gen_steps =100, topk_R = 63, topk_C_int = 111):
    directory = make_dir("~/Documents/GitHub/iDBN_Zambra22/pca_trajectories")
    avg_trials = {}
    all_trials = []
    all_labels = []
    #Label biasing
    hidden,lbls=dbn.label_biasing()
    result_dict = dbn.generate_from_hidden(hidden,nr_gen_steps =nr_gen_steps)

    for digit in range(dbn.Num_classes):
        idxs = lbls == digit
        timeseries = result_dict['hid_prob'][-1,idxs,:,:].cpu().numpy()
        timeseries = np.mean(timeseries, axis=0).T
        avg_trials[f'{digit}'] = timeseries
        all_trials.append(timeseries)
        all_labels.append(f'{digit}')
        
    #random biasing
    hidden_R=dbn.random_hidden_bias(topk = topk_R, size=(1000, 100), discrete = True)
    result_dict = dbn.generate_from_hidden(hidden_R,nr_gen_steps =nr_gen_steps)
    timeseries = result_dict['hid_prob'][-1,:,:,:].cpu().numpy()
    timeseries = np.mean(timeseries, axis=0).T
    avg_trials['R'] = timeseries
    all_trials.append(timeseries)
    all_labels.append('R')
    #intersection chimeras
    i_an = Intersection_analysis(dbn, top_k_Hidden=topk_C_int, nr_steps=100)
    i_an.do_intersection_analysis()
    _, _ , generated_dict = multiple_chimeras(dbn, classifier, sample_nr = 100, Ian = i_an, 
                            nr_gen_steps = nr_gen_steps, topk = topk_C_int, gather_visits = True, gather_visible = False)
    for k,v in generated_dict.items():
        for k2,v2 in v.items():
            if k2 == 'hid_prob':
                v2 = v2.T.cpu().numpy()
                avg_trials[k] = v2
                all_trials.append(v2)
                all_labels.append(k)
    n_trials = len(avg_trials)
    all_trials =  np.concatenate(list(avg_trials.values()), axis=0)    
    pca_timeseries, explained_variance = PCA_average(all_trials, n_components=2, title = f'All')
    trials = np.array_split(pca_timeseries, n_trials)
    avg_trials_pca = {k:trials[i] for i,k in enumerate(avg_trials.keys())} 
    
    return avg_trials_pca,explained_variance