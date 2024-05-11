import math
import random
from matplotlib import pyplot as plt
import numpy as np


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

