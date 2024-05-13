import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import numpy as np

from plots import Cl_plot, Cl_plot_classwise, StateTimePlot, Transition_mat_plot

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels,batch_norm=False):

        super().__init__()

        conv2_params = {'kernel_size': (3, 3),
                        'stride'     : (1, 1),
                        'padding'   : 1
                        }

        noop = lambda x : x

        self._batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels , **conv2_params)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn1 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, **conv2_params)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn2 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    @property
    def batch_norm(self):
        return self._batch_norm

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.max_pooling(x)

        return x



class VGG16(nn.Module):

  def __init__(self, input_size, num_classes=11,batch_norm=False):
    super(VGG16, self).__init__()

    self.in_channels,self.in_width,self.in_height = input_size

    self.block_1 = VGGBlock(self.in_channels,64,batch_norm=batch_norm)
    self.block_2 = VGGBlock(64, 128,batch_norm=batch_norm)
    self.block_3 = VGGBlock(128, 256,batch_norm=batch_norm)
    self.block_4 = VGGBlock(256,512,batch_norm=batch_norm)

    self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

  @property
  def input_size(self):
      return self.in_channels,self.in_width,self.in_height

  def forward(self, x):

    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    x = self.block_4(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)

    return x
  
def CelebA_ResNet_classifier(ds_loaders = [], num_classes = 4, num_epochs = 20, learning_rate = 0.001, filename=''):
    Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/' # Define the directory path where the model and data are stored
    classifier = models.resnet18(pretrained=True) # Initialize a ResNet-18 model pretrained on ImageNet
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)  # Modify the fully connected layer (classifier head) to have the desired number of output classes
    if ds_loaders == []:  # Load a model already trained if ds_loaders is empty
        classifier.load_state_dict(torch.load(Zambra_folder_drive+filename)) 
        classifier = classifier.to('cuda') # Move the model to the GPU if available
    else:
        # Define the data loaders for training and validation
        train_loader = ds_loaders[0]
        val_loader = ds_loaders[1]
        # Enable gradient computation for all model parameters (fine-tuning)
        for param in classifier.parameters():
            param.requires_grad = True
        # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        # Move the model and loss function to the GPU if available
        classifier = classifier.to('cuda')
        criterion = criterion.to('cuda')

        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            correct = 0 #counter of the total number of correct predictions
            total = 0 #counter of the total number of examples observed

            # Training
            classifier.train()
            for images, labels in train_loader:
                images = images.to('cuda')
                labels = labels.to('cuda')
                labels = labels.to(torch.long)  # Convert labels to long data type
                optimizer.zero_grad()  #this line sets the gradients of all the model's parameters to zero.
                outputs = classifier(images) #obtain the predictions of the classifier on images
                loss = criterion(outputs, labels) #and compute the loss
                loss.backward() #backward() computes the gradients of that tensor with respect to all the tensors that were used to compute it.
                #it starts the process of computing gradients by tracing back through the computational graph, from the loss value all the way to the model's parameters using the chain rule
                optimizer.step() #step(),when called on an optimizer, performs a single optimization step, which includes updating the model's parameters based on the computed gradients.
                train_loss += loss.item() * images.size(0) # Accumulate the training loss by scaling it with the batch size
                _, predicted = torch.max(outputs.data, 1) # Compute the predicted class labels for the current batch
                # Update total and correct predictions counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item() 
            train_loss = train_loss / len(train_loader.dataset) # Calculate the average training loss over all batches
            train_acc = correct / total # Calculate the training accuracy by dividing correct predictions by the total number of samples

            # Validation
            classifier.eval() #put the classifier into evaluation mode
            with torch.no_grad(): #When entering this block, it temporarily sets a flag that tells PyTorch not to track gradients for tensor operations inside the block.
                #you do the same operations performed in training on the validation set
                for images, labels in val_loader:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                    labels = labels.to(torch.long) 
                    outputs = classifier(images) 
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = correct / total

            # print training and validation loss and accuracy
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
                .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
        torch.save(classifier.state_dict(), Zambra_folder_drive+filename) #save the classifier
            
    return classifier


  
def Classifier_accuracy(input_dict, classifier,model,labels=[], cl_lbls =[],
                      Batch_sz= 100, plot=1, dS=30, l_sz=3):
  classifier = classifier.to(model.DEVICE)
  input_data = input_dict['vis_states']
  #input_data = nr_examples x 784 (i.e. image size) x nr_steps
  nr_samples,img_sz_1D ,nr_gen_steps = input_data.size()
  image_side = int(np.sqrt(img_sz_1D)) 
  #Initialization of the output variables
  Cl_pred_matrix = torch.zeros(nr_samples,nr_gen_steps, device=model.DEVICE) 
  classwise_acc = torch.zeros(model.Num_classes,nr_gen_steps, device=model.DEVICE)  
  acc = torch.zeros(nr_gen_steps) #len = number of generation steps
  if labels==[]:
    labels = torch.zeros(nr_samples, device=model.DEVICE)

  for step in range(nr_gen_steps):#i.e for each generation step
    V = input_data[:,:,step] # extract the visible of each example at that generation step 'step'
    #i change the size of the tensor: from nr_samples x 784 to nr_samples x 1 x 28 x 28 (for MNIST)
    V = torch.unsqueeze(V.view((nr_samples,image_side,image_side)),1) 
    if image_side==28: #MNIST
      V_int = F.interpolate(V, size=(32, 32), mode='bicubic', align_corners=False) # tensor with dimensionality nr_samples x 1 x 32 x 32
    elif image_side==64: #CelebA
      V_int = F.interpolate(V, size=(224, 224), mode='bicubic', align_corners=False) 
      V_int = V_int.repeat(1, 3, 1, 1) #this should repeat the tensor 3 times, to account for RGB channels
    #the following operations are done with batching, in order to avoid using all the GPU on colab
    _dataset = torch.utils.data.TensorDataset(V_int.to('cuda'),labels) # create your datset
    if Batch_sz > nr_samples: # if batch size is bigger than input size...
      Batch_sz = nr_samples
    _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=Batch_sz,drop_last = True) # create your dataloader
    
    index = 0
    acc_v = torch.zeros(math.floor(nr_samples/Batch_sz))
    n_it = 0
    for (input, lbls) in _dataloader: #for each batch
      with torch.no_grad():
        pred_vals=classifier(input) #predictions of the classifier
      _, inds = torch.max(pred_vals,dim=1) #find the predicted classes
      Cl_pred_matrix[index:index+Batch_sz,step] = inds #store the predictions into Cl_pred_matrix
      #compute the accuracy for the specific iteration
      acc_v[n_it] = torch.sum(inds.to(model.DEVICE)==lbls)/input.size()[0] 
      n_it = n_it+1 #update the number of iterations of batching
      index = index+ Batch_sz #the index is updated for the next batch
    #accuracy of a generation step is the average accuracy between batches
    acc[step] = torch.mean(acc_v) 
    
    for digit in range(model.Num_classes): #for each class
      l = torch.where(labels == digit) #check the indices of that particular class
      inds_digit = Cl_pred_matrix[l[0],step] #finds the predictions for that specific class in that generation step
      #compute the classwise accuracy for that particular digit
      classwise_acc[digit,step] = torch.sum(inds_digit.to(model.DEVICE)==labels[l[0]])/l[0].size()[0] 

  if plot == 1:
      cmap = cm.get_cmap('hsv') #colormap used
      if cl_lbls==[]:
        cl_lbls = range(model.Num_classes)
      x = range(1,nr_gen_steps+1)
      if not(image_side==64):
        _, axis = plt.subplots(2, 2, figsize=(20,15))
        Cl_plot(axis[0,0],x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', 
                lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
        Cl_plot_classwise(axis[1,0],cl_lbls,x,classwise_acc,x_lab='Generation step',
                y_lab='Accuracy',Num_classes=model.Num_classes, lim_y = [0,1],l_sz=l_sz, dS= dS, cmap=cmap)
      else:
        _, axis = plt.subplots(2, figsize=(10,15))
        Cl_plot(axis[0],x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', 
                lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
        Cl_plot_classwise(axis[1],cl_lbls,x,classwise_acc,x_lab='Generation step',
                y_lab='Accuracy',Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Classifier accuracy - classwise',l_sz=l_sz, dS= dS, cmap=cmap)

      plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.4) 
  elif plot==2:
      
      cmap = cm.get_cmap('hsv')
      if cl_lbls==[]:
        cl_lbls = range(model.Num_classes)
      x = range(1,nr_gen_steps+1)

      _, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot(axis,x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', 
              lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
      _, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot_classwise(axis,cl_lbls,x,classwise_acc,x_lab='Generation step',
              y_lab='Accuracy', Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Classifier accuracy - classwise',l_sz=l_sz, dS= dS, cmap=cmap)

  #the output is the input dictionary ernriched with new keys
  input_dict['Cl_pred_matrix'] = Cl_pred_matrix
  input_dict['Cl_accuracy'] = acc
  input_dict['classwise_acc'] = classwise_acc

  return input_dict

def classification_metrics(dict_classifier,model,test_labels=[], Plot=1, dS = 30, rounding=2, T_mat_labels=[], Ian=0):
  '''
  dict_classifier: classifier dictionary. It contains, among other things, the predictions of the classifier
  model = the DBN model
  test_labels = groundtruth labels
  '''
  #+1 is for MNIST, because we consider also the NON DIGIT class
  nr_states = model.Num_classes + 1 if model.Num_classes == 10 else model.Num_classes
  #classes predicted by the classifier (nr examples x nr generation steps)
  Cl_pred_matrix=dict_classifier['Cl_pred_matrix'] 
  nr_ex=dict_classifier['Cl_pred_matrix'].size()[0] #number of examples
  #Transition matrix initialization
  Transition_matrix = torch.zeros((nr_states,nr_states)) 

  for row in Cl_pred_matrix: #for every example classifications
    for nr_genStep in range(1,len(row)): #for each generation step
      Transition_matrix[row[nr_genStep-1].long(),row[nr_genStep].long()] += 1 #add +1 to the entry of the Transition matrix
      #corresponding to the row of the digit of the previous generation step [nr_genStep-1], 
      #and the column corresponding to the digit of the  current digit [nr_genStep]
      #IN OTHER WORDS, THE TRANSITION MATRIX IS ESTIMATED FROM ALL THE TRANSITIONS IN THE GENERATED DATASET
  #normalize each row of the transition matrix by its sum
  Transition_matrix_rowNorm = torch.div(Transition_matrix, torch.sum(Transition_matrix, dim=1, keepdim=True))

  # Create the list of categories of transitions to a certain digit
  #this is a string list containing all states (included non digit). They will be the columns of the output dataframes
  to_list = [str(digit) if digit < model.Num_classes else 'Non-digit' for digit in range(nr_states)]

  columns = ['Nr_visited_states','Nr_transitions'] + to_list #the columns of the dataframes created below
  #i create two dataframes: one for the means, the other for its relative errors (SEM)
  if test_labels == []:
      df_average = pd.DataFrame(columns=columns)
      df_sem = pd.DataFrame(columns=columns)
  else:
      index = range(model.Num_classes) 
      df_average = pd.DataFrame(index=index, columns=columns)
      df_sem = pd.DataFrame(index=index, columns=columns)


  for digit in range(model.Num_classes): #for each digit...
    if test_labels!=[]:
      digit_idx = test_labels==digit #i find the position of the digit/class among the groundtruth labels...
      Vis_digit = dict_classifier['Cl_pred_matrix'][digit_idx,:] #i find the predictions of the classifier in that positions
    else:
      Vis_digit = dict_classifier['Cl_pred_matrix'] #i just take all predictions together
    nr_visited_states_list =[] #here i will list the visited states
    nr_transitions_list =[] #here i will list the number of transitions that occurred
    to_digits_mat = torch.zeros(Vis_digit.size()[0],nr_states) #this is a matrix with nr.rows=nr.examples, nr.cols = 10+1 (nr. digits+no digits category)
    
    for nr_ex,example in enumerate(Vis_digit): # example=tensor of the reconstructions category guessed by the classifier in a single example (i.e. row-wise)
      no10_example = example[example!=10] #here are all generations different from nondigit
      # find all the states (digits) visited by the RBM (NOTE:I DO NOT COUNT 10(non-digit) HERE)
      nr_visited_states = len(torch.unique(no10_example))
      #transitions are the states sequentially explored during the generation
      transitions,counts = torch.unique_consecutive(no10_example, return_counts=True) 
      nr_transitions = len(transitions)
      to_digits = torch.zeros(nr_states) 
      #now i include 10(non-digit) in the transition count
      transitions,counts = torch.unique_consecutive(example,return_counts=True) 
      visited_states = torch.unique(example) #and in the nr of visited states
      # below, for all states visited, i get the nr of steps in which the state was explored
      for state in visited_states:
        idx_state= transitions == state
        to_digits[state.to(torch.long)] = torch.sum(counts[idx_state])

      nr_visited_states_list.append(nr_visited_states)
      nr_transitions_list.append(nr_transitions)
      to_digits_mat[nr_ex,:] = to_digits

    df_average.at[digit,'Nr_visited_states'] = round(sum(nr_visited_states_list)/len(nr_visited_states_list),2)
    df_average.at[digit,'Nr_transitions'] = round(sum(nr_transitions_list)/len(nr_transitions_list),2)
    df_average.iloc[digit,2:] = torch.round(torch.mean(to_digits_mat.cpu(),0),decimals=2)

    df_sem.at[digit,'Nr_visited_states'] = round(np.std(nr_visited_states_list)/math.sqrt(len(nr_visited_states_list)),2)
    df_sem.at[digit,'Nr_transitions'] = round(np.std(nr_transitions_list)/math.sqrt(len(nr_transitions_list)),2)
    df_sem.iloc[digit,2:] = torch.round(torch.std(to_digits_mat.cpu(),0)/math.sqrt(to_digits_mat.size()[0]),decimals=2)
  
  #ratio tra passaggi alla classe giusta e la seconda classe di più alta frequenza
  # to_mat = df_average.iloc[:, 2:-1]
  # sem_mat = df_sem.iloc[:, 2:-1]

  if Plot==1:
        
        if test_labels!=[]:
          df_average.plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:, ['Nr_visited_states', 'Nr_transitions']],figsize=(30,10),fontsize=dS)
          plt.xlabel("Class",fontsize=dS)
          if not(model.Num_classes==10):
            plt.xticks(range(model.Num_classes), T_mat_labels, rotation=0)

        else:
          df_average.iloc[0:1].plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:,['Nr_visited_states', 'Nr_transitions']],xticks=[], figsize=(20,10),fontsize=dS)
          
        #plt.title("Classification_metrics-1",fontsize=dS)
        
        plt.ylabel("Number of states",fontsize=dS)
        plt.ylim([0,model.Num_classes])
        plt.legend(["Visited states", "Transitions"], bbox_to_anchor=(0.73,1), loc="upper left", fontsize=dS-dS/3)
        

        #NEW PLOT(PER BRAIN INFORMATICS)
        if Ian ==0:
          lS=25
          Trans_nr = df_average.iloc[:, 2:]
          Trans_nr = Trans_nr.apply(pd.to_numeric)
          Trans_nr = Trans_nr.round(rounding)
          Trans_nr = np.array(Trans_nr)

          Trans_nr_err = df_sem.iloc[:, 2:]
          Trans_nr_err = Trans_nr_err.apply(pd.to_numeric)
          Trans_nr_err = Trans_nr_err.round(rounding)
          Trans_nr_err = np.array(Trans_nr_err)

          StateTimePlot(Trans_nr, T_mat_labels, lS=lS)

          #plot of the transition matrix
          Transition_mat_plot(Transition_matrix_rowNorm,T_mat_labels, lS=lS)
        else:
          
          #OLD PLOT: AVERAGE STATES VISITED BEGINNING FROM A CERTAIN LABEL BIASING DIGIT
          #QUESTO è IL PLOT MOSTRATO IN TESI
          cmap = cm.get_cmap('hsv')
          newcolors = cmap(np.linspace(0, 1, 256))
          black = np.array([0.1, 0.1, 0.1, 1])
          newcolors[-25:, :] = black
          newcmp = ListedColormap(newcolors)
          
          if test_labels!=[]:
            df_average.plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp)
            plt.xlabel("Digit",fontsize=dS)       
          else:
            df_average.iloc[0:1].plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp,xticks=[])

          #plt.title("Classification_metrics-2",fontsize=dS)
          
          plt.ylabel("Average nr of steps",fontsize=dS)
          plt.ylim([0,50])
          #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS)

          # Put a legend below current axis
          plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=4, fontsize=dS)
          

  return df_average, df_sem, Transition_matrix_rowNorm



