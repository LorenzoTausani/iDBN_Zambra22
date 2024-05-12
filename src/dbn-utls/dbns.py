import os
import torch
import rbms
import random
from tqdm import tqdm


class NonLinear(torch.nn.Linear):
    '''UNDER MAINTENANCE'''
    def __init__(self, Nin, Nout):
        super(NonLinear, self).__init__()
        
        self.linear = torch.nn.Linear(Nin, Nout, bias = False)
        self.bias   = torch.nn.parameter.Parameter(torch.zeros(1, Nout))
    #end
    
    def forward(self, x):
        return self.linear(x).add(self.bias)
    #end
#end


class DBN(torch.nn.Module):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = 'cuda', last_layer_sz =1000):
        super(DBN, self).__init__()
        
        if dataset_id == 'MNIST' or dataset_id == 'EMNIST':
            self.rbm_layers = [
                rbms.RBM(784, 500, epochs,
                         layer_id = 0,  
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(500, 500, epochs,
                         layer_id = 1,
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(500, 2000, epochs, #ORIGINALLY 500,1000. 2000 in Zambra 2022
                         layer_id = 2, 
                         init_scheme = init_scheme, 
                         dataset_id = dataset_id)
            ]
        #end
        if dataset_id == 'CIFAR10':
            self.rbm_layers = [
                rbms.RBM(1024, 750, epochs,
                         layer_id = 0,  
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(750, 750, epochs,
                         layer_id = 1,
                         init_scheme = init_scheme,
                         dataset_id = dataset_id),
                rbms.RBM(750, 2000, epochs,
                         layer_id = 2, 
                         init_scheme = init_scheme, 
                         dataset_id = dataset_id)
            ]

        if 'CelebA' in dataset_id:
            Arch_MNIST = int(input('vuoi architettura come MNIST (1=si, 0 = no)'))
            if Arch_MNIST == 0:
                self.rbm_layers = [
                    rbms.RBM(4096, 2500, epochs,
                            layer_id = 0,  
                            init_scheme = init_scheme,
                            dataset_id = dataset_id),
                    rbms.RBM(2500, 2500, epochs,
                            layer_id = 1,
                            init_scheme = init_scheme,
                            dataset_id = dataset_id),                      
                    rbms.RBM(2500, 5250, epochs, 
                            layer_id = 2, 
                            init_scheme = init_scheme, 
                            dataset_id = dataset_id)
                ]
            else:
                self.rbm_layers = [
                    rbms.RBM(4096, 500, epochs,
                            layer_id = 0,  
                            init_scheme = init_scheme,
                            dataset_id = dataset_id),
                    rbms.RBM(500, 500, epochs,
                            layer_id = 1,
                            init_scheme = init_scheme,
                            dataset_id = dataset_id),                      
                    rbms.RBM(500, 1000, epochs, #dovrebbe essere 5250
                            layer_id = 2, 
                            init_scheme = init_scheme, 
                            dataset_id = dataset_id)
                ]
            #end            
        
        self.DEVICE = DEVICE
        self.to(DEVICE)
        self.alg_name = alg_name
        self.init_scheme = init_scheme
        self.dataset_id = dataset_id
        self.path_model = path_model
        self.top_layer_size = self.rbm_layers[-1].Nout
        self.depth = len(self.rbm_layers)
        self.idx_max_depth = self.depth - 1
        
        self.Num_classes = None
        self.classes = None
    #end
    
    def forward(self, v, only_forward = False):
        
        for rbm in self.rbm_layers:
            v, p_v = rbm(v)
        #end
        
        if only_forward:
            return p_v, v
        
        else:
            
            for rbm in reversed(self.rbm_layers):
                v, _ = rbm.backward(v)
            #end
            
            return v
        #end
    #end
    
    def test(self, Xtest, Ytest, mode = 'reproduction'):
        
        eval_fn = torch.nn.MSELoss(reduction = 'mean')
        
        if mode == 'reproduction':
            data = Xtest
        if mode == 'reconstruction':
            # corrupt and reconstruct
            # data = ...
            pass
        if mode == 'denoise':
            # data = ...
            # add torch.normal(mean, std, size).to(DEVICE)
            pass
        #end
        
        out_data = self(data)
        error = eval_fn(data, out_data)
        
        return error, out_data
    #end
    
    def get_name(self):
        
        algo_names = {'i' : 'iterative', 'g' : 'greedy', 'fs' : 'fullstack'}
        algo = algo_names[self.alg_name]
        name = f'dbn_{algo}_{self.init_scheme}_{self.dataset_id}'
        return name
    #end
    
    def save(self, run = None):
        
        for layer in self.rbm_layers:
            layer.flush_gradients()
            layer.to(torch.device('cpu'))
        #end
        
        name = self.get_name()
        if run is not None:
            name += f'_run{run}'
        #end
        torch.save(self.to(torch.device('cpu')),
                open(os.path.join(self.path_model, f'{name}.pkl'), 'wb'))
    #end
#end

    def invW4LB(self,train_dataset, L=[]):
        lbls = train_dataset['labels'].view(-1) # Flatten the labels in the training dataset
        # Get the number of batches and batch size from the training dataset
        nr_batches, BATCH_SIZE = train_dataset['data'].shape

        # If L is not provided, create a one-hot encoding matrix L for each label (i.e. num classes x examples)
        if L==[]:
            L = torch.zeros(self.Num_classes,lbls.shape[0], device = self.DEVICE)
            c=0
            for lbl in lbls:
                L[int(lbl),c]=1 #put =1 only the idx corresponding to the label of that example
                c=c+1
        else:
            L = L.view(40, -1) #for CelebA with all 40 labels (UNUSED)

        p_v, v = self(train_dataset['data'].cuda(), only_forward = True) #one step hidden layer of the training data by the model
        V_lin = v.view(nr_batches*BATCH_SIZE, self.top_layer_size)
        #I compute the inverse of the weight matrix of the linear classifier. weights_inv has shape (model.Num_classes x Hidden layer size (10 x 1000))
        weights_inv = torch.transpose(torch.matmul(torch.transpose(V_lin,0,1), torch.linalg.pinv(L)), 0, 1)
        self.weights_inv = weights_inv

        return weights_inv
    
    def getH_label_biasing(self, on_digits: int =1, topk: int = 149):
        # aim of this function is to implement the label biasing procedure described in
        # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full, getting
        # the activations in the hidden layer
        # Now i set the label vector from which i will obtain the hidden layer of interest 
        Biasing_vec = torch.zeros (self.Num_classes,1, device = self.DEVICE)
        Biasing_vec[on_digits] = 1
        #I compute the biased hidden vector as the matmul of the trasposed weights_inv and the biasing vec. gen_hidden will have size (Hidden layer size x 1)
        gen_hidden= torch.matmul(torch.transpose(self.weights_inv,0,1), Biasing_vec)

        if topk>-1: #ATTENZIONE: label biasing con più di una label attiva (e.g. on_digits=[4,6]) funziona UNICAMENTE con topk>-1 (i.e. attivando le top k unità piu attive e silenziando le altre)
            #In caso contrario da errore CUDA non meglio specificato
            H = torch.zeros_like(gen_hidden, device = self.DEVICE) #crate an empty array of the same shape of gen_hidden 
            for c in range(gen_hidden.shape[1]): # for each example in gen_hidden...
                top_indices = torch.topk(gen_hidden[:,c], k=topk).indices # compute the most active indexes
                H[top_indices,c] = 1 #set the most active indexes to 0
            gen_hidden = H # gen_hidden is now binary (1 or 0)

        return gen_hidden
    
    def label_biasing(self, topk: int = 149,n_reps: int = 100):
        LB_hidden = torch.cat([self.getH_label_biasing(on_digits=dig, topk=topk) 
                            for dig in range(self.Num_classes)], dim=1)
        #LB_labels is a tensor containing the labels of the generated examples.
        #This will help for computing classification accuracy
        LB_labels = torch.tensor(range(self.Num_classes), device = self.DEVICE)
        #let's repeat each row of LB_hidden n_reps times 
        #(to compute some statistics about the generation)
        LB_hidden = LB_hidden.repeat(1,n_reps)
        LB_labels=LB_labels.repeat(n_reps)
        return LB_hidden, LB_labels
    
    def random_hidden_bias(self, n: int, size: tuple):
        hidden = torch.zeros(size)
        for i in range(size[0]):
            indices = random.sample(range(size[1]), n)
            hidden[i, indices] = 1
        return hidden
    
    def top_down_1step(self,gen_step,hid_prob,hid_states,vis_prob,vis_states):
        v = hid_states[self.idx_max_depth,:,:,gen_step]
        c=1 # counter of layer depth
        for rbm in reversed(self.rbm_layers):
            #if the layer selected is not the one above the visible layer (i.e. below there is another hidden layer)
            if c<self.depth: 
                p_v, v = rbm.backward(v)
                layer_size = v.shape[1]
                #i store the hid prob and state of the layer below
                # (note: in case the hidden layer it's smaller, i store the vals up to the size of the hidden layer
                # (i.e. :layer_size))
                hid_prob[self.idx_max_depth-c,:,:layer_size,gen_step]  = p_v 
                hid_states[self.idx_max_depth-c,:,:layer_size,gen_step]  = v
            else: #if the layer below is the visible later
                v, p_v = rbm.backward(v) #passo la probabilità (che in questo caso è v) dopo
                layer_size = v.shape[1]
                #i store the visible state and probabilities
                vis_prob[:,:,gen_step]  = v
                vis_states[:,:,gen_step]  = v
            c=c+1
        return hid_prob,hid_states,vis_prob,vis_states
    
    def generate_from_hidden(self, input_hid_prob, nr_gen_steps: int=1):
        #DOVREBBE ESSERE UGUALE A ZAMBRA, ma manca un controllo deterministico causa
        #samping bernoulliano. controlla che tornino i risultati di generazione
        #input_hid_prob has size Nr_hidden_units x num_cases. Therefore i transpose it
        input_hid_prob = torch.transpose(input_hid_prob,0,1)
        #numcases = numbers of samples to generate
        numcases, hidden_layer_size = input_hid_prob.size()
        vis_layerSize = self.rbm_layers[0].Nin
        # Initialize tensors to store hidden and visible probabilities and states
        # hid prob/states : nr layers x numbers of samples to generate x size of the hidden layer x number of generation steps
        # vis prob/states : numbers of samples to generate x size of the visible layer x number of generation steps
        hid_prob = torch.zeros(len(self.rbm_layers),numcases,hidden_layer_size, nr_gen_steps, device=self.DEVICE)
        hid_states = torch.zeros(len(self.rbm_layers), numcases,hidden_layer_size, nr_gen_steps, device=self.DEVICE)
        vis_prob = torch.zeros(numcases, vis_layerSize, nr_gen_steps, device=self.DEVICE)
        vis_states = torch.zeros(numcases ,vis_layerSize, nr_gen_steps, device=self.DEVICE)

        for gen_step in range(0, nr_gen_steps): #for each generation step...
            if gen_step==0: #if it is the 1st step of generation...
                #the hidden probability is the one in the input
                hid_prob[self.idx_max_depth,:,:,gen_step]  = input_hid_prob
                hid_states[self.idx_max_depth,:,:,gen_step]  = input_hid_prob
                #da qui diverge rispetto al vecchio generate_from_hidden
            else:
                v = vis_states[:,:,gen_step-1]
                #do the forward up to the last layer
                for rbm in self.rbm_layers:
                    p_v, v = rbm(v)
                #i store the probability and state of the upper layer
                hid_prob[self.idx_max_depth,:,:,gen_step]  = p_v 
                hid_states[self.idx_max_depth,:,:,gen_step]  = v
            hid_prob,hid_states,vis_prob,vis_states = self.top_down_1step(gen_step,hid_prob,hid_states,vis_prob,vis_states)
                    
        #the result dict will contain the output of the whole generation process
        result_dict = dict(); 
        result_dict['hid_states'] = hid_states
        result_dict['vis_states'] = vis_states
        result_dict['hid_prob'] = hid_prob
        result_dict['vis_prob'] = vis_prob

        return result_dict
        
class gDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = 'cuda'):
        super(gDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = DEVICE)
        
        self.algo = 'g'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams, readout = False):
        
        for rbm in self.rbm_layers:
            
            print(f'--Layer {rbm.layer_id}')
            _Xtrain, _Xtest = rbm.train(Xtrain, Xtest, Ytrain, Ytest, 
                                        lparams, readout = readout)
            Xtrain = _Xtrain
            Xtest = _Xtest
        #end
    #end
#end

class iDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = 'cuda', last_layer_sz=1000):
        super(iDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs,DEVICE = DEVICE, last_layer_sz =1000)
        
        self.algo = 'i'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams, 
            readout = False, num_discr = False):
        
        if self.Num_classes is None and not(self.dataset_id == 'CelebA'):
            self.classes = torch.unique(Ytrain.view(-1))
            self.Num_classes = len(self.classes) #with MNIST = 10
        
        for rbm in self.rbm_layers:
            rbm.dW = torch.zeros_like(rbm.W)
            rbm.da = torch.zeros_like(rbm.a)
            rbm.db = torch.zeros_like(rbm.b)
        #end
        
        for epoch in range(lparams['EPOCHS']):
            
            print(f'--Epoch {epoch}')
            self.current_epoch = epoch
            self.epochs_loop(Xtrain, Xtest, Ytrain, Ytest, lparams, readout)
            
        #end EPOCHS
        
    #end
    
    def epochs_loop(self, Xtrain, Xtest, Ytrain, Ytest, lparams, readout):
        
        n_train_batches = Xtrain.shape[0]
        n_test_batches = Xtest.shape[0]
        batch_size = Xtrain.shape[1]
        
        for rbm in self.rbm_layers:
            
            _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
            _Xtest = torch.zeros((n_test_batches, batch_size, rbm.Nout))
            rbm.current_epoch = self.current_epoch
            train_loss = 0.
            
            _Xtest, _ = rbm(Xtest)
            
            batch_indices = list(range(n_train_batches))
            random.shuffle(batch_indices)
            with tqdm(batch_indices, unit = 'Batch') as tlayer:
                for idx, n in enumerate(tlayer):
                    
                    tlayer.set_description(f'Layer {rbm.layer_id}')
                    _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])
                    pos_v = Xtrain[n,:,:]
                    loss = rbm.CD_params_update(pos_v, lparams)
                    
                    train_loss += loss
                    tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                #end BATCHES
            #end WITH
            
            rbm.loss_profile[self.current_epoch] = train_loss.div(n_train_batches)
            
            if readout:
                readout_acc = rbm.get_readout(_Xtrain, _Xtest, Ytrain, Ytest)
                print(f'Readout accuracy = {readout_acc*100:.2f}')
                rbm.acc_profile[self.current_epoch] = readout_acc
            #end
            
            Xtrain = _Xtrain.clone()
            Xtest  = _Xtest.clone()
        #end LAYERS
    #end
#end

class fsDBN(DBN):
    
    def __init__(self, alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = 'cuda'):
        super(fsDBN, self).__init__(alg_name, dataset_id, init_scheme, path_model, epochs, DEVICE = DEVICE)
        
        self.algo = 'fs'
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams):
        
        for rbm in self.rbm_layers:
            rbm.dW = torch.zeros_like(rbm.W)
            rbm.db = torch.zeros_like(rbm.b)
            rbm.da = torch.zeros_like(rbm.a)
        #end
        
        for epoch in range(lparams['EPOCHS']):
            
            print(f'--Epoch {epoch}')
            self.current_epoch = epoch
            self.epochs_loop(Xtrain, Xtest, Ytrain, Ytest, lparams)
        #end
        
        for rbm in self.rbm_layers:
            rbm.delete_field('act_topdown')
        #end
    #end
    
    def epochs_loop(self, Xtrain, Xtest, Ytrain, Ytest, lparams):
        
        n_train_batches = Xtrain.shape[0]
        batch_size = Xtrain.shape[1]
        
        # Bottom-up loop
        p_act, act = self(Xtrain, only_forward = True)
        
        # Top-down loop
        for rbm in reversed(self.rbm_layers):
            
            rbm.save_topdown_act( (p_act, act) )
            p_act, act = rbm.backward(act)
        #end
        
        # Training loop
        for rbm in self.rbm_layers:
            
            _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
            rbm.current_epoch = self.current_epoch
            train_loss = 0.
            
            batch_indices = list(range(n_train_batches))
            random.shuffle(batch_indices)
            with tqdm(batch_indices, unit = 'Batch') as tlayer:
                for idx, n in enumerate(tlayer):
                    
                    tlayer.set_description(f'Layer {rbm.layer_id}')
                    _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])
                    pos_v = Xtrain[n,:,:]
                    topdown_pact, topdown_act = rbm.get_topdown_act()
                    loss = rbm.CD_params_update(pos_v, lparams, 
                            hidden_saved = (topdown_pact[n,:,:], topdown_act[n,:,:]))
                    
                    train_loss += loss
                    tlayer.set_postfix(MSE = train_loss.div(idx + 1).item())
                #end BATCHES
            #end WITH
            
            rbm.loss_profile[self.current_epoch] = train_loss.div(n_train_batches)
            Xtrain = _Xtrain.clone()
        #end LAYERS
        
#end


