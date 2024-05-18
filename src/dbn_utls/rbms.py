
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score


class RBM(torch.nn.Module):
    
    def __init__(self, Nin, Nout, epochs,
                 layer_id = None,
                 dataset_id = 'mnist', 
                 init_scheme = 'normal',
                 scale_factor = 0.01):
        super(RBM, self).__init__()
        
        # Shape and name specs
        self.layer_id = layer_id
        self.Nin      = Nin
        self.Nout     = Nout
        
        # Train specs
        self.loss_fn = torch.nn.MSELoss(reduction = 'mean')
        
        # Fields that may be useful
        self.current_epoch = None
        self.loss_profile = np.zeros(epochs)
        self.acc_profile = np.zeros(epochs)
        
        # params init
        if dataset_id == 'MNIST' :
            if init_scheme == 'normal':
                scale_factor = 0.01
            elif init_scheme == 'glorot':
                scale_factor = 1
            #end
        if dataset_id == 'SZ': 
            scale_factor = 0.1
        #end
        
        if init_scheme == 'normal':
            self.W = torch.nn.init.normal_(torch.empty((Nout, Nin)),
                                           mean = 0, std = 1) * scale_factor
        elif init_scheme == 'glorot':
            self.W = torch.nn.init.xavier_normal_(torch.empty((Nout, Nin))) * scale_factor
        #end
        
        self.a = torch.zeros((1, Nin))
        self.b = torch.zeros((1, Nout))
    #end
    
    def flush_gradients(self):
        
        delattr(self, 'dW')
        delattr(self, 'da')
        delattr(self, 'db')
    #end
    
    def delete_field(self, field):
        
        delattr(self, field)
    #end
    
    def forward(self, v):
        
        prob = torch.sigmoid(torch.matmul(v, self.W.t()).add(self.b))
        act = torch.bernoulli(prob)
        return prob, act
    #end
    
    def backward(self, h):
        
        prob = torch.sigmoid(torch.matmul(h, self.W).add(self.a))
        act = torch.bernoulli(prob)
        return prob, act
    #end
    
    def Gibbs_sampling(self, v):
        
        p_h, h = self.forward(v)
        p_v, v = self.backward(h)
        p_h, h = self.forward(v)
        
        return p_h, v, p_v            
    #end
    
    def train(self, Xtrain, Xtest, Ytrain, Ytest, lparams,
              readout = False, flush_grad_train_end = False):
        
        n_train_batches = Xtrain.shape[0]
        n_test_batches  = Xtest.shape[0]
        batch_size      = lparams['BATCH_SIZE']
        
        self.loss_profile = np.zeros(lparams['EPOCHS'])
        self.acc_profile  = np.zeros(lparams['EPOCHS'])
        
        # initialize gradients
        self.dW = torch.zeros_like(self.W)
        self.da = torch.zeros_like(self.a)
        self.db = torch.zeros_like(self.b)
        
        _Xtrain = torch.zeros((n_train_batches, batch_size, self.Nout))
        _Xtest  = torch.zeros((n_test_batches, batch_size, self.Nout))
        
        for epoch in range(lparams['EPOCHS']):
            
            self.current_epoch = epoch
            
            train_loss = 0.
            
            batch_indices = list(range(n_train_batches))
            random.shuffle(batch_indices)
            
            with tqdm(batch_indices, unit = 'Batch') as tepoch:
                for idx, n in enumerate(tepoch):
                    
                    tepoch.set_description(f'Epoch {epoch}')
                    pos_v = Xtrain[n,:,:]
                    
                    loss = self.CD_params_update(pos_v, lparams)
                    train_loss += loss
                    
                    _Xtrain[n,:,:], _ = self.forward(pos_v)
                    
                    tepoch.set_postfix(MSE = train_loss.div(idx + 1).item())
                #end BATCHES
            #end WITH
            
            self.loss_profile[self.current_epoch] = train_loss.div(n_train_batches)
            _Xtest, _ = self.forward(Xtest)
            
            if readout:
                readout_acc = self.get_readout(_Xtrain, _Xtest, Ytrain, Ytest)
                print(f'Readout accuracy = {readout_acc*100:.2f}')
                self.acc_profile[self.current_epoch] = readout_acc
            #end
        #end EPOCHS
        
        if flush_grad_train_end:
            self.flush_gradients()
        #end
        
        return _Xtrain, _Xtest
    #end
    
    def CD_params_update(self, pos_v, lparams, hidden_saved = None):
        
        momenta    = [lparams['INIT_MOMENTUM'], lparams['FINAL_MOMENTUM']]
        learn_rate = lparams['LEARNING_RATE']
        penalty    = lparams['WEIGHT_PENALTY']
        batch_size = lparams['BATCH_SIZE']
        
        if hidden_saved is None:
            pos_ph, pos_h = self.forward(pos_v)
            neg_ph, neg_v, neg_pv = self.Gibbs_sampling(pos_v)
        else:
            pos_ph, pos_h = hidden_saved
            neg_pv, neg_v = self.backward(pos_h)
            neg_ph, neg_h = self.forward(neg_v)
        #end
        
        pos_dW = torch.matmul(pos_v.t(), pos_ph).t().div(batch_size)
        pos_da = torch.sum(pos_v, dim = 0).div(batch_size)
        pos_db = torch.sum(pos_ph, dim = 0).div(batch_size)
        
        neg_dW = torch.matmul(neg_v.t(), neg_ph).t().div(batch_size)
        neg_da = torch.sum(neg_v, dim = 0).div(batch_size)
        neg_db = torch.sum(neg_ph, dim = 0).div(batch_size)
        
        if self.current_epoch < 5:
            momentum = momenta[0]
        else:
            momentum = momenta[1]
        #end
        
        self.dW = momentum * self.dW + learn_rate * ((pos_dW - neg_dW) - penalty * self.W)
        self.da = momentum * self.da + learn_rate * (pos_da - neg_da)
        self.db = momentum * self.db + learn_rate * (pos_db - neg_db)
        
        self.W = self.W + self.dW
        self.a = self.a + self.da
        self.b = self.b + self.db
        
        loss = self.loss_fn(pos_v, neg_pv)
        return loss
    #end
    
    def get_readout(self, Xtrain, Xtest, Ytrain, Ytest):
        
        n_feat = Xtrain.shape[-1]
        x_train = Xtrain.cpu().numpy().reshape(-1, n_feat)
        y_train = Ytrain.cpu().numpy().flatten()
        x_test  = Xtest.cpu().numpy().reshape(-1, n_feat)
        y_test  = Ytest.cpu().numpy().flatten()
        
        classifier = RidgeClassifier().fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        
        return accuracy_score(y_test, y_pred), classifier
    #end
    
    def save_topdown_act(self, act_hidden):
        
        self.act_topdown = act_hidden
    #end
    
    def get_topdown_act(self):
        
        return self.act_topdown
    #end
    
    def energy_f(self, hid_states, vis_states):

            sum_h_v_W = torch.zeros(hid_states.size()[0],1).to(self.DEVICE)
            m1=torch.matmul(vis_states,self.W)
            m2 = torch.matmul(m1,torch.transpose(hid_states,0,1))
            sum_h_v_W = torch.diagonal(m2*torch.eye(m2.size()[0]).to(self.DEVICE))
            state_energy = -torch.matmul(vis_states,torch.transpose(self.a,0,1)) - torch.matmul(hid_states,torch.transpose(self.b,0,1)) -sum_h_v_W.unsqueeze(1)
            
            return state_energy

#end
