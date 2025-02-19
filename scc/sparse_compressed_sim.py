with open("output.txt", "a") as f:
    f.write("top of file ")
# Imports
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math
import pickle as pkl
import time
from tqdm import tqdm
import pandas as pd
import gc

with open("output.txt", "a") as f:
    f.write("past imports ")

# Defines distribution 
def sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    sources_signal = np.random.rand(n_source_signal, n_odorants) < p_source
    sources_signal = 10*np.random.rand(n_source_signal, n_odorants) * sources_signal

    sources_noise = np.random.rand(n_source_noise, n_odorants) < p_source
    sources_noise = (n_source_signal/n_source_noise)*10*np.random.rand(n_source_noise, n_odorants) * sources_noise
    def sample(n_samples = 1):
        X_signal = np.random.rand(n_samples, n_odorants)
        X_noise = np.random.rand(n_samples, n_odorants)
        Y = np.random.rand(n_samples, n_source_signal)
        for i in range(n_samples):
            noisy_sources_signal = sources_signal +  concentration_noise * np.random.randn(n_source_signal, n_odorants) * (sources_signal > 0) #add concentration noise to signal and noise
            noisy_sources_signal = noisy_sources_signal * (noisy_sources_signal > 0)

            noisy_sources_noise = sources_noise +  concentration_noise * np.random.randn(n_source_noise, n_odorants) * (sources_noise > 0)
            noisy_sources_noise = noisy_sources_noise * (noisy_sources_noise > 0)


            ids_signal = np.random.rand(n_source_signal) < p_sample
            ids_noise = np.random.rand(n_source_noise) < p_sample
            coeffs_signal =  ids_signal * np.random.rand(n_source_signal)
            coeffs_noise = ids_noise * np.random.rand(n_source_noise)

            Y[i,:] = coeffs_signal 
            X_signal[i,:] = coeffs_signal.T @ noisy_sources_signal
            X_noise[i, :] =  noise_strength * coeffs_noise.T @ noisy_sources_noise
        return X_signal.astype(np.float32), X_noise.astype(np.float32), Y.astype(np.float32)
    return sample, sources_signal, sources_noise

# import sensing matrix 
with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 

with open("output.txt", "a") as f:
    f.write("opened sensing ")

S = S/(10**3)
S -= np.mean(S)
 

n_receptors = S.shape[0]
n_odorants = S.shape[1]


# Network definition 

def L1_penalty(W):
    return torch.sum(torch.relu(torch.sum(torch.abs(W), dim=1) - 1))


class Layer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix can be set non-negative
    """

    def __init__(self,
                 in_features,
                 #noise_features,
                 out_features,
                 bias= True,
                 gain = True,
                 sign_constraint=False,
                 weight_initializer=None,
                 weight_initial_value=None,
                 bias_initial_value=0,
                 gain_initial_value = 1,
                 verbose = False
                 ):

        super(Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if gain:
            self.gain = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('gain', None)

        self.weight_initializer = weight_initializer
        if weight_initial_value:
            self.weight_init_range = weight_initial_value
        else:
            self.weight_init_range = 2. / in_features

        self.bias_initial_value = bias_initial_value
        self.gain_initial_value = gain_initial_value

        self.sign_constraint = sign_constraint
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.activation = nn.Sigmoid() 



        self.verbose = verbose
        self.reset_parameters()

    def reset_parameters(self):
        if self.sign_constraint:
            self._reset_sign_constraint_parameters()
        else:
            self._reset_parameters()

    def _reset_parameters(self): #can probably delete this, yeah? 
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.constant_(self.bias, self.bias_initial_value)
        if self.gain is not None:
            init.constant_(self.gain, self.gain_initial_value)
 
    def _reset_sign_constraint_parameters(self):
        if self.weight_initializer == 'constant':
            init.constant_(self.weight, self.weight_init_range)
        elif self.weight_initializer == 'uniform':
            init.uniform_(self.weight, 0, self.weight_init_range)
        elif self.weight_initializer == 'normal':
            init.normal_(self.weight, 0, self.weight_init_range)
        else:
            raise ValueError('Unknown initializer',
                             str(self.weight_initializer))

        if self.bias is not None:
            init.constant_(self.bias, self.bias_initial_value)
        if self.gain is not None:
            init.constant_(self.gain, self.gain_initial_value)

    @property
    def effective_weight(self):
        if self.sign_constraint:
            weight = torch.abs(self.weight)
        else:
            weight = self.weight

        return weight

    def forward(self, input):
        if self.verbose:
            print("input to layer, ", input)
            print("gain", self.gain)
        weight = self.effective_weight
        signal_features = input[:, 0:self.in_features]
        if input.shape[1] > self.in_features:
            noise_features = input[:, self.in_features:]
            output = self.activation(self.gain*(F.linear(signal_features, weight, self.bias) + noise_features))
        else:
            output = self.activation(self.gain*(F.linear(signal_features, weight, self.bias)))
        return output


class FullModel(nn.Module):
    """"The full 3-layer model."""

    def __init__(self, n_receptor, n_pn, n_kc, n_out, verbose = False):
        self.n_receptor = n_receptor 
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_out= n_out
        self.verbose = verbose
        super(FullModel, self).__init__()

        # receptor - pn
        self.layer1 = Layer(self.n_receptor, self.n_pn,
                            weight_initializer='uniform',
                            weight_initial_value = 1/n_receptor,
                            sign_constraint=True,
                            bias = True,
                            gain = True
                            )

        #lateral inhibition
        self.layer2 = nn.Linear(self.n_pn, self.n_pn)  

        #PN-KC
        self.layer3 = Layer(self.n_pn, self.n_kc, 
                            weight_initializer='uniform',
                            weight_initial_value = 1/n_receptor,
                            sign_constraint=False,
                            bias = True,
                            gain = True)

        #KC-out
        self.layer4 = nn.Linear(self.n_kc, self.n_out) 
        

        self.loss = torch.nn.MSELoss() 

         
    def forward(self, x, target):
        #act1 = self.layer1(x[:,0:self.n_receptor]) + x[:, self.n_receptor:]
        act1 = self.layer1(x)
        #act2 = self.layer2(act1)
        act3 = self.layer3(act1)
        y = self.layer4(act3)
        loss = self.loss(y, target) + .05*L1_penalty(self.layer1.weight)
        return {'loss': loss,  'output': y}

    @property
    def w_orn2pn(self):
        # Transpose to be consistent with tensorflow default
        return self.layer1.effective_weight.data.cpu().numpy().T

    @property
    def w_out(self):
        return self.layer2.weight.data.cpu().numpy().T

n_receptors = S.shape[0]
n_odorants = S.shape[1]
n_train = 40000  # number of training examples 
n_val = 100  # number of validation examples
n_orn = n_receptors  # number of olfactory receptor neurons
n_pn = 25
n_kc = 800 
n_out = n_odorants


p_source = .04
p_sample = .01
n_source_signal = 50
n_source_noise = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256


n_grid_n = 3
n_grid_e = 5
n_epochs = 800
trials = 3


#neural_noise_strengths = np.linspace(0.05, .15, n_grid_n)
neural_noise_strengths = np.array([0.025, 0.05, 0.1])
environmental_noise_strengths = np.linspace(0.00, 2, n_grid_e)





final_coexp_mats = np.zeros((trials, n_grid_e, n_grid_n, n_pn, n_receptors))
learning_curves_val = np.zeros((trials, n_grid_e, n_grid_n, n_epochs))
learning_curves_train = np.zeros((trials, n_grid_e, n_grid_n, n_epochs))


with open("output.txt", "a") as f:
    f.write("start of loop ")
for trial in range(trials):
    sampler, sources_signal, sources_noise = sparse_latent(p_source = p_source, p_sample=p_sample, n_source_signal=n_source_signal, n_source_noise = n_source_noise, n_odorants=n_odorants, concentration_noise=.1, noise_strength = 1)
    train_signal, train_noise,  _ =  sampler(n_samples= n_train)
    Cov_xx = np.cov(train_signal.T)
    Cov_xx = np.nan_to_num(Cov_xx)
    with open("output.txt", "a") as f:
        f.write(f"created dataset {trial}")

    with open(f'../results/tables/sparse/stim_covariance_trial={trial}.pkl', 'wb') as file:
        pkl.dump(Cov_xx, file)
    with open(f'../results/tables/sparse/stim_sources_trial={trial}.pkl', 'wb') as file:
        pkl.dump(sources_signal, file)
    for i, eta in enumerate(environmental_noise_strengths):
        train_x = ((train_signal + eta*train_noise)@ S.T).astype(np.float32)
        train_y = train_signal.astype(np.float32)

        val_signal, val_noise,  _ =  sampler(n_samples= n_val)
        val_x = ((val_signal + eta*val_noise)@ S.T).astype(np.float32)
        val_y = val_signal.astype(np.float32)

        train_y = torch.from_numpy(train_y)
        val_y = torch.from_numpy(val_y)

        for j, sigma in enumerate(neural_noise_strengths):
            with open("output.txt", "a") as f:
                f.write("trial  = {}, eta = {}, sigma = {}".format(trial, eta, sigma))

            train_neur_noise =sigma*torch.randn(n_train, n_pn)
            val_neur_noise = sigma*torch.randn(n_val, n_pn)

            train_xz = torch.cat((torch.from_numpy(train_x), train_neur_noise), dim =1)
            val_xz = torch.cat((torch.from_numpy(val_x), val_neur_noise), dim=1)

            model = FullModel(n_orn, n_pn, n_kc, n_out, verbose = False)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.999, .999))


            loss_train = 0
            total_time, start_time = 0, time.time()

            learning_curve = np.zeros(n_epochs)
            learning_curve_train = np.zeros(n_epochs)
            for epoch in tqdm(range(n_epochs)):
                with torch.no_grad():
                    model.eval()
                    res_val = model(val_xz, val_y)
                    res_val_train = model(train_xz, train_y)
                loss_val = res_val['loss'].item()
                loss_val_train = res_val_train['loss'].item()
                learning_curve[epoch] = loss_val
                learning_curve_train[epoch] = loss_val_train



                start_time = time.time()

                model.train()
                random_idx = np.random.permutation(n_train)
                idx = 0
                while idx < n_train:
                    batch_indices = random_idx[idx:idx+batch_size]
                    idx += batch_size

                    res = model(train_xz[batch_indices],
                                train_y[batch_indices])
                    optimizer.zero_grad()
                    res['loss'].backward()
                    optimizer.step()

                loss_train = res['loss'].item()

            A = model.w_orn2pn.T
            A = A[np.argmax(A, axis = 1).argsort(), :]
            final_coexp_mats[trial, i,j, :, :] = A
            learning_curves_train[trial, i, j, :] = learning_curve_train
            learning_curves_val[trial, i, j, :] = learning_curve



    with open("../results/tables/sparse/var_es.pkl", 'wb') as f:
        pkl.dump(environmental_noise_strengths, f)

    with open("../results/tables/sparse/var_ns.pkl", 'wb') as f:
        pkl.dump(neural_noise_strengths, f)


    with open("../results/tables/sparse/coexp_mats.pkl", 'wb') as f:
        pkl.dump(final_coexp_mats, f)

    with open("../results/tables/sparse/learning_curves_train.pkl", 'wb') as f:
        pkl.dump(learning_curves_train, f)


    with open("../results/tables/sparse/learning_curves_val.pkl", 'wb') as f:
        pkl.dump(learning_curves_val, f)

    with open("output.txt", "a") as f:
        f.write("saved data ")