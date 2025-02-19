import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
import pickle as pkl
import time
from src.quantify_coexpression import coexp_level
from tqdm import tqdm
import pandas as pd

def sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    sources_signal = np.random.rand(n_source_signal, n_odorants) < p_source
    sources_signal = np.random.rand(n_source_signal, n_odorants) * sources_signal

    sources_noise = np.random.rand(n_source_noise, n_odorants) < p_source
    sources_noise = np.random.rand(n_source_noise, n_odorants) * sources_noise
    def sample(n_samples = 1):
        X = np.random.rand(n_samples, n_odorants)
        Y = np.random.rand(n_samples, n_source_signal)
        for i in range(n_samples):
            noisy_sources_signal = sources_signal +  concentration_noise * np.random.randn(n_source_signal, n_odorants) * (sources_signal > 0)
            noisy_sources_signal = noisy_sources_signal * (noisy_sources_signal > 0)

            noisy_sources_noise = sources_noise +  concentration_noise * np.random.randn(n_source_noise, n_odorants) * (sources_noise > 0)
            noisy_sources_noise = noisy_sources_noise * (noisy_sources_noise > 0)

            ids_signal = np.random.rand(n_source_signal) < p_sample
            ids_noise = np.random.rand(n_source_noise) < p_sample
            coeffs_signal =  ids_signal * np.random.rand(n_source_signal)
            coeffs_noise = ids_noise * np.random.rand(n_source_noise)

            Y[i,:] = coeffs_signal 
            X[i,:] = coeffs_signal.T @ noisy_sources_signal + noise_strength * coeffs_noise.T @ noisy_sources_noise
        return X.astype(np.float32), Y.astype(np.float32)
    return sample, sources_signal, sources_noise

with open("../data/hallem_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 

S_scaled = S/(10**3)

n_receptors = S.shape[0]
n_odorants = S.shape[1]

# S = np.random.rand(n_receptors, n_odorants)
# S *=  (np.random.rand(n_receptors, n_odorants) < .2)
# S_scaled = S

def _get_normalization(norm_type, num_features=None):
    if norm_type is not None:
        if norm_type == 'batch_norm':
            return nn.BatchNorm1d(num_features)
    return lambda x: x


def L1_penalty(W):
    return torch.sum(torch.relu(torch.sum(torch.abs(W), dim=1) - 1))


class Layer(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Same as nn.Linear, except that weight matrix can be set non-negative
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 sign_constraint=False,
                 weight_initializer=None,
                 weight_initial_value=None,
                 bias_initial_value=0,
                 pre_norm=None,
                 post_norm=None,
                 dropout=False,
                 dropout_rate=None,
                 verbose = False
                 ):
        super(Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_initializer = weight_initializer
        if weight_initial_value:
            self.weight_init_range = weight_initial_value
        else:
            self.weight_init_range = 2. / in_features
        self.bias_initial_value = bias_initial_value
        self.sign_constraint = sign_constraint
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.pre_norm = _get_normalization(pre_norm, num_features=out_features)
        self.activation = nn.ReLU() 
        self.post_norm = _get_normalization(
            post_norm, num_features=out_features)

        if dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = lambda x: x

        self.verbose = verbose
        self.reset_parameters()



    def reset_parameters(self):
        if self.sign_constraint:
            self._reset_sign_constraint_parameters()
        else:
            self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

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
        weight = self.effective_weight
        pre_act = F.linear(input, weight, self.bias)
        pre_act_normalized = self.pre_norm(pre_act)
        output = self.activation(pre_act_normalized)
        output_normalized = self.post_norm(output)
        output_normalized = self.dropout(output_normalized)
        return output_normalized


class FullModel(nn.Module):
    """"The full 3-layer model."""

    def __init__(self, n_orn, n_pn, n_out, noise_strength, verbose = False):
        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_out= n_out
        self.noise_strength = noise_strength
        self.verbose = verbose
        super(FullModel, self).__init__()

        # ORN-PN
        self.layer1 = Layer(self.n_orn, self.n_pn,
                            weight_initializer='uniform',
                            weight_initial_value = 1/n_orn,
                            sign_constraint=True,
                            pre_norm=None,
                            bias = False
                            )

        # PN-KC


        self.layer2 = nn.Linear(self.n_pn, self.n_out)  # KC-output

        self.loss = torch.nn.MSELoss() 

         
    def forward(self, x, target):
        act1 = self.layer1(x[:,0:n_receptors]) + x[:, n_receptors:]
        y = self.layer2(act1)
        loss = self.loss(y, target) + .05*L1_penalty(self.layer1.weight)
        # with torch.no_grad():
        #     _, pred = torch.max(y, 1)
        #     acc = (pred == target).sum().item() / target.size(0)
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
n_train = 2000  # number of training examples 
n_val = 1000  # number of validation examples
 # number of classes
n_orn = n_receptors  # number of olfactory receptor neurons
n_pn = 10
neural_noise_strength = 0
concentration_noise = 0.05

n_source_signal = 1
n_source_noise = 10
n_out = n_source_signal

p_source = .1
p_sample = 1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16

n_neural = 3
n_environmental = 3
n_epochs = 100


neural_noise_strengths = np.linspace(0, .1, n_neural)
environmental_noise_strengths = np.linspace(0, .1, n_environmental)



trials = 4

sigma_list = []
eta_list = []
coexp_levels = []


for trial in range(trials):
    fig_lc, axs_lc = plt.subplots(n_neural,n_environmental)
    fig_tot, axs_tot = plt.subplots(n_neural, n_environmental)
    for i, eta in enumerate(environmental_noise_strengths):
        sampler, sources_signal, sources_noise = sparse_latent(p_source = p_source, p_sample=p_sample, n_source_signal=n_source_signal, n_source_noise = n_source_noise, n_odorants=n_odorants, concentration_noise=concentration_noise, noise_strength = eta)


        train_x, train_y =  sampler(n_samples= n_train)
        train_x = (train_x @ S_scaled.T).astype(np.float32)

        val_x, val_y =  sampler(n_samples= n_val)
        val_x = (val_x @ S_scaled.T).astype(np.float32)

        train_y = torch.from_numpy(train_y)
        val_y = torch.from_numpy(val_y)

        print("data generated")
        for j, sigma in enumerate(neural_noise_strengths):
            print("trial  = {}, eta = {}, sigma = {}".format(trial, eta, sigma))

            train_neur_noise =sigma* torch.randn(n_train, n_pn)
            val_neur_noise = sigma*torch.randn(n_val, n_pn)

            train_xz = torch.cat((torch.from_numpy(train_x), train_neur_noise), dim =1)
            val_xz = torch.cat((torch.from_numpy(val_x), val_neur_noise), dim=1)

            fig, axs = plt.subplots(2, 1)
            for k, beta1 in enumerate(np.array([.9, .99, .999])):
                model = FullModel(n_orn, n_pn, n_out, noise_strength = sigma, verbose = False)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), betas=(beta1, .999))

                loss_train = 0
                total_time, start_time = 0, time.time()
                learning_curve = np.zeros(n_epochs)
                for epoch in tqdm(range(n_epochs)):
                    with torch.no_grad():
                        model.eval()
                        res_val = model(val_xz, val_y)
                    loss_val = res_val['loss'].item()
                    learning_curve[epoch] = loss_val


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
                axs[0].plot(np.log(learning_curve), label = beta1)
                axs_lc[i,j].plot(np.log(learning_curve), label = beta1)
                axs_lc[i,j].legend()
                axs_lc[i,j].set_title("eta = {:.2f}, sigma = {:.2f}".format(eta, sigma))

                sums = np.sum(model.w_orn2pn, axis = 0)
            
            
                axs_tot[i,j].bar(range(k,3*n_pn+k, 3), sums)
                axs_tot[i,j].hlines(1, 0, n_pn*3)
                axs_tot[i,j].set_title("eta = {:.2f}, sigma = {:.2f}".format(eta, sigma))




                #plot true vs. predicted values
                val_y_pred= model(val_xz, val_y)['output'].detach().numpy()
                train_y_pred = model(train_xz, train_y)['output'].detach().numpy()

                axs[1].scatter(val_y, val_y_pred, alpha = .1,  label = beta1)
                #axs[0].plot(val_y, val_y, color = "red")
                axs[1].plot(train_y, train_y+.1, color = "red")
                axs[1].plot(train_y, train_y-.1, color = "red")
                axs[1].set_title("validation")
                plt.xlabel("ground truth")
                plt.ylabel("predicted")

            axs[0].legend()
            axs[1].legend()
            fig.savefig("../results/plots/sparse/param_tuning={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))
    fig_lc.savefig("../results/plots/sparse/all_learning_curves_trial={}".format(trial))
    fig_tot.savefig("../results/plots/sparse/neuron_totals_trial={}.png".format(trial))


