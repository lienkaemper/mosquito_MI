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
import seaborn as sns

def sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    sources_signal = np.random.rand(n_source_signal, n_odorants) < p_source
    sources_signal = np.random.rand(n_source_signal, n_odorants) * sources_signal
    print(sources_signal)

    sources_noise = np.random.rand(n_source_noise, n_odorants) < p_source
    sources_noise = np.random.rand(n_source_noise, n_odorants) * sources_noise
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

with open("../data/hallem_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 

S_scaled = S/(10**3)
S_scaled -= np.mean(S_scaled)
 

n_receptors = S.shape[0]
n_odorants = S.shape[1]

S = np.random.rand(n_receptors, n_odorants)
S *=  (np.random.rand(n_receptors, n_odorants) < .2)
S_scaled = S

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
        self.activation = nn.ReLU() #changed to leaky relu to see if things getting stuck below threshold is the problem 
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

    def __init__(self, n_orn, n_pn, n_out, verbose = False):
        self.n_orn = n_orn
        self.n_pn = n_pn
        self.n_out= n_out
        self.verbose = verbose
        super(FullModel, self).__init__()

        # ORN-PN
        self.layer1 = Layer(self.n_orn, self.n_pn,
                            weight_initializer='uniform',
                            weight_initial_value = 1/n_orn,
                            sign_constraint=True,
                            pre_norm=None,
                            post_norm = None,
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
n_train = 1000  # number of training examples 
n_val = 1000  # number of validation examples
 # number of classes
n_orn = n_receptors  # number of olfactory receptor neurons
n_pn = 10
neural_noise_strength = 0
concentration_noise = 0.

n_source_signal = 10
n_source_noise = 10
n_out = n_receptors

p_source = .25
p_sample = 1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16

n_neural = 5
n_environmental = 5
n_epochs = 800


neural_noise_strengths = np.linspace(0.001, 3, n_neural)
environmental_noise_strengths = np.linspace(0.001, 3, n_environmental)



trials = 5

sigma_list = []
eta_list = []
coexp_levels = []


for trial in range(trials):
    for eta in environmental_noise_strengths:
        sampler, sources_signal, sources_noise = sparse_latent(p_source = p_source, p_sample=p_sample, n_source_signal=n_source_signal, n_source_noise = n_source_noise, n_odorants=n_odorants, concentration_noise=.1, noise_strength = eta)


        train_signal, train_noise,  _ =  sampler(n_samples= n_train)
        train_x = ((train_signal + train_noise)@ S_scaled.T).astype(np.float32)
        train_y = ((train_signal)@ S_scaled.T).astype(np.float32)

        val_signal, val_noise,  _ =  sampler(n_samples= n_val)
        val_x = ((val_signal + val_noise)@ S_scaled.T).astype(np.float32)
        val_y = ((val_signal)@ S_scaled.T).astype(np.float32)

        train_y = torch.from_numpy(train_y)
        val_y = torch.from_numpy(val_y)

        plt.close()
        C_xx = np.corrcoef(train_signal.T)
        cs = plt.imshow(C_xx)

        plt.colorbar(cs)
       # plt.show()
        plt.close()

        plt.hist(C_xx.flatten())
        #plt.show()
        plt.close()

        print("data generated")
        for sigma in neural_noise_strengths:
            print("trial  = {}, eta = {}, sigma = {}".format(trial, eta, sigma))

            train_neur_noise =sigma* torch.randn(n_train, n_pn)
            val_neur_noise = sigma*torch.randn(n_val, n_pn)

            train_xz = torch.cat((torch.from_numpy(train_x), train_neur_noise), dim =1)
            val_xz = torch.cat((torch.from_numpy(val_x), val_neur_noise), dim=1)

            model = FullModel(n_orn, n_pn, n_out, verbose = False)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.999, .999))


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

            coexp_levels.append(coexp_level(model.w_orn2pn.T))
            sigma_list.append(sigma)
            eta_list.append(eta)
            with open('../results/sparse_autoencoder/w_orn2_pn_trial={}n={:.2f}e={:.2f}'.format(trial,
                                                          sigma, eta), 'wb') as f:
                pkl.dump(model.w_orn2pn, f)
            with open('../results/sparse_autoencoder/learning_curve_trial={}n={:.2f}e={:.2f}'.format(trial,
                                                                sigma, eta), 'wb') as f:
                pkl.dump(learning_curve, f)
                    
             #plot learning curve   
            plt.close() 
            fig, ax = plt.subplots()    
            ax.plot(np.log(learning_curve))
            plt.savefig("../results/plots/sparse_autoencoder/learning_curve={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))

            #plot true vs. predicted values
            val_y_pred= model(val_xz, val_y)['output'].detach().numpy()
            train_y_pred = model(train_xz, train_y)['output'].detach().numpy()

            plt.close()
            fig, axs = plt.subplots(2)
            axs[0].scatter(val_y[:,0], val_y_pred[:,0], alpha = .1, color = 'black')
            axs[0].scatter(val_y[:,1], val_y_pred[:,1], alpha = .1)
            axs[0].scatter(val_y[:,2], val_y_pred[:,2], alpha = .1)

            # #axs[0].plot(val_y, val_y, color = "red")
            # axs[0].plot(train_y, train_y+.1, color = "red")
            # axs[0].plot(train_y, train_y-.1, color = "red")
            axs[0].set_title("validation")

            axs[1].scatter(train_y[:,0], train_y_pred[:,0], alpha = .1, color = 'black')
            axs[1].scatter(train_y[:,1], train_y_pred[:,1], alpha = .1)
            axs[1].scatter(train_y[:,2], train_y_pred[:,2], alpha = .1)
            axs[1].set_title("training")
            plt.xlabel("ground truth")
            plt.ylabel("predicted")

            plt.savefig("../results/plots/sparse_autoencoder/true_vs_pred_trial={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))


            #plot weights histograms for all layers
            plt.close()
            fig, axs = plt.subplots(2)
            axs[0].hist( model.w_orn2pn.flatten())
            axs[0].set_title("OR to ORN weights")
        
            axs[1].set_title("ORN to output weights")
            axs[1].hist(model.w_out.flatten())

            plt.savefig("../results/plots/sparse_autoencoder/weight_hists_trial={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))


            #compute and plot activations for all layers
            with torch.no_grad():
                x = val_xz
                act1 = model.layer1(x[:,0:n_receptors]) + x[:, n_receptors:]
                y = model.layer2(act1)

            plt.close()
            fig, axs = plt.subplots(2)
            PN_acts = act1.cpu().numpy()
            axs[0].hist(PN_acts.flatten())
            axs[0].set_title("ORN activations")

            out_acts = y.cpu().numpy().T
            axs[1].set_title("output activations")
            axs[1].hist(out_acts.flatten())

            plt.savefig("../results/plots/sparse_autoencoder/activation_hists_trial={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))



            #plot final OR-ORN weights
            plt.close()
            fig, ax  = plt.subplots()
            cs = ax.imshow( model.w_orn2pn.T)
            fig.colorbar(cs)
            ax.set_xlabel("receptors")
            ax.set_ylabel("neurons")
            ax.set_title("ORN to PN weights")
            plt.savefig("../results/plots/sparse_autoencoder/expression_trial={}n={:.2f}e={:.2f}.png".format(trial,
                                                                sigma, eta))


df = pd.DataFrame({"coexp_level" : coexp_levels, "response_noise": sigma_list, "environmental_noise": eta_list })
plt.close()
plt.figure()
sns.lineplot(data = df, x = "environmental_noise", y = "coexp_level", hue="response_noise")
plt.savefig("../results/plots/sparse_autoencoder/coexp_levels.png")