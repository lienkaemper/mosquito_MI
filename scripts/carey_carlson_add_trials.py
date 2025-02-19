## Standard libraries

import numpy as np
import pandas  as pd 
import torch
import argparse
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm


from src.optimize_mi import  maximize_mi_new
from src.gen_cov_mats import rand_dot_mat, sparse_latent_cov
from src.clustering import order_cor_matrix
from src.gen_cov_mats import cov_to_cor



parser = argparse.ArgumentParser(description='Find exprssion matrixes which maximize mutual information with Hallem-Carlson used for sensing matrix')
parser.add_argument('--plot', dest = 'plot', type=bool, nargs=1, default=False,
                    help='whether to plot MI curves, final cov mat')
args = parser.parse_args()
plot = args.plot

with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 

n_n = 25
trials = 30
trial_offset = 20
opt_steps = 1000
var_e_min = 0
var_e_max = 4
#var_n_min = .005
#var_n_max = .01
n_grid_e = 5
n_grid_n = 3

var_es = np.linspace(var_e_min, var_e_max, n_grid_e)
var_es[0] += 0.1
#var_ns = np.linspace(var_n_min, var_n_max, n_grid_n)
var_ns = np.array([ 0.005, 0.01, 0.02])


print(var_es)


n_r, n_o = S.shape

S_scaled = S/(10**3)
noise_cov = S_scaled @ S_scaled.T




noise_cov_t = torch.from_numpy(noise_cov.astype(np.float32))
final_cov_mats = np.zeros((trials+trial_offset, n_grid_e, n_grid_n, n_n, n_r))
MI_curves = np.zeros((trials+trial_offset, n_grid_e, n_grid_n, opt_steps))

with open("../results/tables/opt_coexp_cc/coexp_mats_aug.pkl", 'rb') as f:
    old_cov_mats  = pkl.load(f)

with open("../results/tables/opt_coexp_cc/MI_curves_aug.pkl", 'rb') as f:
    old_MI_curves = pkl.load(f)

for trial in range(trial_offset):
    final_cov_mats[trial, :, :, :, :] = old_cov_mats[trial, :, :, :, :]
    MI_curves[trial, :, :, :] = old_MI_curves[trial, :, :]

with tqdm(total = n_grid_e * n_grid_n * trials) as pbar:
    for trial in range(trial_offset, trials+trial_offset):
        p_source = .15
        p_sample = 1
        n_source_signal = 25
        n_source_noise = 1000
        Cov_xx= sparse_latent_cov(p_source = p_source, p_sample=p_sample, n_source_signal=n_source_signal, n_source_noise = n_source_noise, n_odorants=n_o, concentration_noise=0.5, noise_strength = 0)

        #print(Cov_xx)
        with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'wb') as file:
            pkl.dump(Cov_xx, file)
        
        C, ll, z = order_cor_matrix(cov_to_cor(Cov_xx))

        norm =  TwoSlopeNorm(vmin=np.min(Cov_xx), vcenter=0, vmax=np.max(Cov_xx))
        cs =plt.imshow(Cov_xx[np.ix_(ll, ll)], cmap = 'bwr', norm = norm)
        plt.colorbar(cs)

        if plot:
            plt.show()
        receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
        receptor_cov_t = torch.from_numpy(receptor_cov.astype(np.float32))
        for i, var_e in enumerate(var_es):
            for j, var_n in enumerate(var_ns):
                if max(var_e, var_n) > 0:
                    x, gradients, values ,xs =maximize_mi_new(receptor_cov_t, noise_cov_t, env_noise = var_e, neur_noise = var_n, m = n_n, steps = opt_steps, rate = .05, grad_noise = .05)
                    A = x.detach().numpy()
                    A = A[np.argmax(A, axis = 1).argsort(), :]
                    final_cov_mats[trial, i,j, :, :] = A
                    MI_curves[trial, i, j, :] = values
                    if plot:
                        fig, axs = plt.subplots(1,2)
                        axs[0].plot(values)
                        axs[1].imshow(A)
                        plt.show()
                pbar.update(1)



with open("../results/tables/opt_coexp_cc/var_es.pkl", 'wb') as f:
    pkl.dump(var_es, f)

with open("../results/tables/opt_coexp_cc/var_ns.pkl", 'wb') as f:
    pkl.dump(var_ns, f)


with open("../results/tables/opt_coexp_cc/coexp_mats_aug.pkl", 'wb') as f:
    pkl.dump(final_cov_mats, f)

with open("../results/tables/opt_coexp_cc/MI_curves_aug.pkl", 'wb') as f:
    pkl.dump(MI_curves, f)
