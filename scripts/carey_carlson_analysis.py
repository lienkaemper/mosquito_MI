## Standard libraries

import numpy as np
import pandas  as pd 
import torch
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
from tqdm.notebook import tqdm
import pickle as pkl
from scipy.linalg import eigh
from scipy.linalg import subspace_angles
import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm
from src.gen_cov_mats import cov_to_cor
from src.quantify_coexpression import coexp_level, coexd_receptors, bootstrap_confidence_interval
from src.compute_mi import coexp_ratio

plt.style.use('paper_style.mplstyle')



with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 

with open('../results/tables/opt_coexp_cc/stim_covariance.pkl', 'rb') as file:
    Cov_xx = pkl.load(file)

with open("../results/tables/opt_coexp_cc/coexp_mats_aug.pkl", 'rb') as f:
    final_cov_mats = pkl.load(f)

with open("../results/tables/opt_coexp_cc/var_es.pkl", 'rb') as f:
    var_es = pkl.load(f)


with open("../results/tables/opt_coexp_cc/var_ns.pkl", 'rb') as f:
    var_ns = pkl.load(f)

n_grid_e = 5
n_grid_n = 3
trials = 50
m = 25
n_r, n_o = S.shape



coexp_idx = []

pca_alignments = []
resp_gains = []
var_e_list = []
var_n_list = []
fig, axs = plt.subplots(n_grid_e, n_grid_n)
with tqdm(total = n_grid_e * n_grid_n * trials) as pbar:
    for i, var_e in enumerate(var_es):
        for j, var_n in enumerate(var_ns):
            coexd_all_trials = []
            not_coexd_all_trials = []
            for trial in range(trials):
                var_e_list.append(var_e)
                var_n_list.append(var_n)

                with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
                    Cov_xx = pkl.load( file)


                S_scaled = S/(10**3)
                receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
                noise_cov = S_scaled @ S_scaled.T
                stim_cor = cov_to_cor(Cov_xx)
                receptor_cor = cov_to_cor(receptor_cov)
                noise_cor = cov_to_cor(noise_cov)    
                
       


                A = final_cov_mats[trial, i,j, :, :]

                coexp_idx.append(coexp_level(A))

                
                if np.any(np.isnan(A)):
                    resp_gains.append(np.nan)
                    pca_alignments.append(np.nan)
                else:
                    _, V = eigh(Cov_xx, subset_by_index=(n_o-m, n_o-1))
                    thetas = subspace_angles(V, (A@S).T )
                    similarity = np.mean(np.cos(thetas)) 
                    resp_gain = (1/m)*np.sqrt(np.trace(A@S_scaled@S_scaled.T@ A.T))
                    pca_alignments.append(similarity)
                    resp_gains.append(resp_gain)
                pbar.update(1)


            
df = pd.DataFrame({"env_noise" : var_e_list, "neur_noise" : var_n_list, "coexp_idx" : coexp_idx,  "pca_alignment": pca_alignments, "resp_gain" : resp_gains})


with open("../results/tables/opt_coexp_cc/df.pkl", 'wb') as f:
    pkl.dump(df, f)

with open("../results/tables/opt_coexp_cc/coexp_mats.pkl", 'wb') as f:
    pkl.dump(final_cov_mats, f)


diff = []
ci_top = []
ci_bottom = []
var_e_list = []
var_n_list = []
fig, axs = plt.subplots(n_grid_e, n_grid_n)
with tqdm(total = n_grid_e * n_grid_n * trials) as pbar:

    for i, var_e in enumerate(var_es):
        for j, var_n in enumerate(var_ns):
            coexd_all_trials = []
            not_coexd_all_trials = []

            for trial in range(trials):
                with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
                    Cov_xx = pkl.load( file)
                
                S_scaled = S/(10**3)
                A = final_cov_mats[trial, i,j, :, :]

                receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
                noise_cov = S_scaled @ S_scaled.T
                
                coexp_ratios = np.array([coexp_ratio(receptor_cov, noise_cov, i, j) for i in range(n_r) for j in range(n_r)]).reshape(n_r,n_r)
       
                C, N = coexd_receptors(A, thresh = .1)

                coexd = coexp_ratios[C > 0]/(var_n/var_e)
                not_coexd = coexp_ratios[N > 0]/(var_n/var_e)
                coexd[np.isinf(coexd)] = np.nan
                not_coexd[np.isinf(not_coexd)] = np.nan


                coexd_all_trials.extend(coexd)
                not_coexd_all_trials.extend(not_coexd)
                pbar.update(1)

            coexd_all_trials = np.log(np.array(coexd_all_trials))
            coexd_all_trials = coexd_all_trials[np.isfinite(coexd_all_trials)]

            not_coexd_all_trials = np.log(np.array(not_coexd_all_trials))
            not_coexd_all_trials = not_coexd_all_trials[np.isfinite(not_coexd_all_trials)]
            observed_diff, (lower_bound, upper_bound) = bootstrap_confidence_interval(coexd_all_trials, not_coexd_all_trials, n_bootstrap=5000)
            diff.append(observed_diff)
            ci_top.append(upper_bound)
            ci_bottom.append(lower_bound)
            var_e_list.append(var_e)
            var_n_list.append(var_n)
    
            axs[i,j].hist(coexd_all_trials, color = 'red', alpha  =.5, density=True, bins = np.linspace(-8, 8, 20))
            axs[i,j].hist(not_coexd_all_trials, color = 'blue', alpha  =.5, density=True, bins = np.linspace(-8, 8, 20))
            axs[i,j].set_title(f"v_e = {var_e}, v_n = {var_n}", fontsize = 8)


df = pd.DataFrame({"env_noise" : var_e_list, "neur_noise" : var_n_list, "diff" : diff,  "ci_top": ci_top, "ci_bottom" : ci_bottom})
print(df)
plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/coexp_pressure_hists.pdf")
plt.show()


with open("../results/tables/coexp_pressure_diffs.pkl", 'wb') as f:
    pkl.dump(df, f)
