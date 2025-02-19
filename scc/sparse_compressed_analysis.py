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
from scipy import stats





def coexp_level(A):
    denoms = np.sum(A**2, axis =1)
    nums =  np.sum(A, axis =1)**2
    return np.mean(nums/denoms)-1
    
def coexd_receptors(A, thresh = .001):
    n, r = A.shape
    C = np.zeros((r,r))
    N = np.zeros((r,r))
    for r1 in range(r):
        for r2 in range(r1+1, r):
            found = False
            for neur in range(n):
                if min(A[neur, r1], A[neur, r2]) > thresh:
                    C[r1, r2] = 1
                    found = True
            if not found:
                N[r1, r2] = 1
    return C, N

       
def cov_to_cor(mat):
    d = np.copy(np.diag(mat))
    d[d== 0] = 1
    return (1/np.sqrt(d)) *mat * (1/(np.sqrt(d)))[...,None]



with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f) 



with open("../results/tables/sparse/coexp_mats.pkl", 'rb') as f:
    final_coexp_mats = pkl.load(f)

with open("../results/tables/sparse/var_es.pkl", 'rb') as f:
    var_es = pkl.load(f)


with open("../results/tables/sparse/var_ns.pkl", 'rb') as f:
    var_ns = pkl.load(f)

n_grid_e = 5
n_grid_n = 3
trials = 3
n_r, n_o = S.shape
n_pn = 25






coexp_idx = []
coexd_res = []
non_coexd_res = []
pca_alignments = []
resp_gains = []
var_e_list = []
var_n_list = []
for trial in range(trials):
    with open(f'../results/tables/sparse/stim_covariance_trial={trial}.pkl', 'rb') as file:
        Cov_xx = pkl.load( file)
    W, V = eigh(Cov_xx, subset_by_index=(n_o-n_pn, n_o-1))

    S_scaled = S/(10**3)
    receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
    noise_cov = S_scaled @ S_scaled.T
    stim_cor = cov_to_cor(Cov_xx)

    receptor_cor = cov_to_cor(receptor_cov)
    noise_cor = cov_to_cor(noise_cov)    
    res = stats.linregress(noise_cor.flatten(), receptor_cor.flatten())

    b0 = res.intercept
    b1 = res.slope

    residual = receptor_cor - (b0 + b1*(noise_cor))
    for i, var_e in enumerate(var_es):
        for j, var_n in enumerate(var_ns):
            var_e_list.append(var_e)
            var_n_list.append(var_n)
            A = final_coexp_mats[trial, i,j, :, :]
            coexp_idx.append(coexp_level(A))
            A_center = A - np.mean(A, axis = 0)
            AS_center =  A @ S_scaled - np.mean(A @ S_scaled, axis = 0)

            C, N = coexd_receptors(A, thresh = .1)

            coexd_res.append(np.mean(residual[C > 0]))
            non_coexd_res.append(np.mean(residual[C  == 0]))


            exp_cov = AS_center.T @ AS_center
            exp_cor = cov_to_cor(exp_cov)
            
            thetas = subspace_angles(V, (A@S).T )
            similarity = np.mean(np.cos(thetas))
            resp_gain = np.trace(A@S_scaled@S_scaled.T@ A.T)
            pca_alignments.append(similarity)
            resp_gains.append(resp_gain)

            coexd_x = noise_cor[C > 0]
            not_coexd_x = noise_cor[C  == 0]

            coexd_y = receptor_cor[C > 0]
            not_coexd_y = receptor_cor[C  == 0]



df = pd.DataFrame({"env_noise" : var_e_list, "neur_noise" : var_n_list, "coexp_idx" : coexp_idx,  "pca_alignment": pca_alignments, "resp_gain" : resp_gains, "coexd_res" : coexd_res, "non_coexd_res": non_coexd_res})


with open("../results/tables/sparse/df.pkl", 'wb') as f:
    pkl.dump(df, f)
