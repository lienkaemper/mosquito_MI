## Standard libraries

import numpy as np
import pandas  as pd 
import torch
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
from tqdm.notebook import tqdm
import sys
import pickle as pkl
from scipy.stats import spearmanr
from scipy.linalg import eigh
from scipy.linalg import subspace_angles
import matplotlib.pyplot as plt
from scipy import stats


from tqdm import tqdm
from src.quantify_coexpression import coexp_level, coexd_receptors
from src.clustering import order_cor_matrix
from src.gen_cov_mats import cov_to_cor



with open("../data/hallem_carlson_sensing.pkl", "rb") as f:
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

n_pn = 12



# receptor_cov_t = torch.from_numpy(receptor_cov.astype(np.float32))
# noise_cov_t = torch.from_numpy(noise_cov.astype(np.float32))
coexp_idx = []
coexd_res = []
non_coexd_res = []
pca_alignments = []
resp_gains = []
var_e_list = []
var_n_list = []
with tqdm(total = n_grid_e * n_grid_n * trials) as pbar:
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
                # plt.plot(values)
                # plt.show()
                # plt.figure()
                # plt.imshow(A)
                # plt.title("var_e = {}, var_n = {}".format(var_e, var_n))
                # plt.show()
                coexp_idx.append(coexp_level(A))
                A_center = A - np.mean(A, axis = 0)
                AS_center =  A @ S_scaled - np.mean(A @ S_scaled, axis = 0)

                C, N = coexd_receptors(A, thresh = .1)

                coexd_res.append(np.mean(residual[C > 0]))
                non_coexd_res.append(np.mean(residual[C  == 0]))


                exp_cov = AS_center.T @ AS_center
                exp_cor = cov_to_cor(exp_cov)
                
                #thetas = subspace_angles(V, (A@S).T )
               # similarity = np.mean(np.cos(thetas))
                resp_gain = np.trace(A@S_scaled@S_scaled.T@ A.T)
                #pca_alignments.append(similarity)
                pca_alignments.append(0)
                resp_gains.append(resp_gain)

                plt.scatter(noise_cor.flatten(), receptor_cor.flatten(), color = 'blue', s = 10)
                plt.plot(noise_cor.flatten(), res.intercept + res.slope*noise_cor.flatten(),  label='fitted line', color = 'black')
                # plt.set_xlabel("Noise correlation")
                # plt.set_ylabel("Signal correlation")
                coexd_x = noise_cor[C > 0]
                not_coexd_x = noise_cor[C  == 0]

                coexd_y = receptor_cor[C > 0]
                not_coexd_y = receptor_cor[C  == 0]

                plt.scatter(coexd_x, coexd_y, color = 'red', s = 10)
                plt.show()

                pbar.update(1)


            

df = pd.DataFrame({"env_noise" : var_e_list, "neur_noise" : var_n_list, "coexp_idx" : coexp_idx,  "pca_alignment": pca_alignments, "resp_gain" : resp_gains, "coexd_res" : coexd_res, "non_coexd_res": non_coexd_res})


with open("../results/tables/sparse/df.pkl", 'wb') as f:
    pkl.dump(df, f)

