## Standard libraries
import sys
import os
import math
import numpy as np
import time
import pandas  as pd 
import torch
import seaborn as sns
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
from tqdm.notebook import tqdm
import sys
from src.gen_cov_mats import cov_to_cor
from src.clustering import order_cor_matrix
from scipy import stats
from src.quantify_coexpression import coexd_receptors
from src.optimize_mi import mi_loss_new
from src.compute_mi import coexp_ratio


## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use('paper_style.mplstyle')


import pickle as pkl



with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={0}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)


with open("../results/tables/opt_coexp_cc/coexp_mats_aug.pkl", 'rb') as f:
    final_coexp_mats = pkl.load(f)

with open("../results/tables/opt_coexp_cc/MI_curves_aug.pkl", 'rb') as f:
    MI_curves = pkl.load(f)

with open('../results/tables/opt_coexp_cc/df.pkl', 'rb') as file:
    df = pkl.load(file)

with open("../results/tables/opt_coexp_cc/var_es.pkl", 'rb') as f:
    var_es = pkl.load(f)

with open("../results/tables/opt_coexp_cc/var_ns.pkl", 'rb') as f:
    var_ns = pkl.load(f)

with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f)

S_scaled = S/(10**3)
noise_cov = S_scaled @ S_scaled.T
n_r, n_o = S.shape

df['neur_noise'] = df['neur_noise'].astype('float32') 

trials = 50
n_grid_e = 5
n_grid_n = 3

#plot learning curves 

fig, axs = plt.subplots(n_grid_n, n_grid_e, sharex=True, sharey=True)
for i in range(n_grid_e):
    for j in range(n_grid_n):
        for trial in range(trials):
            mi_curve = MI_curves[trial, i,j, :]
            axs[j,i].plot(mi_curve[mi_curve > 0], color = 'black')
            axs[j,i].set_title(f"v_e^2 = {var_es[i]:.2f},\nv_n^2={var_ns[j]:.2f} ")


fig.supxlabel("Steps")
fig.supylabel("Mutual information")
plt.suptitle("Learning curves")
plt.tight_layout()
sns.despine()

plt.savefig("../results/plots/learning_curves_MI_cc.pdf")
plt.show()



#plot singular values of stimulus covariance matrix 

fig, ax = plt.subplots(figsize = (1.5, 1.5))
for trial in range(trials):
    with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
                Cov_xx = pkl.load( file)


    s = np.linalg.svd(Cov_xx, compute_uv=False, hermitian=False)
    s  =np.cumsum(s)
    plt.plot(s)

sns.despine()
plt.savefig("../results/plots/carey_carlson_svdvals.pdf")
plt.show()



fig, axd = plt.subplot_mosaic([['a', 'b', 'c1', 'd'],
                            ['a', 'b', 'c2', 'd'],
                            ['e', 'f', 'g', 'h'],
                            ['e', 'f', 'g', 'h']],
                            figsize =(7, 3))

#plot representetative stimulus covariance matrix

trial = 0
with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)
C, ll, z = order_cor_matrix(cov_to_cor(Cov_xx))

norm =  TwoSlopeNorm(vmin=np.min(Cov_xx), vcenter=0, vmax=np.max(Cov_xx))
cs = axd['a'].imshow(Cov_xx[np.ix_(ll, ll)],cmap = 'binary')
fig.colorbar(cs, ax =axd['a'])
axd['a'].set_title("stimulus covariance")
print("stim covariance!")
print(np.unique(np.diag(Cov_xx)))


#plot two representative coexpression matrixes

i_low = 0
j_low = 0
coexp_mat = final_coexp_mats[trial, i_low,j_low, :, :]
print(coexp_mat)
cs = axd['c1'].pcolor(coexp_mat, vmin = 0, vmax = 1, cmap = 'binary')
axd['c1'].set_title("env_noise = {}, neur_noise = {}".format(var_es[i_low], var_ns[j_low]))
axd['c1'].set_xlabel("receptors")
axd['c1'].set_ylabel("neurons ")
axd['c1'].set_xticks([])
axd['c1'].set_yticks([])

fig.colorbar(cs, ax =axd['c1'])

i_high = 4
j_high = 0
coexp_mat = final_coexp_mats[trial, i_high, j_high, :, :]
cs = axd['c2'].pcolor(coexp_mat, vmin = 0, vmax = 1, cmap = 'binary')
fig.colorbar(cs, ax =axd['c2'])
axd['c2'].set_xticks([])
axd['c2'].set_yticks([])
axd['c2'].set_xlabel("receptors")
axd['c2'].set_ylabel("neurons ")
axd['c2'].set_title("env_noise = {}, neur_noise = {}".format(var_es[i_high], var_ns[j_high]))


#plot co-expression levels 
sns.lineplot(data=df, x="env_noise", y="coexp_idx", hue="neur_noise", ax = axd['d'] )
axd['d'].set_title("coexpression index ")


#plot signal and noise correlation for co-expressed, non-co-expressed receptors
trial = 0
i = 3
j = 1
with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)
S_scaled = S/(10**3)
receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
noise_cov = S @ S.T

receptor_cor = cov_to_cor(receptor_cov)
noise_cor = cov_to_cor(noise_cov)


A = final_coexp_mats[trial, i, j,:, :]


C, N = coexd_receptors(A, thresh = .1)


axd['f'].scatter(noise_cor.flatten(), receptor_cor.flatten(), color = 'blue', s = 10)
axd['f'].set_xlabel("Noise correlation")
axd['f'].set_ylabel("Signal correlation")

coexd_x = noise_cor[C > 0]
not_coexd_x = noise_cor[C  == 0]

coexd_y = receptor_cor[C > 0]
not_coexd_y = receptor_cor[C  == 0]

axd['f'].scatter(coexd_x, coexd_y, color = 'red', s = 10,  zorder = 2)

#plot histogram of coexpression pressure for co-expressed, non-co-expressed receptors over all trials, noise levels
all_coexd_res = []
all_res = []
for trial in range(trials):
    for i in range(1,n_grid_e):
        for j in range(n_grid_n):
            with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
                        Cov_xx = pkl.load( file)

            S_scaled = S/(10**3)
            receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
            noise_cov = S_scaled @ S_scaled.T


            A = final_coexp_mats[trial, i, j,:, :]


            C, N = coexd_receptors(A, thresh = .1)


            coexp_ratios = np.array([coexp_ratio(receptor_cov, noise_cov, i, j) for i in range(n_r) for j in range(n_r)]).reshape(n_r,n_r)
            coexd = coexp_ratios[C > 0]/(var_ns[j]/var_es[i])
            not_coexd = coexp_ratios[N > 0]/(var_ns[j]/var_es[i])
            coexd[np.isinf(coexd)] = np.nan
            not_coexd[np.isinf(not_coexd)] = np.nan

            all_coexd_res.extend(coexd)

            all_res.extend(not_coexd)

all_coexd_res = np.array(all_coexd_res)
all_res = np.array(all_res)
axd['g'].hist(np.log(all_coexd_res), color = 'red', alpha  =.5, density=True, bins = np.linspace(-8, 8, 20))
axd['g'].hist(np.log(all_res), color = 'blue', alpha  =.5, density=True, bins = np.linspace(-8, 8, 20))



axd['g'].vlines(x = [np.mean(np.log(all_coexd_res[all_coexd_res > 0]))], ymin = 0, ymax = .25, color = 'red',)

print("MEAN RESIDUAL OF CO-EXPRESSED RECEPTORS", np.mean(np.log(all_res[all_res > 0])))
axd['g'].vlines(x = [np.mean(np.log(all_res[all_res > 0]))], ymin = 0, ymax = .25, color = 'blue')
print("MEAN RESIDUAL OF NON-CO-EXPRESSED RECEPTORS", np.mean(np.log(all_coexd_res[all_coexd_res > 0])))

axd['g'].set_xlabel("residual")
axd['g'].set_ylabel("density")


#plot mean coexpression pressure difference for co-expressed, non-co-expressed receptors over all trials, noise levels

with open("../results/tables/coexp_pressure_diffs.pkl", 'rb') as f:
    df_press = pkl.load(f)



df_press = df_press[df_press['env_noise'] > 0.1]
# Sort the dataframe by 'env_noise' and 'neur_noise'
df_sorted = df_press.sort_values(by=['neur_noise', 'env_noise'])


sns.barplot(data=df_sorted, x='env_noise', y='diff', hue='neur_noise', ax=axd['h'], ci=None)

for i in range(len(df_sorted)):
    bar = axd['h'].patches[i]
    axd['h'].errorbar(bar.get_x() + bar.get_width() / 2, df_sorted['diff'].iloc[i],
                      yerr=[[df_sorted['diff'].iloc[i] - df_sorted['ci_bottom'].iloc[i]],
                            [df_sorted['ci_top'].iloc[i] - df_sorted['diff'].iloc[i]]],
                      fmt='none', c='gray', capsize = 2)





# plot mutual information for optimal co-expression compared to random co-expression 

var_es_df = []
var_ns_df = []
final_loss = []
shuffle_loss = []
from_curve = []
for trial in range(trials):
    with open(f'../results/tables/opt_coexp_cc/stim_covariance_trial={trial}.pkl', 'rb') as file:
        Cov_xx = pkl.load( file)
    receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
    receptor_cov_t = torch.from_numpy(receptor_cov.astype(np.float32))
    noise_cov = S_scaled @ S_scaled.T
    noise_cov_t = torch.from_numpy(noise_cov.astype(np.float32))
    for i in range(n_grid_e):
        for j in range(n_grid_n):
            #define loss

            #loss_f = mi_loss_new(receptor_cov_t, noise_cov_t, var_ns[j], var_es[i])
            loss_f = mi_loss_new(receptor_cov_t, noise_cov_t, var_ns[j], var_es[i])

            mi_curve = MI_curves[trial, i,j, :]
            A = final_coexp_mats[trial, i, j, :, :]

            #make control matrixes 

            rng = np.random.default_rng()

            n_n, n_r = A.shape
            A_control =  rng.permuted(np.eye(n_n, n_r))
            A_single_opt = final_coexp_mats[trial, 0, j, :, :]
    
            Coex_control = rng.permuted(A)
    


            final_loss.append(loss_f(torch.from_numpy(A.astype(np.float32))).item())
            shuffle_loss.append(loss_f(torch.from_numpy(Coex_control.astype(np.float32))).item())
            from_curve.append(np.max(mi_curve))

            var_es_df.append(var_es[i])
            var_ns_df.append(var_ns[j])


df = pd.DataFrame({"var_e": var_es_df, "var_n": var_ns_df, "final_loss": final_loss,  "shuffle_loss": shuffle_loss, "from_curve": from_curve})
print(df)


sns.lineplot(data = df.loc[df.var_n ==0.0050,:], x = "var_e", y = "final_loss", color = "black", ax = axd['e'] )
sns.lineplot(data = df.loc[df.var_n == 0.0050,:], x = "var_e", y = "shuffle_loss", color = "black", ax = axd['e'], linestyle = "--" )


plt.tight_layout(pad = 0, h_pad = 0.2)
sns.despine()
plt.savefig("../results/plots/carey_carlson.pdf")
plt.show()

fig, axs = plt.subplots(1,3, figsize = (3,1), sharex = True, sharey = True)

sns.lineplot(data = df.loc[df.var_n ==0.0050,:], x = "var_e", y = "final_loss", color = "black", ax = axs[0] )
sns.lineplot(data = df.loc[df.var_n == 0.0050,:], x = "var_e", y = "shuffle_loss", color = "black", ax = axs[0], linestyle = "--" )

sns.lineplot(data = df.loc[df.var_n == 0.01,:], x = "var_e", y = "final_loss", color = "black", ax = axs[1] )
sns.lineplot(data = df.loc[df.var_n == 0.01,:], x = "var_e", y = "shuffle_loss", color = "black", ax = axs[1], linestyle = "--" )

sns.lineplot(data = df.loc[df.var_n == 0.02,:], x = "var_e", y = "final_loss", color = "black", ax = axs[2] )
sns.lineplot(data = df.loc[df.var_n == 0.02,:], x = "var_e", y = "shuffle_loss", color = "black", ax = axs[2], linestyle = "--" )

axs[0].set_xlabel("Enivomnental noise variance")
axs[0].set_ylabel("Mutual information (bits)")
sns.despine()
plt.savefig("../results/plots/compare_to_random_cc.pdf")
plt.show()

with open('../results/tables/opt_coexp_cc/df.pkl', 'rb') as file:
    df = pkl.load(file)

#plot pca alignment and response gain 
fig, axd = plt.subplot_mosaic([['a', 'b']],
                            figsize =(3.6, 1.75))


sns.lineplot(data=df, x="env_noise", y="pca_alignment", hue="neur_noise", ax = axd['a'] )

sns.lineplot(data=df, x="env_noise", y="resp_gain", hue="neur_noise", ax = axd['b'] )



plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/carey_carlson_pca.pdf")
plt.show()

plt.imshow(S, cmap = 'binary')
plt.savefig("../results/plots/carey_carlson_sensing.pdf")
plt.show()