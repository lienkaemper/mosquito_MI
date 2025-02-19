## Standard libraries
import numpy as np
import pandas  as pd 
import torch
import seaborn as sns
torch.manual_seed(1) # Setting the seed
torch.device("mps") 

from scipy import stats
import pickle as pkl


## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.style.use('paper_style.mplstyle')


from scipy.cluster.hierarchy import  linkage, leaves_list, optimal_leaf_ordering


def order_cor_matrix(C):
    n = C.shape[0]
    D = np.ones((n,n)) - C
    Z = linkage(D[np.triu_indices(n,1)])
    optimal_leaf_ordering(Z, D[np.triu_indices(n,1)])
    ll = leaves_list(Z)
    return C[np.ix_(ll, ll)], ll, Z

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

with open("../results/tables/sparse/coexp_mats.pkl", 'rb') as f:
    final_coexp_mats = pkl.load(f)

with open("../results/tables/sparse/learning_curves_train.pkl", 'rb') as f:
    learning_curves_train = pkl.load(f)

with open("../results/tables/sparse/learning_curves_train.pkl", 'rb') as f:
    learning_curves_val = pkl.load(f)

with open('../results/tables/sparse/df.pkl', 'rb') as file:
    df = pkl.load(file)

with open("../results/tables/sparse/var_es.pkl", 'rb') as f:
    var_es = pkl.load(f)

with open("../results/tables/sparse/var_ns.pkl", 'rb') as f:
    var_ns = pkl.load(f)

with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S = pkl.load(f)

df['neur_noise'] = df['neur_noise'].astype('float32') 

trials = 3
n_grid_e = len(var_es)
n_grid_n = len(var_ns)


fig, axd = plt.subplot_mosaic([['blank', 'a', 'b', 'c1'],
                            ['blank', 'a', 'b', 'c2'],
                            ['d', 'e', 'f', 'g'],
                            ['d', 'e', 'f', 'g']],
                            figsize =(7.08, 3))


trial = 0
with open(f'../results/tables/sparse/stim_covariance_trial={trial}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)
#C, ll, z = order_cor_matrix(cov_to_cor(Cov_xx))

norm =  TwoSlopeNorm(vmin=np.nanmin(Cov_xx), vcenter=0, vmax=np.nanmax(Cov_xx))
#cs = axd['a'].imshow(Cov_xx[np.ix_(ll, ll)], cmap = 'bwr', norm = norm)
cs = axd['a'].imshow(Cov_xx, cmap = 'bwr', norm = norm)
fig.colorbar(cs, ax =axd['a'])
axd['a'].set_title("stimulus covariance")


for trial in range(trials):
    lc_val = learning_curves_val[trial, 2,1, :]
    axd['b'].plot(lc_val)
    lc_train = learning_curves_train[trial, 2,1, :]
    axd['b'].plot(lc_train)
axd['b'].set_title("learning curves")
axd['b'].set_xlabel("steps")
axd['b'].set_ylabel("mutual information")


i_low = 0
j_low = 1
coexp_mat = final_coexp_mats[trial, i_low,j_low, :, :]
cs = axd['c1'].pcolor(coexp_mat, vmin = 0, vmax = 1, cmap = 'binary')
axd['c1'].set_title("env_noise = {}, neur_noise = {}".format(var_es[i_low], var_ns[j_low]))
axd['c1'].set_xlabel("receptors")
axd['c1'].set_ylabel("neurons ")
axd['c1'].set_xticks([])
axd['c1'].set_yticks([])

fig.colorbar(cs, ax =axd['c1'])

i_high = 3
j_high = 2
coexp_mat = final_coexp_mats[trial, i_high, j_high, :, :]
cs = axd['c2'].pcolor(coexp_mat, vmin = 0, vmax = 1, cmap = 'binary')
fig.colorbar(cs, ax =axd['c2'])
axd['c2'].set_xticks([])
axd['c2'].set_yticks([])
axd['c2'].set_xlabel("receptors")
axd['c2'].set_ylabel("neurons ")
axd['c2'].set_title("env_noise = {}, neur_noise = {}".format(var_es[i_high], var_ns[j_high]))



sns.lineplot(data=df, x="env_noise", y="coexp_idx", hue="neur_noise", ax = axd['d'])
axd['d'].set_title("coexpression index ")




trial = 0
with open(f'../results/tables/sparse/stim_covariance_trial={trial}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)

S_scaled = S/(10**3)
receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
noise_cov = S_scaled  @ S_scaled.T

receptor_cor = cov_to_cor(receptor_cov)
noise_cor = cov_to_cor(noise_cov)

res = stats.linregress(noise_cor.flatten(), receptor_cor.flatten())

b0 = res.intercept
b1 = res.slope

residual = receptor_cor - (b0 + b1*(noise_cor))

residual = (receptor_cor - noise_cor)
residual[np.isinf(residual)] = np.nan
A = final_coexp_mats[trial, 0, 2,:, :]


C, N = coexd_receptors(A, thresh = .1)


axd['e'].scatter(noise_cor.flatten(), receptor_cor.flatten(), color = 'blue', s = 10)
axd['e'].plot(noise_cor.flatten(), res.intercept + res.slope*noise_cor.flatten(),  label='fitted line', color = 'black')
axd['e'].set_xlabel("Noise correlation")
axd['e'].set_ylabel("Signal correlation")

coexd_x = noise_cor[C > 0]
not_coexd_x = noise_cor[C  == 0]

coexd_y = receptor_cor[C > 0]
not_coexd_y = receptor_cor[C  == 0]

axd['e'].scatter(coexd_x, coexd_y, color = 'red', s = 10)

coexd = []
not_coexd = []
for trial in range(trials):
    with open(f'../results/tables/sparse/stim_covariance_trial={trial}.pkl', 'rb') as file:
            Cov_xx = pkl.load( file)
    receptor_cov = S_scaled @ Cov_xx @ S_scaled.T
    noise_cov = S_scaled  @ S_scaled.T

    receptor_cor = cov_to_cor(receptor_cov)
    noise_cor = cov_to_cor(noise_cov)
    residual = receptor_cor 
    for i in range(1,n_grid_e):
        for j in range(n_grid_n):
            C, N = coexd_receptors(A, thresh = .1)
            A = final_coexp_mats[trial, 0, 2,:, :]
            coexd.extend(residual[C > 0])
            not_coexd.extend(residual[C  == 0])

axd['f'].hist(coexd, color = 'red', alpha  =.5, density=True)
axd['f'].hist(not_coexd, color = 'blue', alpha  =.5, density=True)
axd['f'].vlines(x = [np.mean(coexd)], ymin = 0, ymax = 3, color = 'red')
axd['f'].vlines(x = [np.mean(not_coexd)], ymin = 0, ymax = 3, color = 'blue')
axd['f'].set_xlabel("residual")
axd['f'].set_ylabel("probability")


# axd['f2'].hist(coexd, color = 'red', alpha  =.5, density=True)
# axd['f2'].hist(not_coexd, color = 'blue', alpha  =.5, density=True)
# axd['f2'].vlines(x = [np.mean(coexd)], ymin = 0, ymax = 3, color = 'red')
# axd['f2'].vlines(x = [np.mean(not_coexd)], ymin = 0, ymax = 3, color = 'blue')

sns.lineplot(data=df, x="env_noise", y="coexd_res", hue="neur_noise",  ax = axd['g'])

plt.tight_layout(pad = 0, h_pad = 0.2)
sns.despine()
plt.savefig("../results/plots/sparse.pdf")
plt.show()




fig, axd = plt.subplot_mosaic([['a', 'b'], ['c', 'd']],
                            figsize =(3.5, 3.5))


sns.lineplot(data=df, x="env_noise", y="pca_alignment", hue="neur_noise", ax = axd['c'])
axd['c'].set_title("pca alignment")

sns.lineplot(data=df, x="env_noise", y="resp_gain", hue="neur_noise", ax = axd['d'])
axd['d'].set_title("response gain")



plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/sparse_pca.pdf")
plt.show()
