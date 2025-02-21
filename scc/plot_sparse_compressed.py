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



fig, ax = plt.subplots(figsize =(3, 3))


sns.lineplot(data=df, x="env_noise", y="coexp_idx", hue="neur_noise", ax = ax)
ax.set_title("Coexpression level")
sns.despine(ax=ax)


plt.show()


fig, axs = plt.subplots(n_grid_n, n_grid_e, sharex=True, sharey=True)
for i in range(n_grid_e):
    for j in range(n_grid_n):
        for trial in range(trials):
            mi_curve = learning_curves_val[trial, i,j, :]
            axs[j,i].plot(mi_curve[mi_curve > 0], color = 'black')
            axs[j,i].set_title(f"v_e^2 = {var_es[i]:.2f},\nv_n^2={var_ns[j]:.2f} ")


fig.supxlabel("Steps")
fig.supylabel("Mutual information")
plt.suptitle("Learning curves")
plt.tight_layout()
sns.despine()

plt.savefig("../results/plots/learning_curves_mse_cc.pdf")
plt.show()
