## Standard libraries

import numpy as np
import time
import pandas  as pd 
import torch
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
from tqdm.notebook import tqdm
import sys


## Imports for plotting
import matplotlib.pyplot as plt
plt.style.use('paper_style.mplstyle')

import seaborn as sns
from matplotlib.colors import to_rgba, LogNorm,TwoSlopeNorm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 

from src.compute_mi import gauss_info_sensing, get_odorant_cor, get_odorant_var, get_receptor_cor

n_pts = 200

fig, axd = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'f']], figsize = (3.5, 5.25))

overlap_1 = .1
overlap_2 = .4

df1 = pd.DataFrame({"receptor" : [1,1,2,2], "odorant": [1,2,1,2], "affinity" :[1-overlap_1, overlap_1, overlap_1, 1- overlap_1] })
df2 = pd.DataFrame({"receptor" : [1,1,2,2], "odorant": [1,2,1,2], "affinity" :[1-overlap_2, overlap_2, overlap_2, 1- overlap_2] })

sns.barplot(data = df1, x = "odorant", y = "affinity", hue = "receptor", ax = axd["a"])
sns.barplot(data = df2, x = "odorant", y = "affinity", hue = "receptor", ax = axd["b"])
axd["b"].sharey(axd["a"])

S_1 = np.array([[1-overlap_1, overlap_1], [overlap_1, 1-overlap_1]])
S_2 = np.array([[1-overlap_2, overlap_2], [overlap_2, 1-overlap_2]])

p = .5

od_cor_1 = get_odorant_cor(overlap_1, p)
v1 = get_odorant_var(overlap_1, od_cor_1)

od_cor_2 =  get_odorant_cor(overlap_2, p)
v2 = get_odorant_var(overlap_2, od_cor_2)
noise = np.random.multivariate_normal(np.zeros(2), np.eye(2), n_pts)

print("od cor", od_cor_2)
#C_xx_1 = (v1)*np.array([[1, od_cor_1], [od_cor_1, 1]])
# X_1 = np.random.multivariate_normal(np.zeros(2), C_xx_1, n_pts)
# acts_1 = S_2 @ X_1.T
# noise_acts_1 = S_1 @ noise.T

C_xx_2 = (v2)*np.array([[1, od_cor_2], [od_cor_2, 1]])
X = np.random.multivariate_normal(np.zeros(2), C_xx_2, n_pts)
acts = (S_2 @ X.T).T
noise_acts = (S_2 @ noise.T).T

# stim_cov_1 = S_1 @ C_xx_1 @ S_1.T
# E, V= np.linalg.eigh(stim_cov_1)
# signal_ellipse = Ellipse((0,0), width = 3*np.sqrt(E[0]), height = 3*np.sqrt(E[1]), angle = -45)
# signal_ellipse.set_facecolor('none')
# signal_ellipse.set_edgecolor('black')
# axd['c'].add_patch(signal_ellipse)


axd['c'].scatter(X[:,0], X[:,1], s = 10, color = 'black', alpha = .6, edgecolors = 'none')
axd['c'].set_xlim(-10, 10)
axd['c'].set_ylim(-10, 10)
axd['c'].set_box_aspect(1)


axd['d'].scatter(acts[:,0], acts[:,1], s = 10, color = 'black', alpha = .6, edgecolors = 'none')
axd['d'].set_xlim(-5, 5)
axd['d'].set_ylim(-5, 5)
axd['d'].set_box_aspect(1)

axd['e'].scatter(noise[:,0], noise[:,1], s = 10, color = 'black', alpha = .6, edgecolors = 'none')
axd['e'].sharex(axd["c"])
axd['e'].sharey(axd["c"])

axd['e'].set_box_aspect(1)


axd['f'].scatter(noise_acts[:,0], noise_acts[:,1], s = 10, color = 'black', alpha = .6, edgecolors = 'none')
axd['f'].set_xlim(-5, 5)
axd['f'].set_ylim(-5, 5)
axd['f'].set_box_aspect(1)

axd['f'].sharex(axd['d'])
axd['f'].sharey(axd['d'])
axd['f'].set_box_aspect(1)


plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/sensing_overall.pdf")
plt.show()


fig, ax = plt.subplots(1,figsize = (3,3))



odor_cors = np.linspace(-1,1, 100)
tuning_overlaps = np.linspace(0,.5, 100)
ratio = np.zeros((len(odor_cors), len(tuning_overlaps)))
env_noise = .05
neur_noise = .05
for i, od_cor in enumerate(odor_cors):
    for j, tuning_overlap in enumerate(tuning_overlaps):
        C_xx = np.array([[1, od_cor], [od_cor, 1]])
        S = np.array([[1-tuning_overlap, tuning_overlap],[tuning_overlap, 1-tuning_overlap]])
        A_coexp = np.array([[.5, .5]])
        A_single = np.array([[1,0]])
        coexp_info = gauss_info_sensing(C_xx = C_xx,A = A_coexp, w = env_noise, m = neur_noise, S = S )
        single_info = gauss_info_sensing(C_xx = C_xx,A = A_single, w = env_noise, m = neur_noise, S = S )
        ratio[i,j] = coexp_info/single_info

cutoff = 1 / (1 + env_noise/neur_noise)
norm =  TwoSlopeNorm(vmin=np.nanmin(ratio), vcenter=1, vmax=np.nanmax(ratio))
  
cs = ax.imshow(ratio, norm = norm, cmap = 'bwr', extent =(0, .5, -1, 1), origin  = 'lower', aspect = .25 )
ax.hlines([cutoff], xmin = 0, xmax = .5, color = "black")
colorbar = fig.colorbar(cs, ax =ax, ticks =[.25, .5, 1.0, 1.1] )
ax.set_xlabel("tuning overlap")
ax.set_ylabel("odorant correlation")




plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/sensing_heatmaps.pdf")
plt.show()




fig, axs = plt.subplots(figsize = (2,2))
odorant_cors = np.linspace(-1, 1)
tuning_overlaps = [0, .1, .2, .3, .4, .5]

od_cor_list = []
overlap_list = []
receptor_cors = []
for i, odorant_cor in enumerate(odorant_cors):
    for j, tuning_overlap in enumerate(tuning_overlaps):
        receptor_cor = get_receptor_cor(tuning_overlap, odorant_cor)
        od_cor_list.append(odorant_cor)
        overlap_list.append(tuning_overlap)
        receptor_cors.append(receptor_cor)

df = pd.DataFrame({"Odorant correlation": od_cor_list, "Tuning overlap" : overlap_list, "Receptor correlation" :receptor_cors})
sns.lineplot(data = df, x = "Odorant correlation", y = "Receptor correlation", hue = "Tuning overlap")   
plt.scatter(od_cor_2, p, s = 20, color = 'black')    
sns.despine()
plt.savefig("../results/plots/cor_vs_overlap.pdf")

plt.show()