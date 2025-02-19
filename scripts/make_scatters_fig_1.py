## Standard libraries
import sys
import os
import math
import numpy as np
import time
import pandas  as pd 
import torch
torch.manual_seed(1) # Setting the seed
torch.device("mps") 
from tqdm.notebook import tqdm
import sys
import seaborn as sns

## Imports for plotting
import matplotlib.pyplot as plt

plt.style.use('paper_style.mplstyle')

from src.compute_mi import gauss_info, gauss_info_sensing
fig, axs = plt.subplots(1,3, figsize =(4,1.33), sharex = True, sharey= True)

npoints = 200
p = .8
C_xx = np.array([[1, p], [p, 1]])

X_signal = np.random.multivariate_normal([0,0], C_xx, size=npoints)
axs[1].scatter(X_signal[:,0], X_signal[:, 1], s=5,  color = 'black')
axs[1].set_box_aspect(1)
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-4, 4)
axs[1].set_xticks([-2,0,2])
axs[1].set_yticks([-2,0,2])


X_noise = np.random.multivariate_normal([0,0], 0.2 * np.eye(2), size=npoints)
axs[2].scatter(X_noise[:,0], X_noise[:, 1], s=5,  color = 'black')
axs[2].set_box_aspect(1)
axs[2].set_xlim(-4, 4)
axs[2].set_ylim(-4, 4)

X_total = X_signal + X_noise
fig.supxlabel("Odorant 1 concentration")
fig.supylabel("Odorant 2 concentration")

axs[0].scatter(X_total[:,0], X_total[:, 1], s=5,  color = 'black')
axs[0].set_box_aspect(1)
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)

plt.tight_layout(pad=0.2, w_pad=0, h_pad=1)

plt.savefig("../results/plots/scatters_for_overview.pdf")
plt.show()

fig, axs = plt.subplots(1,4, figsize = (4,1.25), sharex = True, sharey = True)

X_signal_high_corr = np.random.multivariate_normal([0,0], np.array([[1, .8], [.8, 1]]), size=npoints)
X_signal_low_corr = np.random.multivariate_normal([0,0], np.array([[1, .2], [.2, 1]]), size=npoints)

X_noise_small = np.random.multivariate_normal([0,0], 0.5 * np.eye(2), size=npoints)
X_noise_big = np.random.multivariate_normal([0,0], 0.05 * np.eye(2), size=npoints)

axs[0].scatter(X_signal_high_corr[:,0], X_signal_high_corr[:, 1], s=5,  color = 'black')
axs[0].set_box_aspect(1)
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].set_xticks([-2,0,2])
axs[0].set_yticks([-2,0,2])

axs[1].scatter(X_signal_low_corr[:,0], X_signal_low_corr[:, 1], s=5,  color = 'black')
axs[1].set_box_aspect(1)


axs[2].scatter(X_noise_small[:,0], X_noise_small[:, 1], s=5,  color = 'black')
axs[2].set_box_aspect(1)



axs[3].scatter(X_noise_big[:,0], X_noise_big[:, 1], s=5,  color = 'black')
axs[3].set_box_aspect(1)

plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/noise_scatters.pdf")
plt.show()

fig, axd  = plt.subplot_mosaic([["s1", "s2", "_"], ["s1+n1", "s2+n1", "n1"], [ "s1+n2", "s2+n2", "n2"]], figsize = (6,6), sharex=True, sharey=True )

axd["s1"].scatter(X_signal_low_corr[:,0], X_signal_low_corr[:,1], s = 10, color = 'black' )
axd["s2"].scatter(X_signal_high_corr[:,0], X_signal_high_corr[:,1], s = 10, color = 'black' )

axd["n1"].scatter(X_noise_small[:,0], X_noise_small[:,1], s = 10, color = 'black' )
axd["n2"].scatter(X_noise_big[:,0], X_noise_big[:,1], s = 10,color = 'black' )

axd["s1+n1"].scatter(X_signal_low_corr[:,0]+X_noise_small[:,0], X_signal_low_corr[:,1]+X_noise_small[:,1], s = 10 , color = 'black')
axd["s1+n2"].scatter(X_signal_low_corr[:,0]+X_noise_big[:,0], X_signal_low_corr[:,1]+X_noise_big[:,1], s = 10 , color = 'black' )
axd["s2+n1"].scatter(X_signal_high_corr[:,0]+X_noise_small[:,0], X_signal_high_corr[:,1]+X_noise_small[:,1], s = 10 , color = 'black')
axd["s2+n2"].scatter(X_signal_high_corr[:,0]+X_noise_big[:,0], X_signal_high_corr[:,1]+X_noise_big[:,1], s = 10 , color = 'black')

sns.despine()
plt.savefig("../results/plots/signal_and_noise.pdf")
plt.show()
