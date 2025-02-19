import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

#plotting parameters
plt.style.use('paper_style.mplstyle')


df = pd.read_csv("../data/elife-14925-fig1-data1.csv")
odorants = list(df.columns)[1:]

neuron_dict = {"pb1A": odorants[0:5], "pb1B": odorants[5:11], "pb2A": odorants[11:18], "pb2B": odorants[18:20], "pb3A" :odorants[20:26], "pb3B":odorants[26:] }
receptor_dict = {"Or42a": odorants[0:5], "Or71a":odorants[5:11], "Or33c": ['Strawberry Furanone'], "Or85e": ["(-)-camphore", 'Alpha-ionone', 'Beta-Ionone'], "Or46a":odorants[18:20], "Or59c" :odorants[20:26], "Or85d":odorants[26:] }

  
corr_matrix = df.iloc[:, 1:].corr()

norm =  TwoSlopeNorm(vmin=np.min(corr_matrix.to_numpy()), vcenter=0, vmax=np.max(corr_matrix.to_numpy()))

fig, axs = plt.subplots(1, 2, figsize=(7,3), width_ratios=[1.5, 1])
cmap = axs[0].matshow(corr_matrix , cmap = 'bwr', norm =norm)
axs[0].set_xticks(range(len(odorants)), odorants, rotation=45, ha="right",  fontsize = 5)
axs[0].set_yticks(range(len(odorants)), odorants, fontsize = 6)
cb = fig.colorbar(cmap, ax = axs[0], shrink = .67)
axs[0].set_title('Correlation Matrix'); 
axs[0].xaxis.set_ticks_position('bottom')

coexd_corr = corr_matrix.loc["(-)-camphore", "Strawberry Furanone" ]
print("coexpressed:", coexd_corr)
print(corr_matrix.columns)
cross_receptor_cors = []
same_receptor_cors = []

for i, receptor_1 in enumerate(receptor_dict):
    for j, receptor_2 in enumerate(receptor_dict):
        print(receptor_1, receptor_2)
        if i  == j:
            for k, odorant_1 in enumerate(receptor_dict[receptor_1]):
                for l, odorant_2 in enumerate(receptor_dict[receptor_2]):
                    if k > l:
                      same_receptor_cors.append(corr_matrix.loc[odorant_1, odorant_2])
        elif i  > j:
            for k, odorant_1 in enumerate(receptor_dict[receptor_1]):
                for l, odorant_2 in enumerate(receptor_dict[receptor_2]):
                      cross_receptor_cors.append(corr_matrix.loc[odorant_1, odorant_2])  


print("correlation of coexpressed odorants: ", coexd_corr)
corr_matrix = corr_matrix.to_numpy()
corrs = corr_matrix[np.triu_indices(corr_matrix.shape[0], 1)]
axs[1].hist(cross_receptor_cors, color = "black", alpha = .5, bins = np.linspace(-.5, 1, 30))
#axs[1].hist(same_receptor_cors, color = "red", alpha = .5, bins = np.linspace(-.5, 1, 30))
sns.despine()
ymin, ymax = axs[1].get_ylim()
axs[1].vlines([coexd_corr], ymax=ymax, ymin=ymin, color = "black")
plt.savefig("../results/plots/natural_odor_cors.pdf")
plt.show()

