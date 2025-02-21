## Standard libraries
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
plt.style.use('paper_style.mplstyle')
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns


from src.compute_mi import gauss_info_sensing

npoints = 200
w_min = 0.001
w_max = 1
m_min = 0.001
m_max = 1

min_ratio = 2/3
max_ratio = 3/2


fig, axd = plt.subplot_mosaic([["A1", "B", "C", "D1"],["A2", "B", "C", "D2"] ],figsize =(7.2,2.4))


S = np.eye(2)
w = .5
m = .1

axd['D1'].scatter([m],[w], s = 5, color = "black")
axd['D2'].scatter([m],[w], s = 5, color = "black")


p = .2
C_xx = np.array([[1, p], [p, 1]])
X = np.random.multivariate_normal([0,0], C_xx, size=npoints)
axd['A2'].scatter(X[:,0], X[:, 1], s=5,  color = 'gray')
axd['A2'].set_box_aspect(1)
axd['A2'].set_xlim(-3, 3)
axd['A2'].set_ylim(-3, 3)

a1s = np.linspace(0,1)
mi_20 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])

p = .8
C_xx = np.array([[1, p], [p, 1]])

X = np.random.multivariate_normal([0,0], C_xx, size=npoints)
axd['A1'].scatter(X[:,0], X[:, 1], s=5,  color = 'black')
axd['A1'].set_box_aspect(1)
axd['A1'].set_xlim(-3, 3)
axd['A1'].set_ylim(-3, 3)


mi_80 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])


axd["B"].plot(a1s, mi_20, label = "\eta = {}".format(w), color = "black", linestyle = "--")
axd["C"].plot(a1s, mi_80, label = "\eta = {}".format(w), color = "black",  linestyle = "--")
axd["B"].set_xlabel("Receptor 1 expression")
axd["B"].set_ylabel("Mutual information")
axd["B"].set_title("Weak correlation")

axd["B"].sharex(axd["C"])
axd['B'].set_box_aspect(1)




w = .05
m = .1

axd['D1'].scatter([m],[w], s = 5, color = "black")
axd['D2'].scatter([m],[w], s = 5, color = "black")


p = .2
C_xx = np.array([[1, p], [p, 1]])
mi_20 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])



p = .8
C_xx = np.array([[1, p], [p, 1]])
mi_80 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])

axd["B"].plot(a1s, mi_20, label = "\eta = {}".format(w), color = "black")
axd["C"].plot(a1s, mi_80, label = "\eta = {}".format(w), color = "black")
axd["C"].legend()
axd["B"].legend()
axd["C"].set_xlabel("Receptor 1 expression")
axd["C"].set_title("Strong correlation")
axd["C"].sharey(axd["B"])
axd['C'].set_box_aspect(1)


p = .2
C_xx = np.array([[1, p], [p, 1]])
result = np.zeros((npoints, npoints))
for i, w in enumerate(np.linspace(w_min, w_max, npoints)):
    for j, m in  enumerate(np.linspace(m_min, m_max, npoints)):
        I_single = gauss_info_sensing(C_xx, np.array([[1,0]]), w, m,  np.eye(2) )
        I_coexp = gauss_info_sensing(C_xx, np.array([[.5,.5]]), w, m , np.eye(2))
        result[i,j] =I_coexp/ I_single

print(min([result.min(), 1]), max([result.max(), 1]))
cs = axd['D2'].imshow(result, origin='lower', norm=TwoSlopeNorm(vcenter = 1, vmin=min_ratio, vmax=max_ratio),
    cmap='bwr', extent = (m_min, m_max, w_min, w_max))

fig.colorbar(cs, ax =axd['D2'], ticks = [.8, 1, 1.25])

axd['D2'].set_xlabel("Neural noise")
axd['D2'].set_ylabel("Environmental noise")
axd['D2'].plot(np.linspace(0, p / (1-p)),  (1-p)/p * np.linspace(0,p / (1-p)), color = 'black')

p = .8
C_xx = np.array([[1, p], [p, 1]])
result = np.zeros((npoints, npoints))
for i, w in enumerate(np.linspace(w_min, w_max, npoints)):
    for j, m in  enumerate(np.linspace(m_min, m_max, npoints)):
        I_single = gauss_info_sensing(C_xx, np.array([[1,0]]), w, m,  np.eye(2) )
        I_coexp = gauss_info_sensing(C_xx, np.array([[.5,.5]]), w, m , np.eye(2))
        result[i,j] =I_coexp/ I_single


cs = axd['D1'].imshow(result, origin='lower', norm=TwoSlopeNorm(vcenter = 1, vmin=min_ratio, vmax=max_ratio),
    cmap='bwr', extent = (m_min, m_max, w_min, w_max))
fig.colorbar(cs, ax =axd['D1'], ticks = [.8, 1, 1.25])

axd['D1'].set_xlabel("Neural noise")
axd['D1'].set_ylabel("Environmental noise")
axd['D1'].plot(np.linspace(0, 1),  (1-p)/p * np.linspace(0,1), color = 'black')



#plt.tight_layout()
sns.despine()
plt.savefig("../results/plots/figure_2_2_receptor.pdf")
plt.show()