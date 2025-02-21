## Standard libraries
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
plt.style.use('paper_style.mplstyle')

from src.compute_mi import gauss_info_sensing, calculate_roots, coexp_slope


npoints = 200
w_min = 0.001
w_max = 1
m_min = 0.001
m_max = 1

min_ratio = .99
max_ratio = 1.25



fig, axd = plt.subplot_mosaic([["A1", "B", "C", "D1"],["A2", "B", "C", "D2"] ],figsize =(7.2,2.4))

s1 = 1
s2 = 2

S = np.eye(2)
w = .5
m = .1
axd['D1'].scatter([m],[w], s = 5, color = "black")
axd['D2'].scatter([m],[w], s = 5, color = "black")

p = .2
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])


X = np.random.multivariate_normal([0,0], C_xx, size=npoints)
l_lim = np.min(X)
u_lim = np.max(X)
axd['A2'].scatter(X[:,0], X[:, 1], s=5,  color = 'gray')
axd['A2'].set_box_aspect(1)


a1s = np.linspace(0,1)
mi_noise_world_20 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])

p = .8
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])


X = np.random.multivariate_normal([0,0], C_xx, size=npoints)
l_lim = min(np.min(X), l_lim)
u_lim = max(np.max(X), u_lim)
axd['A1'].scatter(X[:,0], X[:, 1], s=5,  color = 'black' )
axd['A1'].set_box_aspect(1)
axd['A1'].sharex(axd['A2'])
axd['A1'].sharey(axd['A2'])
axd["A1"].set_xlim(l_lim, u_lim)
axd["A1"].set_ylim(l_lim, u_lim)



mi_noise_world_80 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])
opt_a1 = calculate_roots(s1, s2, .8, w, m)[1]
opt_a1_val = gauss_info_sensing(C_xx, np.array([[opt_a1, 1-opt_a1]]), w, m, S)





w = 0.05
m = .1

axd['D1'].scatter([m],[w], s = 5, color = "black")
axd['D2'].scatter([m],[w], s = 5, color = "black")

p = .2
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])

mi_clean_world_20 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])



p = .8
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])

mi_clean_world_80 = np.array([gauss_info_sensing(C_xx, np.array([[a1, 1-a1]]), w, m, S) for a1 in a1s])

axd["B"].plot(a1s, mi_clean_world_20, label = "eta^2 = 0.05", color = "black")
axd["B"].plot(a1s, mi_noise_world_20, label = "eta^2 = 0.5", color = "black", linestyle = "--")
axd["B"].legend()
axd["B"].set_xlabel("Receptor 1 expression")
axd["B"].set_ylabel("Mutual information")
axd["B"].set_title("Weak correlation")
axd['B'].set_box_aspect(1)


axd["C"].plot(a1s, mi_clean_world_80, label = "eta^2 = 0.05", color = "black")
axd["C"].plot(a1s, mi_noise_world_80, label = "eta^2 = 0.5", color = "black", linestyle = "--")
axd["C"].scatter(opt_a1, opt_a1_val, s = 10, marker = "*", color = "black")

axd["C"].legend()
axd["C"].set_xlabel("Receptor 1 expression")
axd["C"].set_title("Strong correlation")
axd["C"].sharey(axd["B"])
axd['C'].set_box_aspect(1)


p = .2
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])



result = np.zeros((npoints, npoints))
for i, w in enumerate(np.linspace(w_min, w_max, npoints)):
    for j, m in  enumerate(np.linspace(m_min, m_max, npoints)):
        opt_a1 = calculate_roots(s1, s2, p, w, m)[1]
        opt_a1 = max(opt_a1, 0)
        I_single = gauss_info_sensing(C_xx, np.array([[0,1]]), w, m,  np.eye(2) )
        I_coexp = gauss_info_sensing(C_xx, np.array([[opt_a1, 1-opt_a1]]), w, m , np.eye(2))
        result[i,j] =I_coexp/ I_single


print(min([result.min(), 1]), max([result.max(), 1]))
# result = result > 1 #to show the discrete version
# result = 2 * result
cs = axd['D2'].imshow(result, origin='lower', norm=TwoSlopeNorm(vcenter = 1, vmin=min_ratio, vmax=max_ratio),
    cmap='bwr', extent = (m_min, m_max, w_min, w_max))
axd['D2'].set_xlabel("Neural noise")
axd['D2'].set_ylabel("Environmental noise")
c_s =  coexp_slope(s1,s2, p)
axd['D2'].plot(np.linspace(0,1/c_s),  c_s * np.linspace(0, 1/c_s), color = 'black')
fig.colorbar(cs, ax =axd['D2'])

p = .8
C_xx = np.array([[s1**2, s1*s2*p], [s1*s2*p, s2**2]])

result = np.zeros((npoints, npoints))
for i, w in enumerate(np.linspace(w_min, w_max, npoints)):
    for j, m in  enumerate(np.linspace(m_min, m_max, npoints)):
        opt_a1 = calculate_roots(s1, s2, p, w, m)[1]
        opt_a1 = max(opt_a1, 0)
        I_single = gauss_info_sensing(C_xx, np.array([[0,1]]), w, m,  np.eye(2) )
        I_coexp = gauss_info_sensing(C_xx, np.array([[opt_a1, 1-opt_a1]]), w, m , np.eye(2))
        result[i,j] =I_coexp/ I_single

print(min([result.min(), 1]), max([result.max(), 1]))
# result = result > 1
# result = 2 * result
cs = axd['D1'].imshow(result, origin='lower',  norm=TwoSlopeNorm(vcenter = 1, vmin=min_ratio, vmax=max_ratio),
    cmap='bwr', extent = (m_min, m_max, w_min, w_max))
axd['D1'].set_xlabel("Neural noise")
axd['D1'].set_ylabel("Environmental noise")
c_s =  coexp_slope(s1, s2, p)
fig.colorbar(cs, ax =axd['D1'])
axd['D1'].plot(np.linspace(0,1),  c_s * np.linspace(0, 1), color = 'black')


sns.despine()
plt.savefig("../results/plots/figure_3.pdf")
plt.show()


