import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import torch
from tqdm import tqdm

from src.optimize_mi import maximize_mi



plt.style.use('paper_style.mplstyle')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

n_odors = 10
n_points = 200
trials = 5
opt_steps = 500
plot = False
dimension = 2

e_min = 0
e_max = 2
s_min =0
s_max = 1
n_e = 10
n_s = 4
etas = np.linspace(e_min, e_max, n_e)
sigmas = np.round(np.linspace(s_min, s_max, n_s), decimals = 2)

V = np.random.rand(n_odors, n_odors)
V[0,:] *= 2
V[1,:] *= 1.25
C_xx = V @ V.T

eigs = np.linalg.eig(C_xx)
E = np.diag(np.sort(eigs.eigenvalues)[::-1])
V = eigs.eigenvectors
V[:, 0] *= np.sign(V[0,0])
pc1 = V[:, 0]

neuron_ids = (np.argsort(np.diag(C_xx))[-3:])[::-1]
top_var = C_xx[neuron_ids[0], neuron_ids[0]]






X = np.random.multivariate_normal(np.zeros(n_odors), C_xx, size=n_points)
receptor_cov_t = torch.from_numpy(C_xx.astype(np.float32))
noise_cov_t = torch.eye(n_odors)




fig = plt.figure(facecolor=None, figsize=(4, 4))
if dimension == 3:
    ax3d = fig.add_subplot(111, projection='3d')
    mu = 7
    axlen = 24

    x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = 1, neur_noise = 0, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
    A = x.detach().numpy()
    t = np.linspace(-50, 50).reshape(50,1)
    pts = t @ A
    responses_noisy =( A @ X.T ).flatten()
    gain_noisy = np.sqrt(A @ A.T).item()
    projections_noisy = responses_noisy/gain_noisy
    ax3d.plot(pts[:, neuron_ids[0]]+mu, pts[:, neuron_ids[1]]+mu,  pts[:, neuron_ids[2]]+mu,  color = "r", linewidth=1.5, zorder = 0)


    x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = 0, neur_noise = 1, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
    A = x.detach().numpy()
    t = np.linspace(-top_var, top_var).reshape(50,1)
    pts = t @ A
    responses_clean = (A @ X.T ).flatten()
    gain_clean = np.sqrt(A @ A.T).item()
    projections_clean = responses_clean/gain_clean
    ax3d.plot(pts[:, neuron_ids[0]]+mu, pts[:, neuron_ids[1]]+mu,  pts[:, neuron_ids[2]]+mu,  color = "b", linewidth=1.5, zorder = 0)



    ax3d.scatter(X[:, neuron_ids[0]]+mu, X[:, neuron_ids[1]]+mu, X[:, neuron_ids[2]]+mu, color = 'k' , alpha = .2, linewidth = 0)
    ax3d.plot(xs=[0, 0], ys=[0, 0], zs=[0, axlen], color='k', linewidth=1)
    ax3d.plot(xs=[0, axlen], ys=[0, 0], zs=[0, 0], color='k', linewidth=1)
    ax3d.plot(xs=[0, 0], ys=[0, axlen], zs=[0, 0], color='k', linewidth=1)
    ax3d.scatter(xs = [10, 0, 0, 10/3], ys = [0, 10, 0, 10/3], zs = [0,0,10, 10/3], color = 'k', marker = "*", alpha = 1 )
    ax3d.view_init(43, -35, 0)

    plt.axis('off')

    plt.savefig("../results/plots/both_subspaces_3d.png", transparent=True, dpi = 1000)
    plt.show()

if dimension == 2:
    ax2d = fig.add_subplot(111)
    mu = 0
    axlen = 24

    x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = 1, neur_noise = 0, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
    A = x.detach().numpy()
    t = np.linspace(-50, 50).reshape(50,1)
    pts = t @ A
    responses_noisy =( A @ X.T ).flatten()
    gain_noisy = np.sqrt(A @ A.T).item()
    projections_noisy = responses_noisy/gain_noisy
    ax2d.plot(pts[:, neuron_ids[0]]+mu, pts[:, neuron_ids[1]]+mu,    color = "r", linewidth=1.5)


    x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = 0, neur_noise = 1, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
    A = x.detach().numpy()
    t = np.linspace(-top_var, top_var).reshape(50,1)
    pts = t @ A
    responses_clean = (A @ X.T ).flatten()
    gain_clean = np.sqrt(A @ A.T).item()
    projections_clean = responses_clean/gain_clean
    ax2d.plot(pts[:, neuron_ids[0]]+mu, pts[:, neuron_ids[1]]+mu,    color = "b", linewidth=1.5)



    ax2d.scatter(X[:, neuron_ids[0]]+mu, X[:, neuron_ids[1]]+mu,  color = 'k' , alpha = .2, linewidth = 0)

    ax2d.set_xlim(-12, 12)
    ax2d.set_ylim(-12, 12)
    sns.despine()
    plt.savefig("../results/plots/perceptual_subspaces/both_subspaces_2d.pdf", transparent=True, dpi = 1000)
    plt.show()


fig = plt.figure(facecolor=None, figsize=(2,2))
ax = fig.add_subplot(111)
mu = 7
axlen = 24

x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = 1, neur_noise = 0.5, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
A = x.detach().numpy()
t = np.linspace(-1.5*top_var, 1.5*top_var).reshape(50,1)
pts = t @ A
responses_ex =( A @ X.T ).flatten()
gain_ex= np.sqrt((A @ A.T).item())
projections_ex = responses_ex/gain_ex
proj_mat = (A.T @ A)/gain_ex**2
ax.plot(pts[:, neuron_ids[0]], pts[:, neuron_ids[1]],   color = "k", linewidth=2., zorder = 0)


diff =  3*np.random.rand(n_odors)

diff_ortho = diff - proj_mat@ diff


highlight_points = np.array([X[0, :], X[0, :] + diff, X[0, :] + diff_ortho]) 
highlight_projections = (proj_mat @ highlight_points.T).T
arrow_dirs = highlight_projections - highlight_points

ax.scatter(highlight_points[:, neuron_ids[0]], highlight_points[:, neuron_ids[1]], color = 'm' , alpha = 1, s = 20, zorder = 10)
ax.scatter(highlight_projections[:, neuron_ids[0]], highlight_projections[:, neuron_ids[1]],color = 'c' , alpha = 1, s = 20, zorder = 10)


ax.scatter(X[:, neuron_ids[0]], X[:, neuron_ids[1]], color = 'k' , s = 15, alpha = .2)
sns.despine()
plt.gca().set_aspect('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.savefig("../results/plots/perceptual_subspaces/subspace_projections.pdf")
plt.show()





fig, axs = plt.subplots(1,2,figsize = (4,2))

axs[0].plot(projections_noisy, responses_noisy, color = 'red')
axs[0].plot(projections_clean, responses_clean, color = "blue")


axs[1].hist(responses_noisy, color = "red", alpha = .5, density=True, orientation='horizontal')
axs[1].hist(responses_clean, color = 'blue', alpha = .5, density=True, orientation='horizontal')
sns.despine()
plt.savefig("../results/plots/perceptual_subspaces/hists.pdf", transparent=True)

plt.show()



#final_cov_mats = np.zeros((trials, n_grid_e, n_grid_n, n_n, n_r))
MI_curves = np.zeros((trials, n_e, n_s, opt_steps))
resp_gain = []
pc_alignment = []
eta_list = []
sigma_list = []
resp_sig_var = []
with tqdm(total = n_e*n_s * trials) as pbar:
    for trial in range(trials):
        for i, eta in enumerate(etas):
            for j, sigma in enumerate(sigmas):
                if max(eta,sigma) > 0:
                    x, gradients, values ,xs =maximize_mi(receptor_cov_t, noise_cov_t, env_noise = eta, neur_noise = sigma, m = 1, steps = opt_steps, rate = .1, grad_noise = .05, grad_tol = 0)
                    A = x.detach().numpy()
                    A = A[np.argmax(A, axis = 1).argsort(), :]
                    MI_curves[trial, i, j, :] = values
                    resp_gain.append(np.sqrt(np.sum(A**2)))
                    pc_alignment.append(((A @ pc1)/np.sqrt((A @ A.T) * (pc1.T @ pc1))).item())
                    resp_sig_var.append((A @ C_xx @ A.T).item() )
                    eta_list.append(eta)
                    sigma_list.append(sigma)
                    if plot:
                        fig, axs = plt.subplots(1,2)
                        axs[0].plot(values)
                        axs[1].imshow(A)
                        plt.title("eta = {:.2f}, sigma = {:.2f}".format(eta, sigma))
                        plt.show()
                pbar.update(1)

df = pd.DataFrame({"response gain" : resp_gain, "pc alignment" : pc_alignment, "neural noise variance": sigma_list, "response signal variance" : resp_sig_var, "environmental noise variance" : eta_list})

print(df)
fig, axs = plt.subplots(1,2, figsize = (4,2))
sns.lineplot(data = df, x  = "environmental noise variance", y = "pc alignment", hue = "neural noise variance", ax = axs[0])
sns.lineplot(data = df, x  = "environmental noise variance", y = "response gain", hue = "neural noise variance", ax = axs[1])
sns.despine()
plt.savefig("../results/plots/perceptual_subspaces/tradeoff.pdf")
plt.show()





