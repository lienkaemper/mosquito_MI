with open("non_gauss_output.txt", "a") as f:
    f.write("top of file, running!\n ")
    
# Imports
import numpy as np
import matplotlib.pyplot as plt


import pickle as pkl
from tqdm import tqdm
import pandas as pd
import seaborn as sns


plt.style.use('paper_style.mplstyle')


#== FUNCTIONS
# Defines distribution 
def sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    sources_signal = np.random.rand(n_source_signal, n_odorants) < p_source
    sources_signal = 10*np.random.rand(n_source_signal, n_odorants) * sources_signal

    sources_noise = np.random.rand(n_source_noise, n_odorants) < p_source
    sources_noise =10*np.random.rand(n_source_noise, n_odorants) * sources_noise
    def sample(n_samples = 1, n_samples_noise = 1):
        X_signal = np.random.rand(n_samples, n_odorants)
        X_noise = np.random.rand(n_samples_noise, n_odorants)
        Y = np.random.rand(n_samples, n_source_signal)
        for i in range(n_samples):
            noisy_sources_signal = sources_signal +  concentration_noise * np.random.randn(n_source_signal, n_odorants) * (sources_signal > 0) #add concentration noise to signal and noise
            noisy_sources_signal *= (noisy_sources_signal > 0)
            ids_signal = np.random.rand(n_source_signal) < p_sample
            coeffs_signal =  ids_signal * np.random.rand(n_source_signal)
            Y[i,:] = coeffs_signal 
            X_signal[i,:] = coeffs_signal.T @ noisy_sources_signal
        
        for i in range(n_samples_noise):
            noisy_sources_noise = sources_noise +  concentration_noise * np.random.randn(n_source_noise, n_odorants) * (sources_noise > 0)
            noisy_sources_noise = noisy_sources_noise * (noisy_sources_noise > 0)
            ids_noise = np.random.rand(n_source_noise) < p_sample
            coeffs_noise = ids_noise * np.random.rand(n_source_noise)
            X_noise[i, :] =  noise_strength * coeffs_noise.T @ noisy_sources_noise

        return X_signal.astype(np.float16), X_noise.astype(np.float16), Y.astype(np.float16)
    return sample, sources_signal, sources_noise

def bintropy(X, bins =100, plot = False):
    hist, edges = np.histogram(X, bins, density=True)
    widths = np.diff(edges)
    inds = np.nonzero(hist)
    result = np.sum(-widths[inds]*hist[inds]*np.log(hist[inds]))
    if plot:
        plt.bar(edges[:-1], hist, width = np.diff(edges))
        plt.title("entropy = {}".format(result))
        plt.show()
    return result

def entropy_est_bin(X, env_noises, neur_noises, f, bins = 100, plot = False, conditional_constant = False):
    m_samples, n_in = X.shape
    Hs = np.zeros(m_samples)
    n_e, _ = env_noises.shape
    n_n, n_out = neur_noises.shape


    # Precompute the noise combinations once
    env_combs = np.repeat(env_noises, n_n, axis=0)
    neur_combs = np.tile(neur_noises, (n_e, 1))

    big_Y = np.zeros((m_samples * n_e * n_n, n_out))
    start = 0

    if conditional_constant == True:
        for i in range(m_samples):
            x = X[i, :]  # (n_in,)
            
            # Compute f(x, xi, zeta) in a vectorized way
            X_broadcast = np.tile(x, (n_e * n_n, 1))  # (n_e * n_n, n_in)
            
            # Apply function f to all combinations at once
            Y = f(X_broadcast, env_combs, neur_combs)
            # Calculate conditional entropy 
            if plot == True:
                if i  == 0:
                    Hs += bintropy(Y, bins, plot = plot)
            else:
                if i == 0:
                    Hs+= bintropy(Y, bins)
            big_Y[start:start + n_e*n_n, :] += Y 
            start += n_e*n_n
    else:
        for i in range(m_samples):
            x = X[i, :]  # (n_in,)
            
            # Compute f(x, xi, zeta) in a vectorized way
            X_broadcast = np.tile(x, (n_e * n_n, 1))  # (n_e * n_n, n_in)
            
            # Apply function f to all combinations at once
            Y = f(X_broadcast, env_combs, neur_combs)
            # Calculate conditional entropy 
            if plot == True:
                if i % 100 == 0:
                    Hs[i] = bintropy(Y, bins, plot = plot)
                else:
                    Hs[i] = bintropy(Y, bins) 
            else:
                Hs[i] = bintropy(Y, bins)
            big_Y[start:start + n_e*n_n, :] += Y 
            start += n_e*n_n

    response_entropy = bintropy(big_Y, bins, plot = plot) 
    conditional_entropy = np.mean(Hs)
    mutual_info = response_entropy-conditional_entropy
    return (mutual_info, response_entropy, conditional_entropy)




def mi_loss_sample_bin(signal_sample, env_noise_sample, neur_noise_sample, bins = 100, plot = False, print_val = False, conditional_constant = False):
    def loss_fn(A):
        def f(X_broadcast, env_combs, neur_combs):
            # X_broadcast: (n_e * n_n, n_in), batch of x inputs
            # env_combs: (n_e * n_n, n_in), batch of xi values
            # neur_combs: (n_e * n_n, n_out), batch of zeta values
            
            # Step 1: Perform the element-wise addition (x + xi)
            Y = X_broadcast + env_combs  # Broadcasting works automatically here
            
            # Step 2: Perform the matrix multiplications (A @ S @ (x + xi))
            result = A  @ Y.T  # Transpose to align shapes for matmul (n_in, n_e * n_n)
            # Step 3: Transpose result to get it back to original shape
            result = result.T  # Now result is (n_e * n_n, n_out)
            #print(result.shape)

            # Step 4: Add zeta element-wise
            result = result + neur_combs  # Broadcasting for zeta
            
           # print("neur noise shape", neur_combs.shape)
          # print(result.shape)
            return result
        result = entropy_est_bin(signal_sample, env_noise_sample, neur_noise_sample, f, bins, plot, conditional_constant)
        if print_val:
            print("mutual info:", result[0], "response entropy", result[1], "conditional entropy", result[2])
        return result[0]
    return loss_fn
#====================================


np.random.seed(1)
n_samples_signal = 500
n_samples_noise = 500
n_odorants =  100
n_receptors = 2
n_neurons = 1
p_source = .05
p_sample = .1
n_source_signal = 50
n_source_noise = 1000
noise_strength = 1
concentration_noise = .1


with open("../data/carey_carlson_sensing.pkl", "rb") as f:
    S_cc = pkl.load(f) 

weights = S_cc.flatten()
np.random.shuffle(weights)
S = weights[0:n_odorants*n_receptors].reshape(n_receptors, n_odorants)


S = ((1/np.sum(S, axis = 1)) * S.T).T
S = S.astype(np.float16)

plt.imshow(S)



sampler, sources_signal, sources_noise = sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, noise_strength = noise_strength, concentration_noise = concentration_noise)
signal, env_noises , _ = sampler(n_samples_signal, n_samples_noise)
Sigma_xx = np.cov(signal, rowvar = False)


noise_cov_mat = np.cov(env_noises.T)
noise_var = np.mean(np.diag(noise_cov_mat))
env_noises = env_noises/(np.sqrt(noise_var))
noise_cov_mat = np.cov(env_noises.T)
noise_var = np.mean(np.diag(noise_cov_mat))

neur_noises = np.random.multivariate_normal(mean = np.zeros(n_neurons), cov = np.eye(n_neurons), size = n_samples_noise).astype(np.float16)


p_source = .2
p_sample = .4
n_source_signal = 5


sampler, sources_signal, sources_noise = sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, noise_strength = noise_strength, concentration_noise = concentration_noise)
signal_h, _ , _ = sampler(n_samples_signal)
Sigma_xx_h = np.cov(signal_h, rowvar = False)

response = (S @ signal.T).T
sig_cor = np.corrcoef(response,rowvar=False)[0,1]

response_h = (S @ signal_h.T).T
sig_cor_h = np.corrcoef(response_h,rowvar=False)[0,1]

noise_response = (S @ env_noises.T).T
noise_cor = np.corrcoef(noise_response,rowvar=False)[0,1]



fig, axs = plt.subplots(2,4, figsize = (6,3))


axs[0, 1].hist(signal.flatten(), color = 'k', density = True)
axs[0, 1].set_xlabel("number of odorants")
axs[0, 1].set_ylabel("number of samples")

axs[0,2].scatter(response[:,0], response[:,1], s = 10, color = 'k')
axs[0,2].set_title(f"signal correlation = {sig_cor:.2f}")
axs[0,3].scatter(response_h[:,0], response_h[:,1], s = 10,  color = 'k')
axs[0,3].set_title(f"signal correlation = {sig_cor_h:.2f}")
axs[0,3].sharex(axs[0,2])
axs[0,3].sharey(axs[0,2])


n_grid = 10
v_e = .05
v_n = .1
f = mi_loss_sample_bin(response_h, v_e* noise_response, v_n* neur_noises, bins = 50, plot = False, print_val = False, conditional_constant=True)

a1s = np.linspace(0, 1, n_grid)
mis = np.zeros(n_grid)
g_approxs = np.zeros(n_grid)
for i, a in enumerate(tqdm(a1s)):
    A = np.array([[a, 1-a]]).astype(np.float16)
    mi = f(A)
    mis[i] += mi
    cond_entropy_ext = .5 * np.log(np.linalg.det( v_e**2 *A @ S @ S.T @ A.T+ v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
    resp_entropy_ext = .5 * np.log(np.linalg.det(A @ S @ (Sigma_xx_h +  v_e**2 *np.eye(n_odorants))@ S.T @ A.T +  v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
    mi_gauss =  resp_entropy_ext - cond_entropy_ext
    g_approxs[i] += resp_entropy_ext - cond_entropy_ext

axs[1,0].plot(a1s, mis, label = "sampled")
axs[1,0].plot(a1s, g_approxs, label = "gauss approximaiton")
axs[1,0].legend()

v_e = .5
v_n = .1
f = mi_loss_sample_bin(response_h, v_e*noise_response, v_n* neur_noises, bins = 50, plot = False, print_val = False, conditional_constant=True)

a1s = np.linspace(0, 1, n_grid)
mis = np.zeros(n_grid)
g_approxs = np.zeros(n_grid)
for i, a in enumerate(tqdm(a1s)):
    A = np.array([[a, 1-a]]).astype(np.float16)
    mi = f(A)
    mis[i] += mi
    cond_entropy_ext = .5 * np.log(np.linalg.det( v_e**2 *A @ S @ S.T @ A.T+ v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
    resp_entropy_ext = .5 * np.log(np.linalg.det(A @ S @ (Sigma_xx_h  +  v_e**2 *np.eye(n_odorants))@ S.T @ A.T +  v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
    mi_gauss =  resp_entropy_ext - cond_entropy_ext
    g_approxs[i] += resp_entropy_ext - cond_entropy_ext

axs[1,1].plot(a1s, mis, label = "sampled")
axs[1,1].plot(a1s, g_approxs, label = "gauss approximaiton")
axs[1,1].legend()

print("line plots done")
with open("non_gauss_output.txt", "a") as f:
    f.write("line plots done")
  
n_grid = 10
a1s = np.linspace(0, 1, n_grid)

npoints = 15
result = np.zeros((npoints, npoints))
approx_result = np.zeros((npoints, npoints))
env_noise_levels = np.linspace(0.01, 1, npoints)
neur_noise_levels = np.linspace(0.01, .5, npoints)
mi_curves = np.zeros((npoints, npoints, n_grid))
g_mi_curves = np.zeros((npoints, npoints, n_grid))
total_iterations = npoints **2 

print("starting first heatmap")
pbar = tqdm(total=total_iterations, desc="Overall Progress")
for i, v_e in enumerate(env_noise_levels):
    for j, v_n in  enumerate(neur_noise_levels):
        f = mi_loss_sample_bin(response, v_e * noise_response, v_n* neur_noises, bins = 50, plot = False, print_val = False, conditional_constant=True)
        for k, a in enumerate(a1s):
            A = np.array([[a, 1-a]])
            mi = f(A)
            mi_curves[i, j, k] += mi
            cond_entropy_ext = .5 * np.log(np.linalg.det( v_e**2 *A @ S @ S.T @ A.T+ v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
            resp_entropy_ext = .5 * np.log(np.linalg.det(A @ S @ (Sigma_xx  +  v_e**2 *np.eye(n_odorants))@ S.T @ A.T +  v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
            mi_gauss =  resp_entropy_ext - cond_entropy_ext
            g_mi_curves[i,j,k] += resp_entropy_ext - cond_entropy_ext
        mi_max = np.argmax( mi_curves[i, j, :])
        mi_max_gauss =  np.argmax( g_mi_curves[i, j, :])
        if mi_max > 0 and mi_max < n_grid-1:
            result[i,j] += 1
        if mi_max_gauss > 0 and mi_max_gauss < n_grid-1:
            approx_result[i,j] += 1
        pbar.update()

print("first_heatmap_done")
with open( "non_gauss_output.txt", "a") as f:
    f.write("first_heatmap_done")

with open("../results/tables/non_gauss_info.pkl", "wb") as f:
    pkl.dump(result, f)

cs = axs[1, 2].pcolor(result, vmin = 0, vmax = 1, cmap = 'binary')
axs[1,2].contour(approx_result, levels  = [.5])

print(approx_result)
print("starting second heatmap")

result = np.zeros((npoints, npoints))
approx_result = np.zeros((npoints, npoints))
mi_curves_higher = np.zeros((npoints, npoints, n_grid))
g_mi_curves_higher = np.zeros((npoints, npoints, n_grid))

pbar = tqdm(total=total_iterations, desc="Overall Progress")

for i, v_e in enumerate(env_noise_levels):
    for j, v_n in  enumerate(neur_noise_levels):
        f = mi_loss_sample_bin(response_h, v_e* noise_response, v_n* neur_noises, bins = 50, plot = False, print_val = False, conditional_constant=True)
        for k, a in enumerate(a1s):
            A = np.array([[a, 1-a]]).astype(np.float16)
            mi = f(A)
            mi_curves_higher[i, j, k] += mi
            cond_entropy_ext = .5 * np.log(np.linalg.det( v_e**2 *A @ S @ S.T @ A.T+ v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
            resp_entropy_ext = .5 * np.log(np.linalg.det(A @ S @ (Sigma_xx_h  +  v_e**2 *np.eye(n_odorants))@ S.T @ A.T +  v_n **2 *np.eye(n_neurons))) +  n_neurons * .5 *(1 + np.log(2*np.pi))
            mi_gauss =  resp_entropy_ext - cond_entropy_ext
            g_mi_curves[i,j,k] += resp_entropy_ext - cond_entropy_ext
        mi_max = np.argmax(mi_curves_higher[i, j, :])
        mi_max_gauss =  np.argmax( g_mi_curves[i, j, :])
        if mi_max > 0 and mi_max < n_grid-1:
            result[i,j] += 1
        if mi_max_gauss > 0 and mi_max_gauss < n_grid-1:
            approx_result[i,j] += 1
        pbar.update()

with open("../results/tables/non_gauss_info_higher.pkl", "wb") as f:
    pkl.dump(result, f)


cs = axs[1, 3].pcolor(result, vmin = 0, vmax = 1, cmap = 'binary')
axs[1,3].contour(approx_result, levels = [.5])

print("second heatmap done")
with open( "non_gauss_output.txt", "a") as f:
    f.write("first_heatmap_done")

sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/non_gauss_fig.pdf")

print("first plot saved")
fig, axs = plt.subplots(npoints, npoints)
for i, v_e in enumerate(env_noise_levels):
    for j, v_n in  enumerate(neur_noise_levels):
        axs[i,j].plot(mi_curves[i,j,:])
        axs[i,j].set_title(f"v_e = {v_e:.2f}, v_n = {v_n:.2f}")

sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/non_gauss_curves.pdf")

print("second plot saved")
fig, axs = plt.subplots(npoints, npoints)
for i, v_e in enumerate(env_noise_levels):
    for j, v_n in  enumerate(neur_noise_levels):
        axs[i,j].plot(mi_curves_higher[i,j,:])
        axs[i,j].set_title(f"v_e = {v_e:.2f}, v_n = {v_n:.2f}")

with open("non_gauss_output.txt", "a") as f:
    f.write("everything generated, saving plot")
sns.despine()
plt.tight_layout()
plt.savefig("../results/plots/non_gauss_curves_higher.pdf")

print("done")