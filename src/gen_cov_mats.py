## Standard libraries
import numpy as np
np.random.seed(1)


def rand_powerlaw(n,exp):
    S = np.random.rand(n,n)
    S = S @ S.T
    l, v = np.linalg.eigh(S)
    l = (2*np.pi*n)**(1/(2*n)) *(n/np.e)*((np.arange(n) + 1.0)**(-exp))
    S = v @ np.diag(l) @ v.T
    return S

def block_cov_mat(n,block_size, diag, block, ext):
    A = np.ones((n,n))*ext
    n_blocks = n//block_size
    for i in range(n_blocks):
        A[i*block_size:(i+1)*block_size,i*block_size:(i+1)*block_size] = block
    for i in range(n):
        A[i,i] = diag
    return A

def rand_dot_mat(n, d):
    U = np.random.randn(n,d)
    C = U @ U.T
    return C
    
       
def cov_to_cor(mat):
    d = np.copy(np.diag(mat))
    d[d== 0] = 1
    return (1/np.sqrt(d)) *mat * (1/(np.sqrt(d)))[...,None]


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
            noisy_sources_signal = noisy_sources_signal * (noisy_sources_signal > 0)
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

        return X_signal.astype(np.float64), X_noise.astype(np.float64), Y.astype(np.float64)
    return sample, sources_signal, sources_noise

def sparse_latent_cov(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    sampler, _, _ = sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise, noise_strength)
    X_signal, _, _= sampler(2000)
    signal_cov = np.cov(X_signal, rowvar = False)
    return signal_cov