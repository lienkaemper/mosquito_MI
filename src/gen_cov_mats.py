## Standard libraries
import numpy as np
np.random.seed(1)


    
def cov_to_cor(mat):
    """
    Convert a covariance matrix to a correlation matrix.
    Parameters:
    mat (numpy.ndarray): A square covariance matrix.
    Returns:
    numpy.ndarray: A correlation matrix derived from the input covariance matrix.
    """
    
    d = np.copy(np.diag(mat))
    d[d== 0] = 1
    return (1/np.sqrt(d)) *mat * (1/(np.sqrt(d)))[...,None]


# Defines distribution 
def sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise = .1, noise_strength = 1):
    """
    Generates a function to sample sparse odor concentration vectors  generated from a mixture of odor sources, 
    each of which has a characteristic odor blend that is sparse relative to the total number of odorants
    Parameters:
    p_source (float): Probability of a source being active.
    p_sample (float): Probability of a sample being taken.
    n_source_signal (int): Number of signal sources.
    n_source_noise (int): Number of noise sources.
    n_odorants (int): Number of odorants.
    concentration_noise (float, optional): Noise level added to the concentration of sources. Default is 0.1.
    noise_strength (float, optional): Strength of the noise. Default is 1.
    Returns:
    function: A function that generates samples of signal and noise.
    np.ndarray: Matrix of signal sources.
    np.ndarray: Matrix of noise sources.
    Sample function parameters:
    n_samples (int, optional): Number of signal samples to generate. Default is 1.
    n_samples_noise (int, optional): Number of noise samples to generate. Default is 1.
    Sample function returns:
    np.ndarray: Matrix of signal samples.
    np.ndarray: Matrix of noise samples.
    np.ndarray: Matrix of coefficients for signal sources.
    """

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
    """
    Generates a covariance matrix for a sparse odor concentration model.
    Parameters:
    p_source (float): Probability of a source being active.
    p_sample (float): Probability of a sample being taken.
    n_source_signal (int): Number of signal sources.
    n_source_noise (int): Number of noise sources.
    n_odorants (int): Number of odorants.
    concentration_noise (float, optional): Noise level added to the concentration of sources. Default is 0.1.
    noise_strength (float, optional): Strength of the noise. Default is 1.
    Returns:
    numpy.ndarray: Covariance matrix of the signal component.
    """

    sampler, _, _ = sparse_latent(p_source, p_sample, n_source_signal, n_source_noise, n_odorants, concentration_noise, noise_strength)
    X_signal, _, _= sampler(2000)
    signal_cov = np.cov(X_signal, rowvar = False)
    return signal_cov