## Standard libraries
import numpy as np
import matplotlib.pyplot as plt



def gauss_info_sensing(C_xx, A, w, m, S):
    """
    Computes the mutual information between olfactory signal and neural response.
    Args:
        C_xx (numpy.ndarray): Covariance matrix of the olfactory signal distribution.
        A (numpy.ndarray): Receptor expression matrix.
        w (float): Level of environmental noise.
        m (float): Level of neural noise.
        S (numpy.ndarray): Receptor affinity matrix.
    Returns:
        float: The mutual information between the olfactory signal and neural response.
    """

    r = C_xx.shape[0]
    n = A.shape[0]
    C_xz=   C_xx @ (A @ S).T
    C_zz = m*np.eye(n) + A@S @(C_xx + w*np.eye(r))@(A @ S).T
    C = np.concatenate((np.concatenate((C_xx, C_xz), axis=1), np.concatenate((C_xz.T, C_zz), axis=1)), axis=0)
    return (1/2)*(np.log(np.linalg.det(C_xx)) +  np.log(np.linalg.det(C_zz)) - np.log(np.linalg.det(C)))
    

def calculate_roots(sigma1, sigma2, rho, e, n):
    """
    Calculate the roots of the derivative of mutual information with respect to receptor 1 expression.
    Args:
        sigma1 (float): Variance of receptor 1.
        sigma2 (float): Variance of receptor 2.
        rho (float): Correlation between receptor 1 and receptor 2.
        e (float): Level of environmental noise.
        n (float): Level of neural noise.
    Returns:
        tuple: A tuple containing root1 and root2, which are the zeros of the derivative of mutual information with respect to receptor 1 expression.
    """

    AA = e * (sigma2**2 - sigma1**2)
    BB = e * (sigma1**2 - 2 * rho * sigma1 * sigma2 - sigma2**2) + n * (sigma1**2 - 2 * rho * sigma1 * sigma2 +sigma2**2)
    CC = e*sigma1*sigma2*rho + n * rho *sigma1 * sigma2 -n * sigma2**2

    discriminant = np.sqrt(BB**2 - 4 * AA * CC)

    root1 = (-BB + discriminant) / (2 * AA)
    root2 = (-BB - discriminant) / (2 * AA)

    return root1, root2



def coexp_slope(sigma1, sigma2, rho):
    """
    Calculate the slope of the line which divides values of neural noise and environmental noise 
    where co-expression is optimal from values where single expression is optimal.

    Parameters:
    sigma1 (float): Variance of odorant 1 (sigma2 > sigma1).
    sigma2 (float): Variance of odorant 2.
    rho (float): Correlation between odorants.

    Returns:
    float: Slope of the line.
    """
    if sigma2 < sigma1:
        raise ValueError("sigma2 must be greater than or equal to sigma1")
    return (sigma2/sigma1-  rho)/ rho

def get_odorant_cor(overlap, stim_cor):
    """
    Calculate the correlation between two odorants over the olfactory signal.
    Parameters:
    overlap (float): Tuning overlap between the two receptors.
    stim_cor (float): Correlation of the two receptors' activations over the olfactory signal.
    Returns:
    float: Correlation between the two odorants over the olfactory signal.
    """

    c = 2 * (overlap**2-overlap)
    return (stim_cor + c*stim_cor +c)/(1 + c*stim_cor + c)

def get_odorant_var(overlap, stim_cor):
    """
    Computes the odorant variance needed to ensure that the receptor variance is 1.
    Parameters:
    overlap (float): The overlap value.
    stim_cor (float): The stimulus correlation.
    Returns:
    float: The computed odorant variance.
    """

    c = 2 * (overlap**2-overlap)
    return 1/(1 + c*(1-stim_cor))

def get_receptor_cor(overlap, odorant_cor):
    """
    Computes the correlation between receptors given tuning overlap and odorant correlation.
    Parameters:
    overlap (float): The tuning overlap between receptors.
    odorant_cor (float): The correlation between odorants.
    Returns:
    float: The computed correlation between receptors.
    """

    c  = overlap
    r = odorant_cor
    S = np.array([[1-c, c],[c, 1-c]] )
    C = np.array([[1, r], [r, 1]])
    C_R = S @ C @ S
    return C_R[0,1]/C_R[0,0]

def coexp_ratio(sig_cov, noise_cov, i, j):
    """
    Computes the co-expression ratio between receptors i and j.
    This function calculates the co-expression ratio, which is defined as the ratio of environmental noise variance to neural noise variance where co-expression becomes optimal
    when restricted to only one neuron which can express only receptors i and j. See equation 34.
    Parameters:
    sig_cov (numpy.ndarray): Signal covariance matrix.
    noise_cov (numpy.ndarray): Noise covariance matrix.
    i (int): Index of the first receptor.
    j (int): Index of the second receptor.
    Returns:
    float: The co-expression ratio between receptors i and j.
    """

    rat_i = (noise_cov[i,i]*sig_cov[i,j] - noise_cov[i,j]*sig_cov[i,i])/(sig_cov[i,i] - sig_cov[i,j])
    rat_i = max(0, rat_i)
    rat_j = (noise_cov[j,j]*sig_cov[i,j] - noise_cov[i,j]*sig_cov[j,j])/(sig_cov[j,j] - sig_cov[i,j])
    rat_j = max(0,rat_j)
    return min(rat_i, rat_j)


def bintropy(X, bins =100, plot = False):
    """
    Calculate the binned entropy of a given dataset.
    Parameters:
    X (array-like): Input data for which the entropy is to be calculated.
    bins (int, optional): Number of bins to use for the histogram. Default is 100.
    plot (bool, optional): If True, plot the histogram with the calculated entropy. Default is False.
    Returns:
    float: The calculated binned entropy of the input data.
    """
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
    """
    Estimate the mutual information between input data and output responses using binning entropy estimation.
    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix of shape (m_samples, n_in).
    env_noises : numpy.ndarray
        Environmental noise matrix of shape (n_e, n_in).
    neur_noises : numpy.ndarray
        Neuronal noise matrix of shape (n_n, n_out).
    f : function
        Function that computes the output responses given input data and noise combinations.
    bins : int, optional
        Number of bins to use for entropy estimation (default is 100).
    plot : bool, optional
        If True, plot histograms (default is False).
    conditional_constant : bool, optional
        If True, compute conditional entropy with a constant value (default is False).
    Returns:
    --------
    tuple
        A tuple containing:
        - mutual_info : float
            Estimated mutual information.
        - response_entropy : float
            Estimated response entropy.
        - conditional_entropy : float
            Estimated conditional entropy.
    """

    m_samples, _ = X.shape
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
                    Hs += bintropy(Y, bins)
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




def mi_loss_sample_bin(signal_sample, env_noise_sample, neur_noise_sample, S, bins = 100, plot = False, print_val = False, conditional_constant = False):
    """
    Computes the mutual information loss for a given sample using binning method.
    Parameters:
    signal_sample (array-like): The sample of the signal.
    env_noise_sample (array-like): The sample of the environmental noise.
    neur_noise_sample (array-like): The sample of the neuronal noise.
    S (array-like): The transformation matrix.
    bins (int, optional): The number of bins to use for entropy estimation. Default is 100.
    plot (bool, optional): If True, plot the results. Default is False.
    print_val (bool, optional): If True, print the mutual information, response entropy, and conditional entropy values. Default is False.
    conditional_constant (bool, optional): If True, use a conditional constant in the entropy estimation. Default is False.
    Returns:
    function: A loss function that computes the mutual information loss given a transformation matrix A.
    The returned loss function takes the following parameters:
    A (array-like): The transformation matrix.
    The loss function computes the mutual information loss by:
    1. Performing element-wise addition of the input and environmental noise.
    2. Performing matrix multiplication with the transformation matrices.
    3. Adding the neuronal noise element-wise.
    4. Estimating the entropy using the binning method.
    The loss function returns the mutual information loss value.
    """

    def loss_fn(A):
        def f(X_broadcast, env_combs, neur_combs):
            # X_broadcast: (n_e * n_n, n_in), batch of x inputs
            # env_combs: (n_e * n_n, n_in), batch of xi values
            # neur_combs: (n_e * n_n, n_out), batch of zeta values
            
            # Step 1: Perform the element-wise addition (x + xi)
            Y = X_broadcast + env_combs  # Broadcasting works automatically here
            
            # Step 2: Perform the matrix multiplications (A @ S @ (x + xi))
            result = A @  S @ Y.T  # Transpose to align shapes for matmul (n_in, n_e * n_n)
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