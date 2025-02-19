## Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def gauss_info(C_xx, A, w, m):
    r = C_xx.shape[0]
    n = A.shape[0]
    C_xy =   C_xx @ A.T
    C_yy = m*np.eye(n) + A@(C_xx + w*np.eye(r))@A.T
    C = np.concatenate((np.concatenate((C_xx, C_xy), axis=1), np.concatenate((C_xy.T, C_yy), axis=1)), axis=0)
    return (1/2)*(np.log(np.linalg.det(C_xx)) +  np.log(np.linalg.det(C_yy)) - np.log(np.linalg.det(C)))
    
    

def gauss_info_sensing(C_xx, A, w, m, S):
    r = C_xx.shape[0]
    n = A.shape[0]
    C_xz=   C_xx @ (A @ S).T
    C_zz = m*np.eye(n) + A@S @(C_xx + w*np.eye(r))@(A @ S).T
    C = np.concatenate((np.concatenate((C_xx, C_xz), axis=1), np.concatenate((C_xz.T, C_zz), axis=1)), axis=0)
    return (1/2)*(np.log(np.linalg.det(C_xx)) +  np.log(np.linalg.det(C_zz)) - np.log(np.linalg.det(C)))
    
def calculate_roots(sigma1, sigma2, rho, e, n):
    AA = e * (sigma2**2 - sigma1**2)
    BB = e * (sigma1**2 - 2 * rho * sigma1 * sigma2 - sigma2**2) + n * (sigma1**2 - 2 * rho * sigma1 * sigma2 +sigma2**2)
    CC = e*sigma1*sigma2*rho + n * rho *sigma1 * sigma2 -n * sigma2**2

    discriminant = np.sqrt(BB**2 - 4 * AA * CC)

    root1 = (-BB + discriminant) / (2 * AA)
    root2 = (-BB - discriminant) / (2 * AA)

    return root1, root2

def opt_coexp(sigma1, sigma2, rho, e, n):
    if sigma2 > sigma1:
        opt_a1 =  max(calculate_roots(sigma1, sigma2, rho, e, n)[1], 0)
        opt_a2 = 1 - opt_a1
        return np.array([[opt_a1, opt_a2]])
    else:
        opt_a2 =  max(calculate_roots(sigma2, sigma1, rho, e, n)[1], 0)
        opt_a1 = 1 - opt_a2
        return np.array([[opt_a1, opt_a2]])

def get_odorant_cor(overlap, stim_cor):
    c = 2 * (overlap**2-overlap)
    return (stim_cor + c*stim_cor +c)/(1 + c*stim_cor + c)

def get_odorant_var(overlap, stim_cor):
    c = 2 * (overlap**2-overlap)
    return 1/(1 + c*(1-stim_cor))

def get_receptor_cor(overlap, odorant_cor):
    c  = overlap
    r = odorant_cor
    S = np.array([[1-c, c],[c, 1-c]] )
    C = np.array([[1, r], [r, 1]])
    C_R = S @ C @ S
    return C_R[0,1]/C_R[0,0]

def coexp_ratio(sig_cov, noise_cov, i, j):
    rat_i = (noise_cov[i,i]*sig_cov[i,j] - noise_cov[i,j]*sig_cov[i,i])/(sig_cov[i,i] - sig_cov[i,j])
    rat_i = max(0, rat_i)
    rat_j = (noise_cov[j,j]*sig_cov[i,j] - noise_cov[i,j]*sig_cov[j,j])/(sig_cov[j,j] - sig_cov[i,j])
    rat_j = max(0,rat_j)
    return min(rat_i, rat_j)


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