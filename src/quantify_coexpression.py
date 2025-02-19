import torch 
import numpy as np 

# def coexp_level(A):
#     maxs = sum(np.max(A, 1))
#     total = np.sum(A)
#     return total/maxs - 1
    
def coexp_level(A):
    denoms = np.sum(A**2, axis =1)
    nums =  np.sum(A, axis =1)**2
    return np.mean(nums/denoms)-1
    
def coexd_receptors(A, thresh = .001):
    n, r = A.shape
    C = np.zeros((r,r))
    N = np.zeros((r,r))
    for r1 in range(r):
        for r2 in range(r1+1, r):
            found = False
            for neur in range(n):
                if min(A[neur, r1], A[neur, r2]) > thresh:
                    C[r1, r2] = 1
                    found = True
            if not found:
                N[r1, r2] = 1
    return C, N

def resp_gain(A, S):
    A_eff = A@S
    A_eff @ A_eff.T

def weighted_alignment(A, S, C_xx):
    A_eff = A@S
    U, E, V = np.svd(A_eff)
    return V.T @ C_xx @ V


def simplex_sample(k, n):
    result = np.zeros((n, k+1))
    for i in range(n):
        x = np.zeros(k+2)
        x[1:-1] += np.random.rand(k)
        x[-1] += 1
        result[i, :] = np.diff(np.sort(x))
    return result

def bootstrap_confidence_interval(coexp, not_coexp, n_bootstrap=10000, confidence=0.95):
    # Compute observed difference of means
    observed_diff = np.mean(coexp) - np.mean(not_coexp)

    # Bootstrap resampling
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_coexp = np.random.choice(coexp, size=len(coexp), replace=True)
        boot_not_coexp = np.random.choice(not_coexp, size=len(not_coexp), replace=True)
        boot_diffs.append(np.mean(boot_coexp) - np.mean(boot_not_coexp))
    
    # Compute confidence interval percentiles
    lower_bound = np.percentile(boot_diffs, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(boot_diffs, (1 + confidence) / 2 * 100)

    return observed_diff, (lower_bound, upper_bound)