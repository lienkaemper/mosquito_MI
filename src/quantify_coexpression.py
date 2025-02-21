import numpy as np 


    
def coexp_level(A):
    """
    Calculate the coexpression level of a given matrix.
    The coexpression level is computed as the mean of the ratio of the sum of squares of each row 
    to the square of the sum of each row, minus 1.
    Parameters:
    A (numpy.ndarray): A 2D numpy array where each row represents a set of expression levels.
    Returns:
    float: The coexpression level of the input matrix.
    """

    denoms = np.sum(A**2, axis =1)
    nums =  np.sum(A, axis =1)**2
    return np.mean(nums/denoms)-1
    
def coexd_receptors(A, thresh = .001):
    """
    Determine co-expressed and non-co-expressed receptors based on a threshold.
    Parameters:
    A (numpy.ndarray): A 2D array where rows represent neurons and columns represent receptors.
    thresh (float, optional): Threshold value to determine co-expression. Default is 0.001.
    Returns:
    tuple: A tuple containing two 2D arrays:
        - C (numpy.ndarray): A binary matrix indicating co-expressed receptors (1 if co-expressed, 0 otherwise).
        - N (numpy.ndarray): A binary matrix indicating non-co-expressed receptors (1 if not co-expressed, 0 otherwise).
    """
    
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





def bootstrap_confidence_interval(coexp, not_coexp, n_bootstrap=10000, confidence=0.95):
    """
    Calculate the bootstrap confidence interval for the difference of means between two samples.
    Parameters:
    coexp (array-like): Sample data for the coexpressed group.
    not_coexp (array-like): Sample data for the non-coexpressed group.
    n_bootstrap (int, optional): Number of bootstrap resamples to perform. Default is 10000.
    confidence (float, optional): Confidence level for the interval. Default is 0.95.
    Returns:
    tuple: A tuple containing the observed difference of means and the confidence interval as (observed_diff, (lower_bound, upper_bound)).
    """

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