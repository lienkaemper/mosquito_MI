import numpy as np
from scipy.cluster.hierarchy import  linkage, leaves_list, optimal_leaf_ordering


def order_cor_matrix(C):
    """
    Reorders the correlation matrix to group similar items together.

    This function takes a correlation matrix `C` and reorders it using hierarchical clustering
    to group similar items together. The reordered matrix, the order of the leaves, and the 
    hierarchical clustering linkage matrix are returned.

    Parameters:
    C (numpy.ndarray): The correlation matrix to be reordered.

    Returns:
    tuple: A tuple containing:
        - reordered_C (numpy.ndarray): The reordered correlation matrix.
        - ll (numpy.ndarray): The order of the leaves after clustering.
        - Z (numpy.ndarray): The hierarchical clustering linkage matrix.
    """
    n = C.shape[0]
    D = np.ones((n,n)) - C
    Z = linkage(D[np.triu_indices(n,1)])
    optimal_leaf_ordering(Z, D[np.triu_indices(n,1)])
    ll = leaves_list(Z)
    return C[np.ix_(ll, ll)], ll, Z
