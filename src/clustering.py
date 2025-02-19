import numpy as np
from scipy.cluster.hierarchy import  linkage, leaves_list, optimal_leaf_ordering


def order_cor_matrix(C):
    n = C.shape[0]
    D = np.ones((n,n)) - C
    Z = linkage(D[np.triu_indices(n,1)])
    optimal_leaf_ordering(Z, D[np.triu_indices(n,1)])
    ll = leaves_list(Z)
    return C[np.ix_(ll, ll)], ll, Z