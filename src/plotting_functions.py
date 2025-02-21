import matplotlib.pyplot as plt 
import numpy as np 


def abline(slope, intercept, axes, color = 'black'):
    """Plot a line from slope and intercept"""
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, color = color)



def gaussian(x, sigma, mu):
    """
    Compute the value of a Gaussian (normal) distribution function.
    Parameters:
    x (float or array-like): The input value(s) at which to evaluate the Gaussian function.
    sigma (float): The standard deviation of the Gaussian distribution.
    mu (float): The mean (center) of the Gaussian distribution.
    Returns:
    float or array-like: The value of the Gaussian function at the input value(s) x.
    """

    return (sigma * np.sqrt(2 * np.pi))**(-1) * np.exp(-(1/2) * ((x - mu)/sigma)**2)

def gaussian_2d(x, Sigma, mu):
    """
    Compute the value of a 2D Gaussian function.
    Parameters:
    x (numpy.ndarray): A 1D array representing the point at which to evaluate the Gaussian function.
    Sigma (numpy.ndarray): A 2D array representing the covariance matrix of the Gaussian distribution.
    mu (numpy.ndarray): A 1D array representing the mean vector of the Gaussian distribution.
    Returns:
    float: The value of the Gaussian function at point x.
    """

    d = Sigma.shape[0]
    return (2 * np.pi)**(-d/2) * np.linalg.det(Sigma)**(-1/2) * np.exp(-(1/2) * (x - mu).T @np.linalg.inv(Sigma) @ (x-mu) )