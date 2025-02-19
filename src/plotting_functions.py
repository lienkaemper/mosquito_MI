import matplotlib.pyplot as plt 
import numpy as np 


def abline(slope, intercept, axes, color = 'black'):
    """Plot a line from slope and intercept"""
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, color = color)



def gaussian(x, sigma, mu):
    return (sigma * np.sqrt(2 * np.pi))**(-1) * np.exp(-(1/2) * ((x - mu)/sigma)**2)

def gaussian_2d(x, Sigma, mu):
    d = Sigma.shape[0]
    return (2 * np.pi)**(-d/2) * np.linalg.det(Sigma)**(-1/2) * np.exp(-(1/2) * (x - mu).T @np.linalg.inv(Sigma) @ (x-mu) )