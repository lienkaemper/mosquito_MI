## Standard libraries
import sys
#print(sys.version)
import os
import math
import numpy as np
import time
import pandas  as pd 



import torch
#print("Using torch", torch.__version__)
#torch.manual_seed(42) # Setting the seed
torch.device("mps") 
device = 'mps' if torch.cuda.is_available() else 'cpu'



def project_to_simplex(x):
  n = len(x)
  y, ind = torch.sort(x)
  t_last = 0
  for i,j in enumerate(ind):
    t_i = (sum(y[i:n]) -1)/(n-i)
    if y[i] - t_i > 0:
      return(torch.relu(x -t_i ))
    else:
      t_last = t_i
  return x - t_last

def project_to_simplex_mat(A):
  m, n = A.shape
  result = torch.zeros(m,n)
  for i in range(m):
    result[i,:] = project_to_simplex(A[i,:])
  return result

def simplex_sample(m, n):
  A = torch.zeros((m,n))
  for i in range(m):
    nums = torch.zeros(n+1)
    nums[1:n] = torch.rand(n-1)
    nums[n] = 1
    nums,_ = torch.sort(nums)
    A[i, :] = torch.diff(nums)
  return A


  
def projected_grad_asc(f, proj, x0, steps = 100,  rate = 0.001, grad_noise = 0.001, noise_decay = 0.9, grad_tol = .001):
  grads = np.zeros(steps)
  values = np.zeros(steps) 
  x = x0.clone().detach()
  x.requires_grad_()
  x_old = x0.clone().detach()
  xs = np.zeros((steps, torch.numel(x)))
  noise_scale = 1
  #x.to(device)
  #x_old.to(device)
  for i in range(steps):
    x = x.clone().detach()
    x.requires_grad_()
    f_x = f(x)
    f_x.requires_grad_()
    f_x.backward()
    g = x.grad
    values[i] = f_x
    x = x + rate * g +grad_noise*rate* torch.randn_like(g)*noise_scale
    noise_scale *= noise_decay
    x = proj(x)
    diff = x_old - x
    grads[i] = torch.flatten(diff).dot(torch.flatten(diff))
    x_old = x
    xs[i, :]=x.detach().numpy().flatten()
    if torch.sum(torch.abs(diff)) < grad_tol:
      break
  return x, grads, values, xs



def mi_loss(C_xx, m, n, s_w, s_m, S = None):
  """ Compute the mutual inforation loss

   Arguments:
  
  n -- numer of odorants 
  m -- number of neurons 
  s_w -- variance of environmentla noise
  s_m -- variance of neural noise
  S -- sensing matrix (default is identity)

  Computes mutual information loss via the conditional entropy of X given Y H(X | Y). 
  I(X, Y) = H(X) - H(X|Y)
  """
  if S is None : S = torch.eye(n)
  def f(A):
    if np.isscalar(s_w):
      S_xy = C_xx @ (A@ S).T
      S_yy = (A @S)@(C_xx + s_w*torch.eye(n))@(A@ S).T + s_m*torch.eye(m) 
      D = -torch.log(torch.det(torch.eye(n) - (A@S).T @ torch.inverse(S_yy) @S_xy.T)) 
      return D
    else: #assumes s_w is a n by n covariance matrix 
      ss_w = s_w.astype(np.float32)
      S_xy = C_xx @ (A@ S).T
      S_yy = (A@ S)@(C_xx + ss_w)@(A@ S).T + s_m*torch.eye(m) 
      D = -torch.log(torch.det(torch.eye(n) - (A@ S).T @ torch.inverse(S_yy) @S_xy.T)) 
      return D
  return f 

def mi_loss_new(stim_cov, noise_cov, sigma_n, sigma_e):
  """ Compure the mutual inforation loss

  Arguments:

  stim_cov -- receptor by receptor stimulus driven covariance, 
              computed as S @ Sigma @ S.T if Sigma is odorant covariance, S sensing matrix
  noise_cov -- receptor by receptor noise covarinace, 
              computed as S @ S.T
  sigma_n -- scalar, neural noise covariance
  sigma_e -- scalar, environmental noise covariance 
  

  Computes mutual information loss as 

  I_A(X, Y) = 
  """
  def f(A):
    n = A.shape[0]
    D = (1/2)*torch.log(torch.det(torch.eye(n) + A @ stim_cov @ A.T @ torch.inverse(sigma_e * A @ noise_cov @A.T + sigma_n*torch.eye(n)))) 
    return D
  return f 


def mi_loss_np(stim_cov, noise_cov, sigma_n, sigma_e):
  """ Compure the mutual inforation loss

  Arguments:

  stim_cov -- receptor by receptor stimulus driven covariance, 
              computed as S @ Sigma @ S.T if Sigma is odorant covariance, S sensing matrix
  noise_cov -- receptor by receptor noise covarinace, 
              computed as S @ S.T
  sigma_n -- scalar, neural noise covariance
  sigma_e -- scalar, environmental noise covariance 
  

  Computes mutual information loss as 

  I_A(X, Y) = 
  """
  def f(A):
    n = A.shape[0]
    D = (1/2)*np.log(np.linalg.det(np.eye(n) + A @ stim_cov @ A.T @ np.linalg.inv(sigma_e * A @ noise_cov @A.T + sigma_n*np.eye(n)))) 
    return D
  return f 


def mse_loss(S, m, n, s_w, s_m):
  def f(A):
    S_xy = S @ (A.T)
    S_yy = A@(S + s_w*torch.eye(n))@(A.T) + s_m*torch.eye(m) 
    D = torch.trace(S_xy @torch.inverse(S_yy) @ S_xy.T)
    return D
  return f 



def maximize_mi(C_xx ,  s_w=.1,  s_m = .1, m = None, S = None,  steps = 100,rate = 0.1, grad_noise = 0.1,  noise_decay = 0.99):
  n = C_xx.shape[0]
  if m is None:
    m = n
  if S is None:
      S = torch.eye(n)
  r = S.shape[0]
  f = mi_loss(C_xx , m, n, s_w, s_m, S = S)
  proj = project_to_simplex_mat
  x0 = torch.ones((m,r))/r + 0.1*torch.randn(m,r)
  return projected_grad_asc(f, proj, x0, steps, rate, grad_noise, noise_decay)

def maximize_mi_new(stim_cov, noise_cov,  env_noise = .1,  neur_noise = .1, m = None, steps = 100,rate = 0.1, grad_noise = 0.1,  noise_decay = 0.99, grad_tol = .001):
  n = stim_cov.shape[0]
  if m is None:
    m = n
  f = mi_loss_new(stim_cov, noise_cov, neur_noise, env_noise)
  proj = project_to_simplex_mat
  x0 = torch.ones((m,n))/n + 0.1*torch.randn(m,n)
  x0 = project_to_simplex_mat(x0)
  return projected_grad_asc(f, proj, x0, steps, rate, grad_noise, noise_decay, grad_tol = grad_tol)

def minimize_mse(S,  s_w=.1,  s_m = .1, m = None,  steps = 100,rate = 0.1, grad_noise = 0.1,  noise_decay = 0.99, grad_tol = .001):
  n = S.shape[0]
  if m is None:
    m = n
  f = mse_loss(S, m, n, s_w, s_m)
  proj = project_to_simplex_mat
  x0 = torch.ones((m,n))/n + 0.001*torch.randn(m,n)
  return projected_grad_asc(f, proj, x0, steps, rate, grad_noise, noise_decay, grad_tol = grad_tol)




