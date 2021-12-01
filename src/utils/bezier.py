



import os
import glob

import numpy as np
import scipy.interpolate as si 
import torch
# from scipy.interpolate import UnivariateSpline
import logging
# import pwlf
from math import factorial

def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def fit_bezier(points, n_control):
    n_points = len(points)
    A = np.zeros((n_points,n_control))
    
    t = np.arange(n_points)/(n_points-1)
    
    for i in range(n_points):
        for j in range(n_control):
            A[i,j] = comb(n_control - 1, j)*np.power(1-t[i],n_control - 1 - j)*np.power(t[i],j)
            
    conts = np.linalg.lstsq(A,points,rcond=None)
    return conts
def interpolate_bezier(conts, n_int=100):    
    n_control = len(conts)
    A = np.zeros((n_int,n_control))
    
    t = np.arange(n_int)/(n_int-1)
    
    for i in range(n_int):
        for j in range(n_control):
            A[i,j] = comb(n_control - 1, j)*np.power(1-t[i],n_control - 1 - j)*np.power(t[i],j)
    
    res = np.dot(A,conts)
    return res


def bezier_matrix(n_control=5,n_int=100):    
 
    A = np.zeros((n_int,n_control))
    
    t = np.arange(n_int)/(n_int-1)
    
    for i in range(n_int):
        for j in range(n_control):
            A[i,j] = comb(n_control - 1, j)*np.power(1-t[i],n_control - 1 - j)*np.power(t[i],j)
    
    
    A = torch.Tensor(A)
  
    A = torch.unsqueeze(A,dim=0)
    return A
#
#def gaussian_line_from_traj(points,size=(196,200)):
#    
#    var = 0.05
#    
#    
#    my_x = torch.linspace(0,1,size[1])
#    my_y = torch.linspace(0,1,size[0])
#    
#    
##    grid_y, grid_x = torch.meshgrid(my_y, my_x)
#    
#    grid_x = torch.unsqueeze(grid_x,dim=0).cuda()
#    grid_y = torch.unsqueeze(grid_y,dim=0).cuda()
#    
#    x_est = torch.unsqueeze(torch.unsqueeze(points[0],dim=-1),dim=-1)
#    y_est = torch.unsqueeze(torch.unsqueeze(points[1],dim=-1),dim=-1)
#    
#    gauss = torch.exp(-(torch.square(x_est - grid_x) + torch.square(y_est - grid_y))/var)
#    
#    
#    return gauss.sum(0)

def gaussian_line_from_traj(points,size=(196,200)):
    
    var = 0.01
    
    
    my_x = torch.linspace(0,1,size[1])
    my_y = torch.linspace(0,1,size[0])

    grid_x = torch.unsqueeze(torch.unsqueeze(my_x,dim=0),dim=0).cuda()
    grid_y = torch.unsqueeze(torch.unsqueeze(my_y,dim=0),dim=0).cuda()
    
    x_est = points[:,:,0:1]
    y_est = points[:,:,1:]
    
    x_part = torch.exp(-(torch.square(x_est - grid_x))/var)
    y_part = torch.exp(-(torch.square(y_est - grid_y))/var)
    
    gauss = torch.matmul(torch.transpose(y_part,1,2),x_part)
    
    
#    gauss = torch.exp(-(torch.square(x_est - grid_x) + torch.square(y_est - grid_y))/var)
    
    
    return torch.clamp(gauss,0,1)


def interpolate_bezier_torch(conts, n_int=100):    
    n_control = len(conts)
    A = np.zeros((n_int,n_control))
    
    t = np.arange(n_int)/(n_int-1)
    
    for i in range(n_int):
        for j in range(n_control):
            A[i,j] = comb(n_control - 1, j)*np.power(1-t[i],n_control - 1 - j)*np.power(t[i],j)
    
    
    A = torch.Tensor(A).cuda()
  
    A = torch.unsqueeze(A,dim=0)
    A = A.expand(conts.size(0),-1,-1)
    
    
    
    
    res = torch.dot(A,conts)
    return res
