#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:07:33 2021

@author: cany
"""

from src.data.nuscenes import utils as nusc_utils

import numpy as np


LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']

intrinsic_dict = dict()

intrinsic_dict['boston-seaport'] = np.array([[1.25281310e+03, 0.00000000e+00, 8.26588115e+02],
       [0.00000000e+00, 1.25281310e+03, 4.69984663e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

intrinsic_dict['singapore-onenorth'] = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
       [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

intrinsic_dict['singapore-queenstown'] = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
       [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

intrinsic_dict['singapore-hollandvillage'] = np.array([[1.26641720e+03, 0.00000000e+00, 8.16267020e+02],
       [0.00000000e+00, 1.26641720e+03, 4.91507066e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

camera_matrix_dict = dict()


camera_matrix_dict['boston-seaport'] = np.array([[ 0.01026021,  0.00843345,  0.9999118 ,  1.72200568],
       [-0.99987258,  0.01231626,  0.01015593,  0.00475453],
       [-0.01222952, -0.99988859,  0.00855874,  1.49491292],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


camera_matrix_dict['singapore-onenorth'] = np.array([[ 5.68477868e-03, -5.63666773e-03,  9.99967955e-01, 1.70079119e+00],
       [-9.99983517e-01, -8.37115272e-04,  5.68014846e-03,
         1.59456324e-02],
       [ 8.05071338e-04, -9.99983763e-01, -5.64133364e-03,
         1.51095764e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

camera_matrix_dict['singapore-queenstown'] = np.array([[ 5.68477868e-03, -5.63666773e-03,  9.99967955e-01,
         1.70079119e+00],
       [-9.99983517e-01, -8.37115272e-04,  5.68014846e-03,
         1.59456324e-02],
       [ 8.05071338e-04, -9.99983763e-01, -5.64133364e-03,
         1.51095764e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])


camera_matrix_dict['singapore-hollandvillage'] = np.array([[ 5.68477868e-03, -5.63666773e-03,  9.99967955e-01,
         1.70079119e+00],
       [-9.99983517e-01, -8.37115272e-04,  5.68014846e-03,
         1.59456324e-02],
       [ 8.05071338e-04, -9.99983763e-01, -5.64133364e-03,
         1.51095764e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

augment_steps=[0.5,1,1.5,2]
my_dict = dict()

for loc in intrinsic_dict.item().keys():
    my_list = []
    for k in augment_steps:

        write_row, write_col, total_mask = nusc_utils.zoom_augment_grids((900,1600,3),intrinsic_dict.item().get(loc),
                                                                                 camera_matrix_dict.item().get(loc)[:3,-1], k)
        
        
        temp = np.stack([write_row.flatten(),write_col.flatten(),total_mask.flatten()],axis=-1)
        my_list.append(np.copy(temp))
        
    my_dict[loc] = np.copy(np.stack(my_list,axis=0))



np.save('zoom_sampling_dict.npy', my_dict)