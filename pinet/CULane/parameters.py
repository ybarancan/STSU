#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np

class Parameters():
    n_epoch = 1000
    l_rate = 0.001
    weight_decay=1e-5
    save_path = "savefile/"
    model_path = "savefile/"
    batch_size = 12
    x_size = 512
    y_size = 256
    resize_ratio = 8
    grid_x = x_size//resize_ratio  #64
    grid_y = y_size//resize_ratio  #32
    feature_size = 4
    regression_size = 110
    mode = 3
    threshold_point = 0.96 #0.88 #0.93 #0.95 #0.93
    threshold_instance = 0.08

    #loss function parameter
    K1 = 1.0                     #  ####################################
    K2 = 2.0
    constant_offset = 0.2
    constant_exist = 2.5    #2
    constant_nonexist = 1.0
    constant_angle = 1.0
    constant_similarity = 1.0
    constant_attention = 0.01
    constant_alpha = 0.5 #in SGPN paper, they increase this factor by 2 every 5 epochs
    constant_beta = 0.5
    constant_l = 1.0
    constant_lane_loss = 1.0  #10  ######################################
    constant_instance_loss = 1.0

    #data loader parameter
    flip_ratio=0.6
    translation_ratio=0.6
    rotate_ratio=0.6
    noise_ratio=0.6
    intensity_ratio=0.6
    shadow_ratio=0.6
    
    train_root_url="/home/kym/research/autonomous_car_vision/dataset/CULane_dataset/train_set/"
    test_root_url="/home/kym/research/autonomous_car_vision/dataset/CULane_dataset/test_set/"

    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((grid_y, grid_x, 2))
    for y in range(grid_y):
        for x in range(grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
    num_iter = 30
    threshold_RANSAC = 0.1
    ratio_inliers = 0.1
