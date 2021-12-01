#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from util_hourglass import *
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from hourglass_network import lane_detection_network
from torch.autograd import Function as F
from parameters import Parameters
import math
import util
import hard_sampling

############################################################
##
## agent for lane detection
##
############################################################
class Agent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(Agent, self).__init__()

        self.p = Parameters()

        self.lane_detection_network = lane_detection_network()

        self.setup_optimizer()

        self.current_epoch = 0

        self.hard_sampling = hard_sampling.hard_sampling()

        print("model parameters: ")
        print(self.count_parameters(self.lane_detection_network))

    def count_parameters(self, model):
	    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.lane_detection_optim = torch.optim.Adam(self.lane_detection_network.parameters(),
                                                    lr=self.p.l_rate,
                                                    weight_decay=self.p.weight_decay)

    #####################################################
    ## Make ground truth for key point estimation
    #####################################################
    def make_ground_truth_point(self, target_lanes, target_h):

        target_lanes, target_h = util.sort_batch_along_y(target_lanes, target_h)

        ground = np.zeros((len(target_lanes), 3, self.p.grid_y, self.p.grid_x))
        ground_binary = np.zeros((len(target_lanes), 1, self.p.grid_y, self.p.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            for lane_index, lane in enumerate(batch):
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.p.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.p.resize_ratio)
                        ground[batch_index][0][y_index][x_index] = 1.0
                        ground[batch_index][1][y_index][x_index]= (point*1.0/self.p.resize_ratio) - x_index
                        ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.p.resize_ratio) - y_index
                        ground_binary[batch_index][0][y_index][x_index] = 1

        return ground, ground_binary


    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, target_lanes, target_h):

        ground = np.zeros((len(target_lanes), 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            temp = np.zeros((1, self.p.grid_y, self.p.grid_x))
            lane_cluster = 1
            for lane_index, lane in enumerate(batch):
                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.p.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.p.resize_ratio)
                        temp[0][y_index][x_index] = lane_cluster
                    if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while False:      ###############################################false
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x+1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x-1] = lane_cluster
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y+1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y-1][temp_x] = lane_cluster
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                lane_cluster += 1

            for i in range(self.p.grid_y*self.p.grid_x): #make gt
                temp = temp[temp>-1]
                gt_one = deepcopy(temp)
                if temp[i]>0:
                    gt_one[temp==temp[i]] = 1   #same instance
                    if temp[i] == 0:
                        gt_one[temp!=temp[i]] = 3 #different instance, different class
                    else:
                        gt_one[temp!=temp[i]] = 2 #different instance, same class
                        gt_one[temp==0] = 3 #different instance, different class
                    ground[batch_index][0][i] += gt_one

        return ground

    #####################################################
    ## train
    #####################################################
    def train(self, inputs, target_lanes, target_h, epoch, agent, data_list):
        point_loss = self.train_point(inputs, target_lanes, target_h, epoch, data_list)
        return point_loss

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, inputs, target_lanes, target_h, epoch, data_list):
        real_batch_size = len(target_lanes)

        #generate ground truth
        ground_truth_point, ground_binary = self.make_ground_truth_point(target_lanes, target_h)
        ground_truth_instance = self.make_ground_truth_instance(target_lanes, target_h)
        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], 0, inputs[0])

        # convert numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad=False

        ground_binary = torch.LongTensor(ground_binary.tolist()).cuda()
        ground_binary.requires_grad=False

        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
        ground_truth_instance = Variable(ground_truth_instance).cuda()
        ground_truth_instance.requires_grad=False

        #util.visualize_gt(ground_truth_point[0], ground_truth_instance[0], inputs[0])

        # update lane_detection_network
        result, attentions = self.predict_lanes(inputs)
        lane_detection_loss = 0
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        x_offset_loss = 0
        y_offset_loss = 0
        sisc_loss = 0
        disc_loss = 0
        
        # hard sampling ##################################################################
        confidance, offset, feature = result[-1]
        hard_loss = 0

        for i in range(real_batch_size):
            confidance_gt = ground_truth_point[i, 0, :, :]
            confidance_gt = confidance_gt.view(1, self.p.grid_y, self.p.grid_x)
            hard_loss =  hard_loss +\
                torch.sum( (1-confidance[i][confidance_gt==1])**2 )/\
                (torch.sum(confidance_gt==1)+1)

            target = confidance[i][confidance_gt==0]
            hard_loss =  hard_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            node = hard_sampling.sampling_node(loss = hard_loss.cpu().data, data = data_list[i], previous_node = None, next_node = None)
            self.hard_sampling.insert(node)

        # compute loss for point prediction #############################################
        for (confidance, offset, feature) in result:
            #compute loss for point prediction

            #exist confidance loss##########################
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.p.grid_y, self.p.grid_x)
            a = confidance_gt[0][confidance_gt[0]==1] - confidance[0][confidance_gt[0]==1]
            exist_condidence_loss =  exist_condidence_loss +\
				torch.sum( (1-confidance[confidance_gt==1])**2 )/\
				(torch.sum(confidance_gt==1)+1)

            #non exist confidance loss##########################
            target = confidance[confidance_gt==0]
            nonexist_confidence_loss =  nonexist_confidence_loss +\
				torch.sum( ( target[target>0.01] )**2 )/\
				(torch.sum(target>0.01)+1)

            #offset loss ##################################
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            offset_loss = offset_loss + \
			            torch.sum( (offset_x_gt[confidance_gt==1] - predict_x[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1) + \
			            torch.sum( (offset_y_gt[confidance_gt==1] - predict_y[confidance_gt==1])**2 )/\
				        (torch.sum(confidance_gt==1)+1)

            #compute loss for similarity #################
            feature_map = feature.view(real_batch_size, self.p.feature_size, 1, self.p.grid_y*self.p.grid_x)
            feature_map = feature_map.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

            point_feature = feature.view(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x,1)
            point_feature = point_feature.expand(real_batch_size, self.p.feature_size, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)#.detach()

            distance_map = (feature_map-point_feature)**2 
            distance_map = torch.sum( distance_map, dim=1 ).view(real_batch_size, 1, self.p.grid_y*self.p.grid_x, self.p.grid_y*self.p.grid_x)

            # same instance
            sisc_loss = sisc_loss+\
				torch.sum(distance_map[ground_truth_instance==1])/\
				torch.sum(ground_truth_instance==1)

            # different instance, same class
            disc_loss = disc_loss + \
				torch.sum((self.p.K1-distance_map[ground_truth_instance==2])[(self.p.K1-distance_map[ground_truth_instance==2]) > 0])/\
				torch.sum(ground_truth_instance==2)

        #attention loss
        attention_loss = 0
        source = attentions[:-1]
        m = nn.Softmax(dim=0)
        
        for i in range(real_batch_size):
            target = torch.sum((attentions[-1][i].data)**2, dim=0).view(-1) 
            #target = target/torch.max(target)
            # print(len(target))
            target = m(target)
            for j in source:
                s = torch.sum(j[i]**2, dim=0).view(-1)
                attention_loss = attention_loss + torch.sum( (m(s) - target)**2 )/(len(target)*real_batch_size)

        lane_detection_loss = lane_detection_loss + self.p.constant_exist*exist_condidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_nonexist*nonexist_confidence_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_offset*offset_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_alpha*sisc_loss
        lane_detection_loss = lane_detection_loss + self.p.constant_beta*disc_loss + 0.00001*torch.sum(feature**2)
        lane_detection_loss = lane_detection_loss + self.p.constant_attention*attention_loss

        print("######################################################################")
        print("seg loss")
        print("same instance loss: ", sisc_loss.data)
        print("different instance loss: ", disc_loss.data)

        print("point loss")
        print("exist loss: ", exist_condidence_loss.data)
        print("non-exit loss: ", nonexist_confidence_loss.data)
        print("offset loss: ", offset_loss.data)

        print("attention loss")
        print("attention loss: ", attention_loss.data)

        print("--------------------------------------------------------------------")
        print("total loss: ", lane_detection_loss.data)

        self.lane_detection_optim.zero_grad()
        lane_detection_loss.backward()   #divide by batch size
        self.lane_detection_optim.step()

        del confidance, offset, feature
        del ground_truth_point, ground_binary, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss

        trim = 180
        if epoch>0 and self.current_epoch != epoch:
            self.current_epoch = epoch
            if epoch == 1-trim:
                self.p.l_rate = 0.0005
                self.setup_optimizer()
            elif epoch == 2-trim:
                self.p.l_rate = 0.0002
                self.setup_optimizer()
            elif epoch == 3-trim:
                self.p.l_rate = 0.0001
                self.setup_optimizer()
            elif epoch == 5-trim:
                self.p.l_rate = 0.00005
                self.setup_optimizer()
            elif epoch == 7-trim:
                self.p.l_rate = 0.00002
                self.setup_optimizer()
            elif epoch == 9-trim:
                self.p.l_rate = 0.00001
                self.setup_optimizer()
            elif epoch == 11-trim:
                self.p.l_rate = 0.000005
                self.setup_optimizer()
            elif epoch == 13-trim:
                self.p.l_rate = 0.000002
                self.setup_optimizer()
            elif epoch == 15-trim:
                self.p.l_rate = 0.000001
                self.setup_optimizer()
            elif epoch == 21-trim:  
                self.p.l_rate = 0.0000001
                self.setup_optimizer()
        return lane_detection_loss

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()
        outputs, features = self.lane_detection_network(inputs)

        return outputs

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, epoch, loss):
        self.lane_detection_network.load_state_dict(
            torch.load(self.p.model_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl', map_location='cuda:0'),False
        )

    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, loss):
        torch.save(
            self.lane_detection_network.state_dict(),
            self.p.save_path+str(epoch)+'_'+str(loss)+'_'+'lane_detection_network.pkl'
        )

    def get_data_list(self):
        return self.hard_sampling.get_list()

    def sample_reset(self):
        self.hard_sampling = hard_sampling.hard_sampling()
