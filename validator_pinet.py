import logging
import os

from argparse import ArgumentParser

import torch

import torch.nn as nn



from src.detr.matcher import build_matcher

from src.utils.configs import get_default_configuration, load_config


from src.utils.confusion import BinaryConfusionMatrix
from src.data import data_factory

import src.utils.visualise as vis_tools

from tqdm import tqdm
import numpy as np
from PIL import Image

import time
import glob


import pinet.CurveLanes.agent as agent

from pinet.CurveLanes.parameters import Parameters
import pinet.CurveLanes.util as util
import cv2
p = Parameters()

image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]

def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh

    grid = p.grid_location[mask]

    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([point_x])
                y.append([point_y])
            else:
                flag = 0
                index = 0
                min_feature_index = -1
                min_feature_dis = 10000
                for feature_idx, j in enumerate(lane_feature):
                    dis = np.linalg.norm((feature[i] - j)**2)
                    if min_feature_dis > dis:
                        min_feature_dis = dis
                        min_feature_index = feature_idx
                if min_feature_dis <= p.threshold_instance:
                    lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                    x[min_feature_index].append(point_x)
                    y[min_feature_index].append(point_y)
                elif len(lane_feature) < 12:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                
    return x, y
def test_ori(lane_agent, ori_image, test_images,w_ratio, h_ratio, draw_type, thresh=p.threshold_point):  # p.threshold_point:0.81

    result = lane_agent.predict_lanes_test(test_images)
    torch.cuda.synchronize()
    confidences, offsets, instances = result[-1]
    test_images = test_images.cpu().numpy()
#    logging.error('TEST TEST IMAGES ' + str(test_images.shape))
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []
    
    for i in range(num_batch):
        # test on test data set
        image = np.copy(test_images[i])
        image = np.rollaxis(image, axis=2, start=0)
        image = np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

#        logging.error('TEST LOOP IMAGE ' + str(image.shape))

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()
 
        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)
   
        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
     
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  

        if draw_type == 'line':
            result_image = util.draw_lines_ori(in_x, in_y, ori_image,w_ratio, h_ratio) 
        elif draw_type == 'point':
            result_image = util.draw_point_ori(in_x, in_y, ori_image,w_ratio, h_ratio)  
        else:
            result_image = util.draw_points(in_x, in_y,np.copy(image)) 

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)
        
    return out_x, out_y,  out_images


def evaluate(dataloader, model,  confusion, config,args):
     
     model.evaluate_mode()

     
     logging.error('VALIDATION')
     # Iterate over dataset
     for i, batch in enumerate(tqdm(dataloader)):
        
        seq_images, targets, _ = batch
        if seq_images == None:
            continue
        seq_images = seq_images.cuda()
        cuda_targets = []

            
        for b in targets:
            temp_dict={}
            temp_dict['center_img'] = b['center_img'].cuda()

            temp_dict['labels'] = b['labels'].cuda()
            temp_dict['roads'] = b['roads'].cuda()
            temp_dict['control_points'] = b['control_points'].cuda()
            temp_dict['con_matrix'] = b['con_matrix'].cuda()
            temp_dict['endpoints'] = b['endpoints'].cuda()

            temp_dict['mask'] = b['mask'].cuda()
            temp_dict['bev_mask'] = b['bev_mask'].cuda()
            
            temp_dict['obj_corners'] = b['obj_corners'].cuda()
            temp_dict['obj_converted'] = b['obj_converted'].cuda()
            temp_dict['obj_exists'] = b['obj_exists'].cuda()
            
            

            temp_dict['left_traffic'] = b['left_traffic'].cuda()
            temp_dict['outgoings'] = b['outgoings']
            temp_dict['incomings'] = b['incomings']
            cuda_targets.append(temp_dict)
            
            
        logging.error('SCENE ' + targets[0]['scene_name'])
        logging.error('SAMPLE ' + targets[0]['sample_token'])
  
        test_image = seq_images/255
        

        w_ratio = p.x_size * 1.0 / 800
        h_ratio = p.y_size* 1.0 / 448
        
        ori_image = np.uint8(cv2.resize(np.squeeze(np.transpose(seq_images.data.cpu().numpy(),(0,2,3,1)),axis=0),(800,448)))
        
        out_x, out_y, ti = test_ori(model, ori_image, test_image, w_ratio, h_ratio,draw_type= 'point',thresh=p.threshold_point)
        
        calib = targets[0]['calib'].numpy()

        
        coefs_list, boundaries_list, out_dict = vis_tools.get_spline_for_pinet(out_x[0],out_y[0], calib, targets[0])
        
        
        '''
        GET ESTIMATES BASED ON THRESHOLDING
        '''
 
        static_inter_dict = dict()
        static_inter_dict['src_boxes'] = out_dict['src_boxes']
        
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = vis_tools.hausdorff_match(out_dict, targets[0],pinet=True)
        try:
             confusion.update(out_dict, hausdorff_gt, hausdorff_static_idx,  targets[0],   static=True,pinet=True)
            
        except Exception as e:
             logging.error('EXCEPTION IN CONFUSION ')
             logging.error(str(e))
             continue

#        vis_tools.pinet_save_results_eval(seq_images.cpu().numpy(), [out_x, out_y, ti], coefs_list,boundaries_list,targets,  config)

        
     return confusion

def load_checkpoint(path, model, load_orig_ckpt=False):
    
    ckpt = torch.load(path)
    
  
    if isinstance(model, nn.DataParallel):
        model = model.module
        
        
        
    model.load_state_dict(ckpt['model'],strict=True)
    # with torch.no_grad():
    #     model.left_object_embed.weight.copy_(model.object_embed.weight)
 

    
    if 'iteration' not in ckpt.keys():
        to_return_iter = 0
    else:
        to_return_iter = ckpt['iteration']
    # to_return_iter = 0
    logging.error('LOADED MY')
    return ckpt['epoch'], ckpt['best_iou'],to_return_iter



# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration()



    return config


def create_experiment(config,  resume=None):

    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
        
    else:
        # Otherwise, generate a run directory based on the current time
        # name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format('run')
        name = 'pinet'
        logdir = os.path.join(os.path.expandvars(config.logdir), name)
        print("\n==> Creating new experiment in directory:\n" + logdir)
        os.makedirs(logdir,exist_ok=True)
        os.makedirs(os.path.join(config.logdir,'val_images'),exist_ok=True)
        os.makedirs(os.path.join(config.logdir,'train_images'),exist_ok=True)
        
        # Display the config options on-screen
        print(config.dump())
        
        # Save the current config
        with open(os.path.join(logdir, 'config.yml'), 'w') as f:
            f.write(config.dump())
        
    return logdir



    
def freeze_backbone_layers(model):
    logging.error('MODEL FREEZE')
    for n, p in model.named_parameters():
#        logging.error('STR ' + str(n))
        if "backbone" in n and p.requires_grad:
            
#            if  (('block14' in n) |('block15' in n) |('block16' in n) |('block17' in n) |('block18' in n) 
#                 |('block19' in n) | ('block20' in n) | ('block21' in n) | ('spp' in n)):
            if  ( ('block18' in n) |('block19' in n) | ('block20' in n) | ('block21' in n) | ('spp' in n)):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
            # logging.error(str(n) + ', '+str(p.requires_grad))
    
#                logging.error(str(n) + ', '+str(p.requires_grad))


object_refinement = True

apply_poly_loss = True

split_pe = True

apply_bev_pe = True
abs_bev = True

only_bev_pe=False


num_object_classes = 8

base_dir = '/scratch_net/catweazle/cany/lanefinder'


def main():

    large_parameters =  dict()
    large_parameters['hidden_dim'] = 256
    large_parameters['dim_feedforward'] = 512
    
    large_parameters['class_embed_dim']=256
    large_parameters['class_embed_num']=3
    
    large_parameters['box_embed_dim']=256
    large_parameters['box_embed_num']=3
    large_parameters['endpoint_embed_dim']=256
    large_parameters['endpoint_embed_num']=3
    large_parameters['assoc_embed_dim']=256
    large_parameters['assoc_embed_last_dim']=128
    large_parameters['assoc_embed_num']=3
    large_parameters['assoc_classifier_dim']=256
    large_parameters['assoc_classifier_num']=3
    
    
    num_queries = 100
    num_enc_layers = 4
    num_dec_layers = 4
    
#        model_name = 'maxi_combined_objects_3'
    model_name = 'pinet'
     
    parser = ArgumentParser()

    parser.add_argument('--resume', default=None, 
                        help='path to an experiment to resume')

    parser.add_argument('--apply_poly_loss', type=bool, default=apply_poly_loss,
                    help='whether it is on dgx')
    
    parser.add_argument('--objects', type=bool, default=True,
                help='whether estimate objects')
    
    parser.add_argument('--num_object_queries', default=100, type=int,
                    help="Number of query slots")
    
  
    
    parser.add_argument('--num_object_classes', default=8, type=int,
                help="Num object classes")
    
    parser.add_argument('--num_spline_points', default=3, type=int,
                help="Num object classes")
    
    

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=num_enc_layers, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=num_dec_layers, type=int,
                        help="Number of decoding layers in the transformer")
    
    
    parser.add_argument('--dim_feedforward', default=large_parameters['dim_feedforward'], type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    
    
    parser.add_argument('--hidden_dim', default=large_parameters['hidden_dim'], type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=num_queries, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks',default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_obj_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_obj_cost_center', default=3, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_len', default=0.5, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_orient', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_obj_cost_image_center', default=0, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_end', default=1, type=float,
                        help="L1 endpoint coefficient in the matching cost")
    
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    
    parser.add_argument('--object_detection_loss_coef', default=4, type=float)
    parser.add_argument('--object_center_loss_coef', default=3, type=float)
    parser.add_argument('--object_len_loss_coef', default=0.5, type=float)
    parser.add_argument('--object_orient_loss_coef', default=0.5, type=float)
    
    parser.add_argument('--polyline_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--assoc_loss_coef', default=1, type=float)
    parser.add_argument('--detection_loss_coef', default=3, type=float)
    parser.add_argument('--endpoints_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--focal_loss_coef', default=0.1, type=float)
    
    parser.add_argument('--loss_end_match_coef', default=1, type=float)
    
    
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--visible_loss_coef', default=1, type=float)
    
    parser.add_argument('--eos_coef', default=0.2, type=float,
                        help="Relative classification weight of the no-object class")
    
    parser.add_argument('--object_eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
   
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval',default=False, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    args = parser.parse_args()
    
    
    print('GOT ARGS ')
    logging.error(str(args))
    
    # Load configuration
    config = get_configuration(args)
    
    # Create a directory for the experiment
    logdir = create_experiment(config, args.resume)
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    config.freeze()
    
    device = torch.device(args.device)
    # Setup experiment
    model = agent.Agent()
        # lane_agent.load_weights(804, "tensor(0.5786)")
    model.load_weights(32, "tensor(1.1001)")

    model.to(device)
    
    if config.train_dataset == 'nuscenes':
    
        train_loader,train_dataset, val_loader, val_dataset = data_factory.build_nuscenes_dataloader(config,args, val=True, pinet=True)
        
    else:
        train_loader,train_dataset, val_loader, val_dataset = data_factory.build_argoverse_dataloader(config,args, val=True, pinet=True)

    
    logging.error('LOADED MY CHECKPOINT')

    val_confusion = BinaryConfusionMatrix(1,args.num_object_classes)
    val_con = evaluate(val_loader, model, val_confusion,config, args)
    
    static_res_dict, object_res_dict = val_con.get_res_dict
    file1 = open(os.path.join(logdir,'val_res.txt'),"a")
    
    for k in static_res_dict.keys():
        logging.error(str(k) + ' : ' + str(static_res_dict[k]))
        file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
        
    for k in object_res_dict.keys():
        logging.error(str(k) + ' : ' + str(object_res_dict[k]))
        file1.write(str(k) + ' : ' + str(object_res_dict[k]) + ' \n')
    
    file1.close()    
    

if __name__ == '__main__':
    main()

                

