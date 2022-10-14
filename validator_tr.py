import logging
import os
from datetime import datetime
from argparse import ArgumentParser
#from progressbar import progressbar

import torch

import torch.nn as nn
from src.detr.detr import build

from src.utils.configs import get_default_configuration, load_config

from src.utils.confusion import BinaryConfusionMatrix
from src.data import data_factory

import src.utils.visualise as vis_tools

from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import glob


image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]



def evaluate(dataloader, model, criterion, postprocessors, confusion, config,args,
            thresh):
     
     model.eval()
  
     criterion.eval()
     
     
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
            temp_dict['calib'] = b['calib'].cuda()
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
        
  
        outputs = model(seq_images,cuda_targets[0]['calib'], targets[0]['left_traffic'])

        
        static_thresh = thresh
        object_thresh = 0.3
        
        match_static_indices, match_object_indices = criterion.matcher(outputs, cuda_targets, val=True, thresh=static_thresh)
        matched_static_outputs = criterion.get_assoc_estimates( outputs, match_static_indices)
        
        static_inter_dict, static_idx, static_target_ids = criterion.get_interpolated(matched_static_outputs, cuda_targets, match_static_indices)

        
        '''
        GET ESTIMATES BASED ON THRESHOLDING
        '''
     
        outputs = model.thresh_and_assoc_estimates(outputs,thresh=static_thresh)
        
        base_postprocessed, object_post = postprocessors['bbox'](outputs,torch.Tensor(np.tile(np.expand_dims(np.array(config.patch_size),axis=0),[seq_images.shape[0],1])).cuda(),objects=True)
        
        out = vis_tools.get_selected_estimates(base_postprocessed , thresh = static_thresh)
        
        hausdorff_static_dist, hausdorff_static_idx, hausdorff_gt = vis_tools.hausdorff_match(out[0], targets[0])
      
        try:
             confusion.update(out[0], hausdorff_gt, hausdorff_static_idx,  targets[0],   static=True)
            
        except Exception as e:
             logging.error('EXCEPTION IN CONFUSION ')
             logging.error(str(e))
             continue
            
        
        '''
        OBJECT
        '''
        if args.objects:
            out_objects = vis_tools.get_selected_objects(object_post , thresh = object_thresh)
            if args.object_refinement:
                
                if out_objects['anything_to_feed']:
                    refine_logits, refine_out = model.refine_obj_seg(outputs, out_objects, cuda_targets[0]['calib'])
                      
                    out_objects['refine_out'] = refine_out
            if targets[0]['obj_exists']:
                object_inter_dict, object_idx, object_target_ids = criterion.get_object_interpolated(outputs, cuda_targets, match_object_indices)
            else:
                object_inter_dict = None
                object_idx = None
                object_target_ids = None
            
            try:
                confusion.update(out_objects, None,object_idx, targets[0],  static=False)
            except Exception as e:
                 logging.error('EXCEPTION IN CONFUSION ')
                 logging.error(str(e))
                 continue
      
        if targets[0]['obj_exists']:
            vis_tools.save_results_eval(seq_images.cpu().numpy(), out,out_objects, targets, static_inter_dict, object_inter_dict, static_target_ids, object_target_ids, config)
        else:
            vis_tools.save_results_eval(seq_images.cpu().numpy(), out,out_objects, targets, static_inter_dict,None, static_target_ids, None, config)
#                
            
     return confusion

def load_checkpoint(path, model, load_orig_ckpt=False):
    
    
    ckpt = torch.load(path)
    logging.error('LOADED ' + path)
  
    if isinstance(model, nn.DataParallel):
        model = model.module
        
        
        
    model.load_state_dict(ckpt['model'],strict=False)

    return 0,0,0



# Load the configuration for this experiment
def get_configuration(args):

    # Load config defaults
    config = get_default_configuration()


    return config


def create_experiment(config, name = 'tr_lanefinder_nuscenes'):

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

base_dir = '/cluster/work/cvl/cany/lanefinder'


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
    
    parser = ArgumentParser()
    
    parser.add_argument('--split_pe', type=bool, default=split_pe,
                    help='whether it is on dgx')
    
    parser.add_argument('--object_refinement', type=bool, default=object_refinement,
                    help='whether it is on dgx')
    
    parser.add_argument('--only_bev_pe', type=bool, default=only_bev_pe,
                    help='whether it is on dgx')
    # '/scratch_net/catweazle/cany/lanefinder/combined_objects_4'
    parser.add_argument('--bev_pe', type=bool, default=apply_bev_pe,
                    help='whether it is on dgx')
    parser.add_argument('--abs_bev', type=bool, default=abs_bev,
                    help='whether it is on dgx')
    
    parser.add_argument('--apply_poly_loss', type=bool, default=apply_poly_loss,
                    help='whether it is on dgx')
    
    parser.add_argument('--objects', type=bool, default=True,
                help='whether estimate objects')
    
    parser.add_argument('--num_object_queries', default=100, type=int,
                    help="Number of query slots")
    
    parser.add_argument('--num_object_classes', default=num_object_classes, type=int,
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
    
    parser.add_argument('--dropout', default=0.0, type=float,
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
    parser.add_argument('--set_obj_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_obj_cost_center', default=2, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_len', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_obj_cost_orient', default=1, type=float,
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

#    parser.add_argument('--object_detection_loss_coef', default=4, type=float)
    
    parser.add_argument('--object_detection_loss_coef', default=3, type=float)
    
    
    parser.add_argument('--object_center_loss_coef', default=3, type=float)
    parser.add_argument('--object_len_loss_coef', default=1, type=float)
    parser.add_argument('--object_orient_loss_coef', default=2, type=float)
    
    parser.add_argument('--object_refine_loss_coef', default=1, type=float)
    
    
    parser.add_argument('--polyline_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--assoc_loss_coef', default=1, type=float)
    parser.add_argument('--detection_loss_coef', default=2, type=float)
    parser.add_argument('--endpoints_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--focal_loss_coef', default=0.1, type=float)
    
    parser.add_argument('--loss_end_match_coef', default=1, type=float)
    
    
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--visible_loss_coef', default=1, type=float)
    
    parser.add_argument('--eos_coef', default=0.3, type=float,
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
    logdir = create_experiment(config, name = 'tr_lanefinder_'+str(config.train_dataset))
    
    config.save_logdir = logdir
    config.n_control_points = args.num_spline_points
    config.freeze()
    

    # Set default device
    # if len(config.gpus) > 0:
    #     torch.cuda.set_device(config.gpus[0])
    device = torch.device(args.device)
    # Setup experiment
    model, criterion, postprocessors = build(args, config,large_parameters)
    
    model.to(device)
    
    if config.train_dataset == 'nuscenes':
    
        train_loader,train_dataset, val_loader, val_dataset = data_factory.build_nuscenes_dataloader(config,args, val=True)
        
    else:
        train_loader,train_dataset, val_loader, val_dataset = data_factory.build_argoverse_dataloader(config,args, val=True)

    # epoch, best_iou, iteration = load_checkpoint(os.path.join('/scratch_net/catweazle/cany/simplice/maxi_objectsFalse_BASE_gt_train', 'temp.pth'),
    #                               model)
    epoch, best_iou, iteration = load_checkpoint(os.path.join(base_dir, 'maxi_poly_loss_split_True_refineTrue', 'keep', 'latest.pth'),
                                  model)
    
    logging.error('LOADED MY CHECKPOINT')

    
    freeze_backbone_layers(model)
   
    thresh = 0.3
    val_con = evaluate(val_loader, model, criterion, postprocessors,BinaryConfusionMatrix(1,args.num_object_classes), config, args, thresh)

    static_res_dict, object_res_dict = val_con.get_res_dict
    file1 = open(os.path.join(logdir,'val_res_thresh_'+str(thresh)+'.txt'),"a")
    
    for k in static_res_dict.keys():
         logging.error(str(k) + ' : ' + str(static_res_dict[k]))
         file1.write(str(k) + ' : ' + str(static_res_dict[k]) + ' \n')
        
    for k in object_res_dict.keys():
         logging.error(str(k) + ' : ' + str(object_res_dict[k]))
         file1.write(str(k) + ' : ' + str(object_res_dict[k]) + ' \n')
    
    file1.close()    
        
    
       


if __name__ == '__main__':
    main()

                

