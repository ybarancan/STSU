import os
import torch
from torch.utils.data import DataLoader, RandomSampler

from nuscenes import NuScenes
from .nuscenes.dataset import NuScenesMapDataset
from .nuscenes.splits import TRAIN_SCENES, VAL_SCENES, CALIBRATION_SCENES

from nuscenes.map_expansion.map_api import NuScenesMap
from src.data.nuscenes import utils as nusc_utils
import logging


from argoverse.data_loading.argoverse_tracking_loader \
      import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap


from .argoverse.dataset import ArgoverseMapDataset
from .argoverse.splits import TRAIN_LOGS, VAL_LOGS
ALL_LOGS = TRAIN_LOGS + VAL_LOGS

def build_nuscenes_datasets(config,args, val=False, pinet=False):
    print('==> Loading NuScenes dataset...')
    nuscenes = NuScenes(config.nuscenes_version, 
                        os.path.expandvars(config.nusc_root))


    my_map_apis = { location : NuScenesMap(os.path.expandvars(config.nusc_root), location) 
             for location in nusc_utils.LOCATIONS }
    
    train_scenes = TRAIN_SCENES
    
    train_data = NuScenesMapDataset(nuscenes, config, my_map_apis, config.zoom_augment_prob > 0,
                                     train_scenes,  work_objects=args.objects)
    
    if val:
        val_seqs = VAL_SCENES
    else:
        val_seqs = CALIBRATION_SCENES
    
    val_data = NuScenesMapDataset(nuscenes, config,my_map_apis, False,
                                 val_seqs, pinet=pinet,work_objects=args.objects )
    return train_data, val_data


def build_argoverse_datasets(config,args, val=False, pinet=False):
      print('==> Loading Argoverse dataset...')
      dataroot = os.path.expandvars(config.argo_root)
      trackroot = os.path.join(dataroot, 'argoverse-tracking')
      am = ArgoverseMap()
      # Load native argoverse splits
    
      loaders = {
          'train' : ArgoverseTrackingLoader(os.path.join(trackroot, 'all_logs')),
          # 'val' : ArgoverseTrackingLoader(os.path.join(trackroot, 'all_logs'))
      }

      # Create datasets using new argoverse splits
      train_data = ArgoverseMapDataset(config, loaders['train'], am,
                                       TRAIN_LOGS,  train=True ,work_objects=args.objects)
      val_data = ArgoverseMapDataset(config,loaders['train'], am, 
                                      VAL_LOGS, train=False, pinet=pinet ,work_objects=args.objects )
      
      return train_data, val_data


#
def my_collate(batch):
    # to_return = []
    # if batch is list:
    #     for b in range(len(batch)):
    #         for k in range(len(batch[b])):
        
    # logging.error('COLLATE ' + str(len(batch)))
    problem = False
    for b in range(len(batch)):
        problem = problem | batch[b][-1]
    if problem:
        return (None,None,True)
    
    else:    
        
        images = []
        targets = []
        for b in range(len(batch)):
            images.append(batch[b][0])
            targets.append(batch[b][1])
            
        return (torch.stack(images,dim=0), targets, False)

def build_nuscenes_dataloader(config,args, val=False, pinet=False):
    train_data, val_data = build_nuscenes_datasets(config,args, val=val, pinet=pinet)
    sampler = RandomSampler(train_data, True)
    train_loader = DataLoader(train_data, config.batch_size, sampler=sampler, collate_fn = my_collate,
                              num_workers=1)
    
    val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                            num_workers=1)
    
    return train_loader,train_data, val_loader, val_data
    
    


def build_argoverse_dataloader(config,args, val=False, pinet=False):
    
      train_data, val_data = build_argoverse_datasets(config,args, val=val, pinet=pinet)

      sampler = RandomSampler(train_data, True)
      train_loader = DataLoader(train_data, config.batch_size, sampler=sampler,  collate_fn = my_collate,
                                 num_workers=1)
    
      val_loader = DataLoader(val_data, 1, collate_fn = my_collate,
                              num_workers=1)
     
      return train_loader, train_data, val_loader,val_data
    

