import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter
from skimage.transform import pyramid_expand
from tqdm import tqdm

from Utils import utils
from DataProvider import cityscapes
from Models.Poly import polyrnnpp
from Evaluation import losses, metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    dataset_train = DataProvider(split='train', opts=opts['train'])
    dataset_val = DataProvider(split='train_val', opts=opts['train_val'])

    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
        shuffle = True, num_workers=opts['train']['num_workers'], collate_fn=cityscapes.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=cityscapes.collate_fn)
    
    return train_loader, val_loader

class Trainer(object):
    def __init__(self, args):
        self.global_step = 0
        self.epoch = 0
        self.opts = json.load(open(args.exp, 'r'))
        utils.create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'))

        # Copy experiment file
        os.system('cp %s %s'%(args.exp, self.opts['exp_dir']))

        self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)
        self.grid_size = self.model.encoder.feat_size

        if 'xe_initializer' in self.opts.keys():
            self.model.reload(self.opts['xe_initializer'])

        elif 'encoder_reload' in self.opts.keys():
            self.model.encoder.reload(self.opts['encoder_reload'])

        print 'Setting only Evaluator to train'
        for n,p in self.model.named_parameters():
            if 'evaluator' not in n:
                p.requires_grad = False

        self.model.encoder.eval()
        self.model.first_v.eval()

        # OPTIMIZER
        no_wd = []
        wd = []
        print 'Weight Decay applied to: '

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'conv_lstm' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                print name,

        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'], 
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)
        # TODO: Test how amsgrad works (On the convergence of Adam and Beyond)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], 
            gamma=0.1)

        if args.resume is not None:
            self.resume(args.resume)

    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch%d_step%d.pth'\
        %(epoch, self.global_step))
        torch.save(save_state, save_name)
        print 'Saved model'

    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

        print 'Model reloaded to resume from Epoch %d, Global Step %d from model at %s'%(self.epoch, self.global_step, path) 

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.epoch = epoch
            self.lr_decay.step()
            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)

    def train(self, epoch):
        print 'Starting training'
        self.model.temperature = self.opts['temperature']
        self.model.evaluator.train()
    
        accum = defaultdict(float)
        # To accumulate stats for printing

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
                self.save_checkpoint(epoch)

            # Forward pass (Sampling)
            output = self.model(data['img'].to(device))
            
            # Get full GT masks
            gt_masks = []
            for instance in data['instance']:
                gt_masks.append(utils.get_full_mask_from_instance(
                    self.opts['dataset']['train_val']['min_area'], 
                    instance))

            # Get sampling masks
            sampling_masks = []
            pred_polys = output['pred_polys'].cpu().numpy()
            for i in range(pred_polys.shape[0]):
                poly = pred_polys[i]
                mask, poly = utils.get_full_mask_from_xy(poly,
                    self.grid_size,
                    data['patch_w'][i],
                    data['starting_point'][i],
                    data['instance'][i]['img_height'],
                    data['instance'][i]['img_width'])

                sampling_masks.append(mask)

            # Get IoUs
            sampling_ious = np.zeros((len(gt_masks)), dtype=np.float32)

            for i, gt_mask in enumerate(gt_masks):
                sampling_ious[i] = metrics.iou_from_mask(sampling_masks[i], gt_mask)

            # Get Loss
            total_loss = losses.evaluator_loss(output['ious'], torch.from_numpy(sampling_ious).to(device)) 

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip']) 

            self.optimizer.step()

            accum['loss'] += float(total_loss)
            accum['sampling_iou'] += np.mean(sampling_ious)
            accum['pred_iou'] += torch.mean(output['ious'])
            accum['length'] += 1

            if step % self.opts['print_freq'] == 0:
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                # Add summaries
                for k in accum.keys():
                    if k == 'length':
                        continue
                    self.writer.add_scalar(k, accum[k], self.global_step)

                print("[%s] Epoch: %d, Step: %d, Loss: %f, Sampling IOU: %f, Pred IOU: %f"\
                %(str(datetime.now()), epoch, self.global_step, accum['loss'], accum['sampling_iou'], accum['pred_iou']))
                
                accum = defaultdict(float)

            del(output)

            self.global_step += 1

    def validate(self):
        print 'Validating'
        self.model.temperature = 0
        # Use temperature 0 while validating
        self.model.evaluator.eval()

        ious = []
        pred_ious = []
        loss = []

        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                output = self.model(data['img'].to(device))

                # Get full GT masks
                gt_masks = []
                for instance in data['instance']:
                    gt_masks.append(utils.get_full_mask_from_instance(
                        self.opts['dataset']['train_val']['min_area'], 
                        instance))

                iou = np.zeros((len(gt_masks)), dtype=np.float32)

                pred_masks = []
                pred_polys = output['pred_polys'].cpu().numpy()
                for i in range(pred_polys.shape[0]):
                    poly = pred_polys[i]
                    mask, poly = utils.get_full_mask_from_xy(poly,
                        self.grid_size,
                        data['patch_w'][i],
                        data['starting_point'][i],
                        data['instance'][i]['img_height'],
                        data['instance'][i]['img_width'])

                    pred_masks.append(mask)

                for i, gt_mask in enumerate(gt_masks):
                    iou[i] = metrics.iou_from_mask(pred_masks[i], gt_mask)

                l = losses.evaluator_loss(output['ious'], torch.from_numpy(iou).to(device)) 

                iou = np.mean(iou)
                ious.append(iou)
                pred_ious.append(float(torch.mean(output['ious'])))
                loss.append(l)

                del(output)

            iou = np.mean(ious)
            pred_iou = np.mean(pred_ious)
            loss = np.mean(loss)

            self.val_writer.add_scalar('loss', float(loss), self.global_step)
            print '[VAL] GT IoU: %f, Pred IoU: %f, Loss: %f'%(iou, pred_iou, loss)

        # Reset
        self.model.temperature = self.opts['temperature']
        self.model.evaluator.train()

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.loop()
