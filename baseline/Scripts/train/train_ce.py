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

        if 'encoder_reload' in self.opts.keys():
            self.model.encoder.reload(self.opts['encoder_reload'])

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
            self.save_checkpoint(epoch) 
            self.lr_decay.step()
            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)

    def train(self, epoch):
        print 'Starting training'
        self.model.train()

        accum = defaultdict(float)
        # To accumulate stats for printing

        for step, data in enumerate(self.train_loader):
            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
                self.save_checkpoint(epoch)             

            # Forward pass
            output = self.model(data['img'].to(device), data['fwd_poly'].to(device))
            
            # Smoothed targets
            dt_targets = utils.dt_targets_from_class(output['poly_class'].cpu().numpy(),
                self.grid_size, self.opts['dt_threshold'])

            # Get losses
            loss = losses.poly_vertex_loss_mle(torch.from_numpy(dt_targets).to(device), 
                data['mask'].to(device), output['logits'])
            fp_edge_loss = self.opts['fp_weight'] * losses.fp_edge_loss(data['edge_mask'].to(device), 
                output['edge_logits'])
            fp_vertex_loss = self.opts['fp_weight'] * losses.fp_vertex_loss(data['vertex_mask'].to(device), 
                output['vertex_logits'])

            total_loss = loss + fp_edge_loss + fp_vertex_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip']) 

            self.optimizer.step()

            # Get accuracy
            accuracy = metrics.train_accuracy(output['poly_class'].cpu().numpy(), data['mask'].cpu().numpy(), 
                output['pred_polys'].cpu().numpy(), self.grid_size)

            # Get IoU
            iou = 0
            pred_polys = output['pred_polys'].cpu().numpy()
            gt_polys = data['full_poly']

            for i in range(pred_polys.shape[0]):
                p = pred_polys[i]
                p = utils.get_masked_poly(p, self.grid_size)
                p = utils.class_to_xy(p, self.grid_size)
                i, masks = metrics.iou_from_poly(p, gt_polys[i], self.grid_size, self.grid_size)
                iou += i

            iou = iou / pred_polys.shape[0]

            accum['loss'] += float(loss)
            accum['fp_edge_loss'] += float(fp_edge_loss)
            accum['fp_vertex_loss'] += float(fp_vertex_loss)
            accum['accuracy'] += accuracy
            accum['iou'] += iou
            accum['length'] += 1

            if step % self.opts['print_freq'] == 0:
                # Mean of accumulated values
                for k in accum.keys():
                    if k == 'length':
                        continue
                    accum[k] /= accum['length']

                # Add summaries
                masks = np.expand_dims(masks, -1).astype(np.uint8) # Add a channel dimension
                masks = np.tile(masks, [1, 1, 1, 3]) # Make [2, H, W, 3]
                img = (data['img'].cpu().numpy()[-1,...]*255).astype(np.uint8)
                img = np.transpose(img, [1,2,0]) # Make [H, W, 3]
                vert_logits = np.reshape(output['vertex_logits'][-1, ...].detach().cpu().numpy(), (self.grid_size, self.grid_size, 1))
                edge_logits = np.reshape(output['edge_logits'][-1, ...].detach().cpu().numpy(), (self.grid_size, self.grid_size, 1))
                vert_logits = (1/(1 + np.exp(-vert_logits))*255).astype(np.uint8)
                edge_logits = (1/(1 + np.exp(-edge_logits))*255).astype(np.uint8)
                vert_logits = np.tile(vert_logits, [1, 1, 3]) # Make [H, W, 3]
                edge_logits = np.tile(edge_logits, [1, 1, 3]) # Make [H, W, 3]
                vertex_mask = np.tile(np.expand_dims(data['vertex_mask'][-1,...].cpu().numpy().astype(np.uint8)*255,-1),(1,1,3))
                edge_mask = np.tile(np.expand_dims(data['edge_mask'][-1,...].cpu().numpy().astype(np.uint8)*255,-1),(1,1,3))

                self.writer.add_image('pred_mask', masks[0], self.global_step)
                self.writer.add_image('gt_mask', masks[1], self.global_step)
                self.writer.add_image('image', img, self.global_step)
                self.writer.add_image('vertex_logits', vert_logits, self.global_step)
                self.writer.add_image('edge_logits', edge_logits, self.global_step)
                self.writer.add_image('edge_mask', edge_mask, self.global_step)
                self.writer.add_image('vertex_mask', vertex_mask, self.global_step)

                if self.opts['return_attention'] is True:
                    att = output['attention'][-1, 1:4, ...].detach().cpu().numpy()
                    att = np.transpose(att, [0, 2, 3, 1]) # Make [T, H, W, 1]
                    att = np.tile(att, [1, 1, 1, 3]) # Make [T, H, W, 3]
                    def _scale(att):
                        att = att/np.max(att)
                        return (att*255).astype(np.int32)
                    self.writer.add_image('attention_1', pyramid_expand(_scale(att[0]), upscale=8, sigma=10), self.global_step)
                    self.writer.add_image('attention_2', pyramid_expand(_scale(att[1]), upscale=8, sigma=10), self.global_step)
                    self.writer.add_image('attention_3', pyramid_expand(_scale(att[2]), upscale=8, sigma=10), self.global_step)
                
                for k in accum.keys():
                    if k == 'length':
                        continue
                    self.writer.add_scalar(k, accum[k], self.global_step)

                print("[%s] Epoch: %d, Step: %d, Polygon Loss: %f, Edge Loss: %f, Vertex Loss: %f, Accuracy: %f, IOU: %f"\
                %(str(datetime.now()), epoch, self.global_step, accum['loss'], accum['fp_edge_loss'], accum['fp_vertex_loss'],\
                accum['accuracy'], accum['iou']))
                
                accum = defaultdict(float)

            del(output)
            self.global_step += 1

    def validate(self):
        print 'Validating'
        self.model.encoder.eval()
        self.model.first_v.eval()
        # Leave LSTM in train mode

        ious = []
        accuracies = []

        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                output = self.model(data['img'].to(device), data['fwd_poly'].to(device))

                # Get accuracy
                accuracy = metrics.train_accuracy(output['poly_class'].cpu().numpy(), data['mask'].cpu().numpy(), 
                    output['pred_polys'].cpu().numpy(), self.grid_size)

                # Get IoU
                iou = 0
                pred_polys = output['pred_polys'].cpu().numpy()
                gt_polys = data['full_poly']

                for i in range(pred_polys.shape[0]):
                    p = pred_polys[i]
                    p = utils.get_masked_poly(p, self.grid_size)
                    p = utils.class_to_xy(p, self.grid_size)
                    i, masks = metrics.iou_from_poly(p, gt_polys[i], self.grid_size, self.grid_size)
                    iou += i

                iou = iou / pred_polys.shape[0]
                ious.append(iou)
                accuracies.append(accuracy)

                del(output)

            iou = np.mean(ious)
            accuracy = np.mean(accuracies)

            self.val_writer.add_scalar('iou', float(iou), self.global_step)
            self.val_writer.add_scalar('accuracy', float(accuracy), self.global_step)

            print '[VAL] IoU: %f, Accuracy: %f'%(iou, accuracy)

        # Reset
        self.model.train()

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.loop()
