import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    dataset_train = DataProvider(split='train', opts=opts['train'])
    dataset_val = DataProvider(split='train_val', opts=opts['train_val'])

    train_loader = DataLoader(dataset_train, batch_size=opts['train']['batch_size'],
        shuffle=False, num_workers=opts['train']['num_workers'], collate_fn=cityscapes.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle=False, num_workers=opts['train_val']['num_workers'], collate_fn=cityscapes.collate_fn)

    return train_loader, val_loader

class Trainer(object):
    def __init__(self, args):
        self.global_step = 0
        self.epoch = 0

        self.opts = json.load(open(args.exp, 'r'))
        utils.create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'))

        # Copy experiment file
        os.system('cp %s %s' % (args.exp, self.opts['exp_dir']))

        self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.train_loader, self.val_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)

        if 'xe_initializer' in self.opts.keys():
            self.model.reload(self.opts['xe_initializer'])
        elif 'encoder_reload' in self.opts.keys():
            self.model.encoder.reload(self.opts['encoder_reload'])

        # set
        self.model.encoder.eval()
        self.model.first_v.eval()
        self.model.evaluator.eval()

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

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print 'Saved model'

    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])

    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            self.save_checkpoint(epoch)
            self.lr_decay.step()
            print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)

    def train(self, epoch):
        print 'Starting training'
        self.model.temperature = self.opts['temperature']

        self.model.ggnn.encoder.train()

        accum = defaultdict(float)
        # To accumulate stats for printing
        ggnn_grid_size =  self.opts['ggnn_grid_size']

        for step, data in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            if self.global_step % self.opts['val_freq'] == 0:
                self.validate()
                self.save_checkpoint(epoch)

            output = self.model(data['img'].to(device), data['fwd_poly'].to(device), orig_poly=data['orig_poly'])

            ggnn_logits = output['ggnn_logits']
            local_prediction = output['ggnn_local_prediction'].to(device)
            poly_masks = output['ggnn_mask'].to(device)
            pred_polys = output['pred_polys'].data.numpy()

            loss_sum = losses.poly_vertex_loss_mle_ggnn(local_prediction,
                                                        poly_masks,
                                                        ggnn_logits)

            loss_sum.backward()

            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.ggnn.parameters(), self.opts['grad_clip'])

            self.optimizer.step()

            with torch.no_grad():
                # Get IoU
                iou = 0
                orig_poly = data['orig_poly']

                for i in range(pred_polys.shape[0]):
                    p = pred_polys[i]

                    mask_poly = utils.get_masked_poly(p, self.model.ggnn.ggnn_grid_size)
                    mask_poly = utils.class_to_xy(mask_poly, self.model.ggnn.ggnn_grid_size)

                    curr_gt_poly_112 = utils.poly01_to_poly0g(orig_poly[i], ggnn_grid_size)

                    cur_iou, masks = metrics.iou_from_poly(np.array(mask_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly_112, dtype=np.int32),
                                                           ggnn_grid_size,
                                                           ggnn_grid_size)

                    iou += cur_iou
                iou = iou / pred_polys.shape[0]
                accum['loss'] += float(loss_sum.item())
                accum['iou'] += iou
                accum['length'] += 1
                if step % self.opts['print_freq'] == 0:
                    # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    # Add summaries
                    masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
                    masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
                    img = (data['img'].cpu().numpy()[-1, ...] * 255).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]

                    self.writer.add_image('pred_mask', masks[0], self.global_step)
                    self.writer.add_image('gt_mask', masks[1], self.global_step)
                    self.writer.add_image('image', img, self.global_step)

                    for k in accum.keys():
                        if k == 'length':
                            continue
                        self.writer.add_scalar(k, accum[k], self.global_step)

                    print(
                    "[%s] Epoch: %d, Step: %d, Polygon Loss: %f,  IOU: %f" \
                    % (str(datetime.now()), epoch, self.global_step, accum['loss'], accum['iou']))

                    accum = defaultdict(float)

            del (output, local_prediction, poly_masks, masks, ggnn_logits, pred_polys, loss_sum)
            self.global_step += 1

    def validate(self):
        print 'Validating'
        ggnn_grid_size =  self.opts['ggnn_grid_size']
        self.model.ggnn.encoder.eval()
        self.model.temperature = 0
        self.model.mode = "test"
        # Leave LSTM in train mode

        with torch.no_grad():
            ious = []
            for step, data in enumerate(tqdm(self.val_loader)):

                output = self.model(data['img'].to(device), data['fwd_poly'].to(device))
                pred_polys = output['pred_polys'].data.numpy()

                # Get IoU
                iou = 0
                orig_poly = data['orig_poly']

                for i in range(pred_polys.shape[0]):

                    p = pred_polys[i]

                    mask_poly = utils.get_masked_poly(p, self.model.ggnn.ggnn_grid_size)
                    mask_poly = utils.class_to_xy(mask_poly, self.model.ggnn.ggnn_grid_size)

                    curr_gt_poly_112 = utils.poly01_to_poly0g(orig_poly[i], ggnn_grid_size)

                    i, masks = metrics.iou_from_poly(np.array(mask_poly, dtype=np.int32),
                                                     np.array(curr_gt_poly_112, dtype=np.int32), ggnn_grid_size, ggnn_grid_size)

                    iou += i

                iou = iou / pred_polys.shape[0]
                ious.append(iou)

                del (output)
                del (pred_polys)

            iou = np.mean(ious)
            self.val_writer.add_scalar('iou', float(iou), self.global_step)

            print '[VAL] IoU: %f' % iou

        self.model.temperature = self.opts['temperature']
        self.model.mode = "train_ggnn"
        self.model.ggnn.encoder.train()

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.loop()
