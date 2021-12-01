import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import create_encoder
from .spp import create_spp




class SPPNet(nn.Module):
    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8):
        super().__init__()
        self.output_channels = output_channels
        self.num_channels = 256
        self.enc_type = enc_type
        self.dec_type = dec_type
        
        
        assert enc_type in ['xception65', 'mobilenetv2']
        assert dec_type in ['oc_base', 'oc_asp', 'spp', 'aspp', 'maspp']
        
        self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=False)
        # if enc_type == 'mobilenetv2':
        #     self.spp = create_mspp(dec_type)
        # else:
        self.spp = create_spp(dec_type, output_stride=output_stride)
        # self.logits = nn.Conv2d(256, output_channels, 1)

    def forward(self, inputs):
   
        x, low_level_feat = self.encoder(inputs)
        x = self.spp(x)
        
        x = F.interpolate(x, size=(int(x.shape[2]/2),int(x.shape[3]/2)), mode='bilinear', align_corners=False)
        # x = self.decoder(x, low_level_feat)
   
        return x, low_level_feat

    def update_bn_eps(self):
        for m in self.encoder.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eps = 1e-3

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                # for p in m.parameters():
                #     p.requires_grad = False

    def get_1x_lr_params(self):
        for p in self.encoder.parameters():
            yield p

    def get_10x_lr_params(self):
        modules = [self.spp, self.logits]
        if hasattr(self, 'decoder'):
            modules.append(self.decoder)

        for module in modules:
            for p in module.parameters():
                yield p
