#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from pinet.CurveLanes.util_hourglass import *

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        # base:original setting.
        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)


    def forward(self, inputs):
        #feature extraction
        out = self.resizing(inputs)
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)   
        result3, out, feature3 = self.layer3(out)
        result4, out, feature4 = self.layer4(out)
        return [result1, result2, result3, result4], [feature1, feature2, feature3, feature4]
        #return [result1], [feature1]
