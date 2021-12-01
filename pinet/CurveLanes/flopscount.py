from hourglass_network import lane_detection_network
import torch
from thop import profile
from thop import clever_format

model = lane_detection_network()

input = torch.randn(1, 3, 256, 512)
macs, params = profile(model, inputs=(input, ))
print(macs, params)
# print('flops is %.2f G' % (flops/np.power(1024, 3)))
macs, params = clever_format([macs, params], "%.3f")
print('macs:{}\nparams:{}'.format(macs, params))