'''
Convert trained model into onnx.

'''

import torch
import torch.onnx
from hourglass_network import lane_detection_network

# (True)Convert to onnx mode.
# (False)Check converted onnx model mode.
convert = True 
save_dir = './onnx_models/'
if convert == True:

    model = lane_detection_network()
    weights_path = './savefile/32_tensor(1.1001)_lane_detection_network.pkl'

    # Load the weights from a file (.pth or .pkl usually)
    state_dict = torch.load(weights_path)

    # Load the weights now into a model net architecture.
    model.load_state_dict(state_dict)

    # Create the right input shape.
    sample_batch_size = 1
    channel = 3
    height = 256
    width = 512
    dummy_input = torch.randn(sample_batch_size, channel, height, width)

    torch.onnx.export(model, dummy_input, save_dir + "pinet_v2.onnx", verbose = True)

if convert == False:
    import onnx
    # Load the onnx model
    model = onnx.load(save_dir + "pinet_v2.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph.
    print(onnx.helper.printable_graph(model.graph))

