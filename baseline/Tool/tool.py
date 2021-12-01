import os
import numpy as np
import wget
import argparse
import base64
import json
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch

from Utils import utils
from DataProvider import cityscapes
from Models.Poly import polyrnnpp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True)
    parser.add_argument('--reload', required=True)
    parser.add_argument('--image_dir', default='static/images/')
    parser.add_argument('--port', type=int, default=5001)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'
    data_loader = DataProvider(split='val', opts=opts['train_val'], mode='tool')
    
    return data_loader

class Tool(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.image_dir = args.image_dir
        self.data_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)
        self.model.reload(args.reload, strict=False)

    def get_grid_size(self, run_ggnn=True):
        if self.opts['use_ggnn'] and run_ggnn:
            grid_size = self.model.ggnn.ggnn_grid_size
        else:
            grid_size = self.model.encoder.feat_size

        return grid_size

    def annotation(self, instance, run_ggnn=False):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            img = torch.from_numpy(img).to(device)
            # Add batch dimension and make torch Tensor

            output = self.model(
                img, 
                poly=None,
                fp_beam_size=5,
                lstm_beam_size=1,
                run_ggnn=run_ggnn
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

    def fixing(self, instance, run_ggnn=False):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['fwd_poly'], 0)
            img = torch.from_numpy(img).to(device)
            poly = torch.from_numpy(poly).to(device)
            # Add batch dimension and make torch Tensor

            output = self.model(
                img, 
                poly=poly,
                fp_beam_size=5,
                lstm_beam_size=1,
                run_ggnn=run_ggnn
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn)
        return self.process_output(polys, instance,
            grid_size)

    def run_ggnn(self, instance):
        with torch.no_grad():
            img = np.expand_dims(instance['img'], 0)
            poly = np.expand_dims(instance['fwd_poly'], 0)
            grid_size = self.get_grid_size(run_ggnn=False)
            poly = utils.xy_to_class(torch.from_numpy(poly), grid_size).numpy()
            img = torch.from_numpy(img).to(device)

            concat_feats, _ = self.model.encoder(img)
            output = self.model.ggnn(
                img,
                poly,
                mode = 'tool',
                resnet_feature = concat_feats
            )
            polys = output['pred_polys'].cpu().numpy()

        del(output)
        grid_size = self.get_grid_size(run_ggnn=True)
        return self.process_output(polys, instance,
            grid_size)

    def process_output(self, polys, instance, grid_size):
        poly = polys[0]
        poly = utils.get_masked_poly(poly, grid_size)
        poly = utils.class_to_xy(poly, grid_size)
        poly = utils.poly0g_to_poly01(poly, grid_size)
        poly = poly * instance['patch_w']
        poly = poly + instance['starting_point']

        torch.cuda.empty_cache() 
        return [poly.astype(np.int).tolist()]

@app.route('/api/annotation', methods=['POST'])
def generate_annotation():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = np.array([[-1., -1.]])
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.annotation(instance)

    print "Annotation time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/api/annotation_and_ggnn', methods=['POST'])
def generate_annotation_and_ggnn():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = np.array([[-1., -1.]])
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.annotation(instance, run_ggnn=True)
    pred_annotation[0] = pred_annotation[0][::2]    

    print "Annotation time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/api/fix_poly', methods=['POST'])
def fix_poly_request():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = instance['poly']
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.fixing(instance, run_ggnn=False)

    print "Fixing time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/api/fix_poly_ggnn', methods=['POST'])
def fix_poly_ggnn_request():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = instance['poly']
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.fixing(instance, run_ggnn=True)
    pred_annotation[0] = pred_annotation[0][::2]    

    print "Fixing time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/api/ggnn_poly', methods=['POST'])
def ggnn_poly_request():
    start = time.time()
    instance = request.json
    component = {}
    component['poly'] = instance['poly']
    instance = tool.data_loader.prepare_component(instance, component)
    pred_annotation = tool.run_ggnn(instance)

    print "GGNN time: " + str(time.time() - start)

    response = jsonify(results=[pred_annotation])
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/upload_v3', methods=['POST'])
def upload_v3():
    instance = request.json
    url = instance['url']
    out_dir = tool.image_dir
    filename = wget.download(url, out=out_dir)
    response = jsonify(path=filename)
    response.headers['Access-Control-Allow-Headers'] = '*'

    return response

@app.route('/upload_v2', methods=['POST'])
def upload_v2():
    instance = request.json
    base64im = instance['image']
    idx = len(os.listdir(tool.image_dir))
    try:
        extension = base64im.split('/')[1].split(';')[0]
        t = base64im.split('/')[0].split(':')[1]
        assert t == 'image', 'Did not get image data!'
        
        base64im = base64im.split(',')[1]
        out_name = os.path.join(tool.image_dir, str(idx) + '.' + extension)

        with open(out_name, 'w') as f:
            f.write(base64.b64decode(base64im.encode()))

        response = jsonify(path=out_name)

    except Exception as e:
        print e
        response = jsonify(path='')

    response.headers['Access-Control-Allow-Headers'] = '*'

    return response
    
if __name__ == '__main__':
    args = get_args()
    global tool
    tool = Tool(args)

    app.run(host='0.0.0.0', threaded=True, port=args.port)
