
import os
from yacs.config import CfgNode
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass
import logging
ROOT = os.path.abspath(os.path.join(__file__, '..', '..', '..'))


def load_config(config_path):
    with open(config_path) as f:
        return CfgNode.load_cfg(f)

def get_default_configuration():
    
    logging.error('LOADING CONFIG FROM : ' + str(os.path.join(ROOT, 'configs/defaults.yml')))
    
    defaults_path = os.path.join(ROOT, 'configs/defaults.yml')
    return load_config(defaults_path)


def get_default_polyline_configuration():
    defaults_path = os.path.join(ROOT, 'configs/polyline_defaults.yml')
    return load_config(defaults_path)


