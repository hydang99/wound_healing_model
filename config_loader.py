#!/usr/local/bin/python3
import os 
import json 
import logging 
import logging.config
from yaml import Loader 
import yaml
import io 
import sys 
from utils.utils import get_logger
logger = get_logger("CONFIG_LOADER")
class PDEParser:
    def __init__(self, config_path, **kwargs):
        super().__init__()
        self.root = '' #Root path 
        self.log = os.path.join(config_path['log']) #Log Folder 
        self.data = os.path.join(config_path['dataset']['pde'])
        self.res_img_path = os.path.join(self.log, config_path["pde"]["results"]["imgs"])
        self.res_npy = os.path.join(self.log, "npy_saved_final")
        self.res_param_path = os.path.join(self.log, config_path["pde"]["results"]["param_dict"])
        self.param = config_path['pde']['params']

class DLParser:
    def __init__(self, config_path, **kwargs):
        super().__init__()
        self.root = ''
        self.log = os.path.join(config_path['log'])

class Dataset:
    def __init__(self, config_path, **kwargs):
        super().__init__()
        self.root = ''
        self.patient_id = config_path['dataset']["patients"]

config_fp = os.path.join(os.path.dirname(__file__), 'config.yml')
try:
    with open(config_fp, 'r') as f: 
        config = yaml.load(f, Loader = Loader)
    pde_config = PDEParser(config_path = config)
    dl_config = DLParser(config_path = config)
    data_config = Dataset(config_path = config)
    logger.info("Config Loader Loaded")
except Exception as e:
    print(e)
