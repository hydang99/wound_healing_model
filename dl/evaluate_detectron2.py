import os.path as osp
import os
import random
from shutil import copy2
import re
import numpy as np
from datetime import datetime,timedelta
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from tqdm.notebook import tqdm
import re
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import json
from detectron2 import model_zoo 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer 
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import os.path as osp
import os
import random
from shutil import copy2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from PIL import Image
import pickle

cfg = get_cfg()
import os
import torch

"""
Load Predicted Model
"""
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.MODEL.WEIGHTS = os.path.join("/home/hydang/wound_dl_pde/output", "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor = DefaultPredictor(cfg)

PATH = "dataset/data_origin/img_filter_4/"
PATH_SAVE = "dataset/data_origin/img_filter/"
def predict(path_predict, path_save): 
    res = dict()
    for p in tqdm(os.listdir(path_predict)):
        date = []
        if (".ipynb" in p): 
            continue
        res[p] = dict()
        if not os.path.exists(os.path.join(path_save, p)):
            os.makedirs(os.path.join(path_save, p), exist_ok=True)
        for img in os.listdir(os.path.join(path_predict,p)):
            date.append(img.split(".")[0]) #Sorting date
            date.sort(key=lambda my_date: datetime.strptime(my_date, "%Y-%m-%d"))
        if len(date) == 0: 
            continue
        min_date = datetime.strptime(date[0],"%Y-%m-%d").date()
        for img in os.listdir(os.path.join(path_predict,p)):
            im = cv2.imread(os.path.join(*(path_predict,p,img)))
            im = cv2.resize(im, (512,512))
            outputs = predictor(im)
            results_skin=np.zeros((512,512))
            results_leg=np.zeros((512,512))
            for cl in range(len(outputs["instances"].pred_classes)):
                if outputs["instances"].pred_classes[cl] == 0: 
                    results_skin = outputs["instances"].to("cpu").pred_masks[cl].long()
                else:
                    results_leg=outputs["instances"].to("cpu").pred_masks[cl].long()
            results = results_skin + 2*results_leg
            path_name = datetime.strptime(img.split(".")[0],"%Y-%m-%d").date()
            #plt.imsave(os.path.join(*(path_save,p,str((path_name-min_date).days)+back_name)), results.numpy())
            res[p][str((path_name-min_date).days)] = results
    return res
            
            
