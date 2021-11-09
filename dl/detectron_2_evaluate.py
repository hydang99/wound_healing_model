import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

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
from utils.utils import move_train_test 
import random
import pickle
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode

main_folder = "dataset/data_new/data"
path_divided = {"train": "dataset/data_final/train", "valid": "dataset/data_final/valid", "test": "dataset/data_final/test"}
def get_wound_dict(directory):
    classes = ['leg', 'wound']
    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(directory)):
        if "check" in filename:
            continue
        json_file = os.path.join(*[directory, filename, str(filename) + ".json"])
        with open(json_file) as f:
            img_anns = json.load(f)
        record = {}
        leg = False
        filename = os.path.join(*[directory, filename, str(filename) + ".jpg"])
        
        record["file_name"],record["image_id"],record["height"], record["width"] = filename, filename, 512, 512
        img = cv2.imread(record["file_name"])    
        h_scale = 512/img_anns["imageHeight"]
        w_scale = 512/img_anns["imageWidth"]
        img = cv2.resize(img, (512,512))
        cv2.imwrite(record["file_name"],img)
        annos = img_anns["shapes"]
        objs = []
        for anno in annos: 
            px = [(a[0]*w_scale) for a in anno['points']]
            py = [(a[1]*h_scale) for a in anno['points']]
            poly = [(x,y) for x,y in zip(px,py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

def main():
    dataset_dicts = get_wound_dict(path_divided["train"])
    for d in ["train", "valid", "test"]:
        dataset_dicts = get_wound_dict(path_divided[d])
        with open("{}.pickle".format(d), "wb") as f:
            pickle.dump(dataset_dicts, f)
    wound_metadata = MetadataCatalog.get("wound_train")
    for d in random.sample(dataset_dicts, 2):
        img = cv2.imread(d["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=wound_metadata, scale=1)
        v = v.draw_dataset_dict(d)
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("wound_train",)
    cfg.DATASETS.TEST = ("wound_valid",) 
    cfg.TEST.EVAL_PERIOD = 5
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    predictor = DefaultPredictor(cfg)
    dataset_dicts = get_wound_dict(path_divided["test"])
    for d in random.sample(dataset_dicts, 12):   
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=wound_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()
        t_plot_leg = outputs["instances"].to("cpu").pred_masks[0].long()
        if len(outputs["instances"].scores) != 1:
            t_plot_wound = outputs["instances"].to("cpu").pred_masks[1].long()
            plt.imshow(t_plot_leg.numpy()+ t_plot_wound.numpy()*2)
        else:
            plt.imshow(t_plot_leg.numpy())