import logging
import os
import os.path as osp
import json
from shutil import copy2
import random
def get_logger(name):
    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        filename='log/all_log.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(name).addHandler(console)
    return logging.getLogger(name)


"""
Deep Learning Utils, Detectron
"""
def show_sample(ids,train_path_images,train_path_masks,leg = True):
    plt.figure(figsize=(20,10))
    for j, img_name in enumerate(ids):
        q = j+1
        img = imread(train_path_images+"/"+img_name)
        path = img_name.split(".")
        path = path[:-1]
        path = "".join(path)
        if leg == True:
            img_mask = imread(train_path_masks+"/"+path+"-leg.jpg")
        else:
            img_mask = imread(train_path_masks+"/"+path+"-wound.jpg")          

        plt.subplot(1,2*(1+len(ids)),q*2-1)
        plt.imshow(img)
        plt.subplot(1,2*(1+len(ids)),q*2)
        plt.imshow(img_mask)
    plt.show()

#Normalize the image and reshape it to be one dimensional
def normalize_reshape(X_train,Y_train):
    X_train_shaped = X_train.reshape(-1, 3, 512, 512)/255
    print(np.shape(X_train_shaped))
    Y_train_shaped = Y_train.reshape(-1, 1, 512, 512)/255
    X_train_shaped = X_train_shaped.astype(np.float32)
    Y_train_shaped = Y_train_shaped.astype(np.float32)
    return X_train_shaped,Y_train_shaped

def check_polygon_annotation(main_folder, json_file):
    """
    For Detectron2 -> It can not process the json file with number of label points <= 6 
    """
    file_path = os.path.join(main_folder, json_file)
    with open(file_path) as f:
        img_anns = json.load(f)
    annos = img_anns["shapes"]
    for anno in annos:
        px = [(a[0]) for a in anno['points']]
        py = [(a[1]) for a in anno['points']]
        poly = [(x,y) for x,y in zip(px,py)]
        poly = [p for x in poly for p in x]
        if len(poly) <= 6:
            return True
    return False
    
def move_train_test(main_folder, paths, valid = 0.1, test = 0.1):
    classes = ["train", "valid", "test"]
    #Set of indices 
    indices = list(range(0, len([file for file in os.listdir(main_folder) if file.endswith('.json')])))
    random.shuffle(indices)
    train = indices[:int((1-valid-test)*len(indices))]
    valid = indices[int((1-valid-test)*len(indices)):int((1-test)*len(indices))]
    test = indices[int((1-test)*len(indices)):]
    assert(len(train) + len(valid) + len(test) == len(indices))
    for idx, filename in enumerate([file for file in os.listdir(main_folder) if file.endswith('.json')]):
        poly_invalid = check_polygon_annotation(main_folder, filename)
        id_file = "".join(filename.split(".")[:-1])
        if idx in train:
            if poly_invalid:
                train.remove(idx)
                indices.remove(idx)
                continue
            if not osp.exists(osp.join(paths["train"], str(idx))):
                os.makedirs(osp.join(paths["train"], str(idx)), exist_ok = True)
                copy2(osp.join(main_folder, filename), osp.join(*[paths["train"], str(idx),str(idx) + ".json"]))
                try:
                    copy2(osp.join(main_folder, id_file + ".jpg"), osp.join(*[paths["train"], str(idx),str(idx) + ".jpg"]))
                except FileNotFoundError:
                    copy2(osp.join(main_folder, id_file + ".jpeg"), osp.join(*[paths["train"], str(idx),str(idx) + ".jpg"]))      
        elif idx in valid: 
            if poly_invalid:
                valid.remove(idx)
                indices.remove(idx)
                continue
            if not osp.exists(osp.join(paths["valid"], str(idx))):
                os.makedirs(osp.join(paths["valid"], str(idx)), exist_ok = True)

                copy2(osp.join(main_folder, filename), osp.join(*[paths["valid"], str(idx),str(idx) + ".json"]))
                try:
                    copy2(osp.join(main_folder, id_file + ".jpg"), osp.join(*[paths["valid"], str(idx),str(idx) + ".jpg"]))
                except FileNotFoundError:
                    copy2(osp.join(main_folder, id_file + ".jpeg"), osp.join(*[paths["valid"], str(idx),str(idx) + ".jpg"]))
        elif idx in test:
            if poly_invalid:
                test.remove(idx)
                indices.remove(idx)
                continue
            if not osp.exists(osp.join(paths["test"], str(idx))):
                os.makedirs(osp.join(paths["test"], str(idx)), exist_ok = True)
                copy2(osp.join(main_folder, filename), osp.join(*[paths["test"], str(idx),str(idx) + ".json"]))
                try:
                    copy2(osp.join(main_folder, id_file + ".jpg"), osp.join(*[paths["test"], str(idx),str(idx) + ".jpg"]))
                except FileNotFoundError:
                    copy2(osp.join(main_folder, id_file + ".jpeg"), osp.join(*[paths["test"], str(idx),str(idx) + ".jpg"]))
        else:
            print("Check again index")
    assert(len(os.listdir(paths["train"])) + len(os.listdir(paths["valid"])) + len(os.listdir(paths["test"])) == len(indices))