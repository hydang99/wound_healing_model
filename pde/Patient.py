import sys 
sys.path.append('/home/hydang/wound_dl_pde/')

import os 
import io 
import numpy as np 
import cv2
import math
#from "wound_dl_pde.pde" import config_loader
from utils.utils import get_logger
logger = get_logger("PDE")
from config_loader import pde_config
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from skimage import measure
from tqdm import tqdm
import pickle


# class Patient():
#     def __init__(self, patient_id):
        
#         """
#         days: List of days consider 
#         list_img: gray img 
#         skin_img: only the skin of image 
#         wound_img: only the wound image
#         comb_img combination of both skin and wound
#         """
#         logger.info("Loaded Patient....{}".format(patient_id))
#         path =  os.path.join(pde_config.data, patient_id)

#         list_img, skin_img, wound_img, cen_mass, comb, days = self.load_config(path, patient_id)
        
#         self.patient_info = {   
#                             "id": patient_id,
#                             "path": path,
#                             "gray_img": list_img,
#                             "skin_img": skin_img,
#                             "wound_img": wound_img,
#                             "cen_mass": cen_mass,
#                             "comb_img": comb,
#                             "days": days,
#                             }
        

#     """
#     Return 2 different images: 
#         - Skin Image
#         - Wound Image
#     """
#     def load_config(self, path, patient_id):
#         list_img = dict()
#         skin_img = dict()
#         wound_img = dict()
#         cen_mass = dict()
#         comb = dict()
#         days = list()
#         paths = os.listdir(path)
#         logger.info("_binary_converted_".format(patient_id))
#         paths = sorted(paths, key = lambda x: x.split("_")[0])
#         for img in tqdm(paths):
#             list_img["day_{}".format(img.split("_")[0])] =  cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(path, img)),(512,512)),cv2.COLOR_RGB2GRAY)
#             skin_img["day_{}".format(img.split("_")[0])],wound_img["day_{}".format(img.split("_")[0])] = self.convert(list_img["day_{}".format(img.split("_")[0])])
#             comb["day_{}".format(img.split("_")[0])] = skin_img["day_{}".format(img.split("_")[0])] + 2*wound_img["day_{}".format(img.split("_")[0])]
#             try:
#                 cen_mass["day_{}".format(img.split("_")[0])] = self.com(wound_img["day_{}".format(img.split("_")[0])])
#             except IndexError:
#                 cen_mass["day_{}".format(img.split("_")[0])] = None
#             days.append(int(img.split("_")[0]))
#         return list_img, skin_img, wound_img, cen_mass, comb, days
#     def convert(self,img):
#         res_skin = np.zeros((img.shape[0],img.shape[1]),dtype ="uint8")
#         res_wound = np.ones((img.shape[0],img.shape[1]),dtype ="uint8")
#         for i in (range(img.shape[0])):
#             for j in range(img.shape[1]):
#                 if(img[i,j] == 33):
#                     res_skin[i,j] = int(0) #if the pixel is background
#                     res_wound[i,j] = int(1)
#                 elif(img[i,j] == 55):
#                     res_wound[i,j] = int(0) #if the pixel is wound
#                 elif(img[i,j] == 81):
#                     res_skin[i,j] = int(0) #if the pixel is background
#                     res_wound[i,j] = int(1) #if the pixel is noise
#                 elif(img[i,j] == 91):
#                     res_skin[i,j] = int(0) #if the pixel is background
#                     res_wound[i,j] = int(1) #if the pixel is noise
#                 elif(img[i,j] == 128):
#                     res_skin[i,j] = int(1) #if the pixel is skin
#         return res_skin,res_wound
#     """
#     Compute the center of mass (only for the wound image)
#     """
#     def com(self,img):
#         props = measure.regionprops(np.array(img)) #get the centroid of each image
#         com = props[0].centroid #store the centroids of images to a dict
# #         print("The center of mass of the wound in image is: {}".format(com))
#         return com

#     """
#     Take the smaller square of the center
#     """
#     def centroid_sq(self,com,imgs): 
#         square = {}
#         for im in imgs.keys():
#             width = imgs[im].shape[0]
#             length = imgs[im].shape[1]
#             img = imgs[im]
#             centroid = com[im]
#             try:
#                 square[im] = img[int(math.floor(centroid[0])-width/3):int(math.floor(centroid[0]) + width/3),int(math.floor(centroid[1])-length/3):int(math.floor(centroid[1]) + length/3)]
#                 #print("The images {} has a small windows with shape:{}".format(im,square[im].shape))
#                 assert(square[im].shape[0] == square[im].shape[1])
#             except TypeError:
#                 square[im] = None
#         return square
    
#     """
#     Methods for showing list of grays images
#     """
# def show(patient):
#     patient_info = patient.patient_info
#     list_img, patient_id = patient_info["wound_img"], patient_info["id"]
#     w=10
#     h=10
#     fig=plt.figure(figsize=(20, 20))

#     columns = 5
#     rows = 5
#     i = 1
#     while i <= (min(columns*rows+1,len(list_img))):
#         img = list_img[list((list_img.keys()))[i-1]]
#         fig.add_subplot(rows,columns,i,title="{}".format(list((list_img.keys()))[i-1]))
#         plt.imshow(img)
#         i+=1
#     fig.suptitle("Patient {}".format(patient_id),fontsize = 16)
#     plt.show()

class Patient():
    def __init__(self, patient_id):
        
        """
        days: List of days consider 
        list_img: gray img 
        skin_img: only the skin of image 
        wound_img: only the wound image
        comb_img combination of both skin and wound
        """
        logger.info("Loaded Patient....{}".format(patient_id))
        path =  os.path.join(pde_config.data, patient_id)
        with open("predicted.pickle","rb") as f:
            self.patient_pickle = pickle.load(f)
        self.p_info = self.patient_pickle[patient_id]
        list_img, skin_img, wound_img, cen_mass, days = self.load_config()
        self.patient_info = {   
                            "id": patient_id,
                            "path": path,
                            "gray_img": list_img,
                            "skin_img": skin_img,
                            "wound_img": wound_img,
                            "cen_mass": cen_mass,
                            "days": days,
                            }
        

    """
    Return 2 different images: 
        - Skin Image
        - Wound Image
    """
    
    def load_config(self):
        list_img = dict()
        skin_img = dict()
        wound_img = dict()
        cen_mass = dict()
        days = list()
        for d in self.p_info:
            if "index" in d: 
                continue
            count = 0
            day = d.split("_")[0]
            days.append(int(day))
            list_img["day_{}".format(day)] = self.p_info[day]
            skin_img["day_{}".format(day)], wound_img["day_{}".format(day)] = self.convert(self.p_info[day])
            try:
                cen_mass["day_{}".format(day)] = self.com(wound_img["day_{}".format(day)])
            except IndexError:
                cen_mass["day_{}".format(day)] = None
        return list_img, skin_img, wound_img, cen_mass, np.unique(sorted(days))
            

            
#     def load_config(self, path, patient_id):
#         list_img = dict()
#         skin_img = dict()
#         wound_img = dict()
#         cen_mass = dict()
#         comb = dict()
#         days = list()
#         paths = os.listdir(path)
#         logger.info("_binary_converted_".format(patient_id))
#         paths = sorted(paths, key = lambda x: x.split("_")[0])
#         for img in tqdm(paths):
#             list_img["day_{}".format(img.split("_")[0])] =  cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(path, img)),(512,512)),cv2.COLOR_RGB2GRAY)
#             skin_img["day_{}".format(img.split("_")[0])],wound_img["day_{}".format(img.split("_")[0])] = self.convert(list_img["day_{}".format(img.split("_")[0])])
#             comb["day_{}".format(img.split("_")[0])] = skin_img["day_{}".format(img.split("_")[0])] + 2*wound_img["day_{}".format(img.split("_")[0])]
#             try:
#                 cen_mass["day_{}".format(img.split("_")[0])] = self.com(wound_img["day_{}".format(img.split("_")[0])])
#             except IndexError:
#                 cen_mass["day_{}".format(img.split("_")[0])] = None
#             days.append(int(img.split("_")[0]))
#         return list_img, skin_img, wound_img, cen_mass, comb, days
    def convert(self,img):
        res_skin = np.zeros((img.shape[0],img.shape[1]),dtype ="uint8")
        res_wound = np.ones((img.shape[0],img.shape[1]),dtype ="uint8")
        for i in (range(img.shape[0])):
            for j in range(img.shape[1]):
                if(img[i,j] == 0):
                    res_skin[i,j] = int(0) #if the pixel is background
                    res_wound[i,j] = int(1)
                elif(img[i,j] >= 2):
                    res_wound[i,j] = int(0) #if the pixel is wound
                elif(img[i,j] == 1):
                    res_skin[i,j] = int(1) #if the pixel is skin
        return res_skin,res_wound
    """
    Compute the center of mass (only for the wound image)
    """
    def com(self,img):
        props = measure.regionprops(np.array(img)) #get the centroid of each image
        com = props[0].centroid #store the centroids of images to a dict
#         print("The center of mass of the wound in image is: {}".format(com))
        return com

    """
    Take the smaller square of the center
    """
    def centroid_sq(self,com,imgs): 
        square = {}
        for im in imgs.keys():
            width = imgs[im].shape[0]
            length = imgs[im].shape[1]
            img = imgs[im]
            centroid = com[im]
            try:
                square[im] = img[int(math.floor(centroid[0])-width/3):int(math.floor(centroid[0]) + width/3),int(math.floor(centroid[1])-length/3):int(math.floor(centroid[1]) + length/3)]
                #print("The images {} has a small windows with shape:{}".format(im,square[im].shape))
                assert(square[im].shape[0] == square[im].shape[1])
            except TypeError:
                square[im] = None
        return square
    
    """
    Methods for showxing list of grays images
    """
def show(patient):
    patient_info = patient.patient_info
    list_img, patient_id = patient_info["wound_img"], patient_info["id"]
    w=10
    h=10
    fig=plt.figure(figsize=(20, 20))

    columns = 5
    rows = 5
    i = 1
    while i <= (min(columns*rows+1,len(list_img))):
        img = list_img[list((list_img.keys()))[i-1]]
        fig.add_subplot(rows,columns,i,title="{}".format(list((list_img.keys()))[i-1]))
        plt.imshow(img)
        i+=1
    fig.suptitle("Patient {}".format(patient_id),fontsize = 16)
    plt.show()