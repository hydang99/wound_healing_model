# Wound Segmentation Project Using Partial Differential Equations Deep Learning
In this project, we want to segment the wound images using Deep Learning Models. 
Purpose: 
- In SOA researchs, there are many researches focusing on Wound Segmentation Using Deep Learning and achieving significant results. 
- In this project, we want to adopt some of the segmentation models (in this case, U-net and Residual U-net). 
## Datasets
- The dataset is supported by [Department of Mathematics (Texas Christian University)](https://cse.tcu.edu/mathematics/index.php) - Confidential 
- To simplize this project, we divide it into three main steps below and we will analyze details into those.
## Models
- In this project, we use U-Net as the baseline model for segmentation tasks. 
<img src="Images/unet.png" width="1000">

## 1. Traning
Purpose: 
- Initializing the models 
- Get the weights of the deep learning model 
Note: 
- Because there are three models need to be trained then we need to run train 3 times including path to three types of dataset (need to be fixed) 
### Input
Necessary Information:
- Images of the wound (In this program, the default place is "dataset/train/" + training images path)
- Ground truth images of the wound (In this program, the default place is "dataset/train/" + ground truth images path)
Parameters: 
- Path to the dataset (train_src), (path_src)
- Path to model save directory (saved_model)
- Note: Need to update (more arguments for batch_size, epoches, learning_rate,...)
```
python3 train.py --train_src dataset/train/train_wound \
--mask_src dataset/train/mask_wound \
--saved_model model_save
```
### Output
- The weights of the deep learning model. 
  - In this problem, we want three weights (wound, skin (not including the wound), both (skin including the wound))
  - We want to compare the performance between skin_model not including the wound and both_model including the wound.
- The weights are saved default into model_save + name of the model

## 2. Predicting
Purpose: 
- Get all the models to predict the directory of images 
- Combine images with wound and skin 
### Input
Necessary Information: 
- The path to three models that were trained by ```train.py``` 
Parameters: 
- ```'predict_src':```: The directory contain images to predict
- ```'predict_save'```: The directory contain images after predicting
- ```'wound_model'```: Path to wound model
- ```'skin_model'```: Path to skin model
- ```'both_model'```: Path to both model
```
python3 predict.py --predict_src dataset/test/test_img \
--predict_save dataset/prediction \
--wound_model model_save/wound_model/ \
--skin_model model_save/skin_model/ \
--both_model model_save/both_com_model/
```
### Output
- The numpy array of combining images (after predicting) 
<p float="left">
  <img src="Images/image.png" width="300" />
  <img src="Images/image2.png" width="300" /> 
</p>

## 3. Mappler
Purpose: 
- Because all of the images are taken with different angles and scales, then we want to map all of those into the same scale and angle
### Input
-```'align_path'```: The directory contain images to transform,
```
python3 align.py --align_path dataset/label/
```
### Output
- The list of rotation and scale to match the original image in the align_path directory
