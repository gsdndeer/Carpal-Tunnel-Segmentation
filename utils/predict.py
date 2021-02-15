import torch
import os
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm_notebook
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, models, transforms, models, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import random
import albumentations as A
import time
import cv2


# data loader
class carpal_dataset(Dataset):
    def __init__(self, imgs_path, transforms=None):
        # path
        self.images = imgs_path
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.images[index]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if transforms:
            img = transforms(img)
        return img

    def __len__(self):
        return len(self.images)


# load data
def load_data(filepath):
    img_train = []
    img = os.listdir(os.path.join(filepath))
    for i in range(len(img)):
        img_train.append(os.path.join(filepath, str(i)+'.jpg'))
    return img_train


def read_gt(imgfile):
    img = []
    for i in imgfile:
        im = Image.open(i)
        im = np.array(im)
        for x in range(len(im)):
            for y in range(len(im[x])):
                if im[x][y] > 128:
                    im[x][y] = 1
                else:
                    im[x][y] = 0
        img.append(im)
    return img


# transforms
transforms = transforms.Compose([transforms.ToTensor(),])


def predict(img_pre, modelpath):
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load data
    train_data = carpal_dataset(img_pre)
    trainloader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=3, drop_last=True)
    # load model
    model = smp.DeepLabV3Plus(encoder_name="efficientnet-b1", encoder_weights="imagenet", in_channels=3, classes=1, aux_params=dict(pooling='avg', classes=1))
    model = torch.nn.DataParallel(model).to(device)
    model = (torch.load(modelpath, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    # predict
    save_img = []
    with torch.no_grad():
        for data in tqdm(trainloader):
            data = data.to(device)
            output, _ = model(data)
            output = torch.sigmoid(output)
            mask = output.squeeze().cpu().numpy().round() #####為啥要round???
            save_img.append(mask)
    return save_img


def vote(save_img, img_num):
    threshold = 0.7
    array_img = []
    for image in range(img_num):
        array_img.append(np.zeros(save_img[0][0].shape))
        for model in range(5):
            array_img[image] += save_img[model][image]
            array_img[image] += save_img[model][image+img_num]
        for x in range(len(array_img[image])):
            for y in range(len(array_img[image][x])):
                if array_img[image][x][y] > threshold:
                    array_img[image][x][y] = 1
                else:
                    array_img[image][x][y] = 0
    return array_img


def save_pre_img(pred_CT, pred_FT, pred_MN):
    try:
        os.mkdir("./result")
        os.mkdir("./result/CT")
        os.mkdir("./result/FT")
        os.mkdir( "./result/MN")
    except FileExistsError:
        pass
    
    pred_CT = np.array(pred_CT)
    pred_FT = np.array(pred_FT)
    pred_MN = np.array(pred_MN)
    for i in range(len(pred_CT)):
        img_CT = pred_CT[i,:,:]
        img_FT = pred_FT[i,:,:]
        img_MN = pred_MN[i,:,:]
        img_CT[img_CT!=0] = 255 
        img_FT[img_FT!=0] = 255 
        img_MN[img_MN!=0] = 255 
        cv2.imwrite("./result/CT/"+str(i)+".jpg", img_CT)
        cv2.imwrite("./result/FT/"+str(i)+".jpg", img_FT)
        cv2.imwrite("./result/MN/"+str(i)+".jpg", img_MN)


def DC(pred_mask, gt_mask):
    pred_mask = np.array(pred_mask)
    gt_mask = np.array(gt_mask)
    list_dc = []
    mean = 0
    for img in range(len(pred_mask)):
        numerator =  np.sum(np.multiply(pred_mask[img, :, :], gt_mask[img, :, :]))
        denominator = np.sum(pred_mask[img, :, :] +  gt_mask[img, :, :])
        dc = numerator*2/denominator
        mean += dc
        list_dc.append(dc)
    mean = mean/len(pred_mask)
    list_dc.append(mean)
    return list_dc


def Contours(path_img,path_ct,path_ft,path_mn):
    img = cv2.imread(path_img)
    ct = cv2.imread(path_ct)
    ft = cv2.imread(path_ft)
    mn = cv2.imread(path_mn)

    gray_ct = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY) 
    _,binary_ct = cv2.threshold(gray_ct,127,255,cv2.THRESH_BINARY) 
    contours_ct ,_ = cv2.findContours(binary_ct, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    img = cv2.drawContours(img,contours_ct,-1,(255,0,0),2) 

    gray_ft = cv2.cvtColor(ft, cv2.COLOR_BGR2GRAY) 
    _,binary_ft = cv2.threshold(gray_ft,127,255,cv2.THRESH_BINARY) 
    contours_ft ,_ = cv2.findContours(binary_ft, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    img = cv2.drawContours(img,contours_ft,-1,(160,102,211),2) 

    gray_mn = cv2.cvtColor(mn, cv2.COLOR_BGR2GRAY) 
    _,binary_mn = cv2.threshold(gray_mn,127,255,cv2.THRESH_BINARY) 
    contours_mn ,_ = cv2.findContours(binary_mn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(img,contours_mn,-1,(255,255,0),2) 
    return img