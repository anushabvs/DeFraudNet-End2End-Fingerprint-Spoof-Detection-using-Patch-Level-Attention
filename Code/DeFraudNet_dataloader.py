###DeFraudNet Dataloader##
from __future__ import division
from __future__ import print_function
import numpy.random as rng
import numpy as np
import pdb
import os,sys
import pickle
import matplotlib.pyplot as plt
from PIL import Image

#Import Torch Libraries

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


def rgb_im(image):
	image = image.transpose((1, 2, 0))
	image = np.dstack([image.astype(np.uint8)]*3)
	image = image.transpose((2,0, 1))
	return image

def patches(img,label):
	img_array = [img]
	count = 0
	w=h= 96
	for y in range(0, img.shape[0], 20):
		for x in range(0, img.shape[1], 20):
			crop_img = img[y:y+h, x:x+w, :]
			crop_img_shape = crop_img.shape
			r = crop_img_shape[0] % crop_img_shape[1]
			s = np.sum(crop_img.ravel())
			if s == 0 or r != 0 or crop_img_shape[0] !=w :
				continue
				
			elif s!=0 and crop_img_shape == (96,96,3):
				count+=1
				img_array.append(crop_img)
	p = img_array[1:]
	
        for num, x in enumerate(p):
		plt.subplot(7,7,num+1)
		plt.axis('off')
		plt.imshow(x)
	plt.show()
        
	label_array = [label]*len(img_array)
	return img_array,label_array

#path consists of training or testing directory##
class fingerprint_data(Dataset):
    def __init__(self, path, transform=None,patch_transform = None):
        self.data = {}
	self.classes = {}
	self.authentication = {}
	self.classes_no = {}
	self.auth_no = {}
	self.categories = {}
        file_path = os.path.join(path,"image_type_classes.txt" )
        with open(file_path,'rb') as f:
            self.classes = pickle.load(f)
        file_path = os.path.join(path,"image_names.txt" )
        with open(file_path,'rb') as f:
            self.categories = pickle.load(f)
        file_path = os.path.join(path,"image_name_classes.txt" )
        with open(file_path,'rb') as f:
            self.authentication = pickle.load(f)
        file_path = os.path.join(path,"images.npy" )
        with open(file_path,'rb') as f:
            self.data= np.load(f)
        file_path = os.path.join(path,"image_type_class_no.npy" )
        with open(file_path,'rb') as f:
            self.classes_no = np.asarray(np.load(f))
        file_path = os.path.join(path,"image_auth_class_no.npy" )
        with open(file_path,'rb') as f:
            self.auth_no = np.load(f)
        
        self.transform = transform
        self.patch_transform = patch_transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be careful for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        img = self.data[index].reshape(224,224,3)
        label = self.classes_no[index]
	imgs,labels = patches(img,label)
	if self.transform is not None:
		imgs[0] = self.transform(imgs[0])
	if self.patch_transform is not None:
	    for i in range(1,len(imgs)):
	    	imgs[i] = self.patch_transform(imgs[i])
        # Return image and the label arrays
        return (imgs, labels)


