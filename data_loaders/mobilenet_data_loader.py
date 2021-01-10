from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image as PILImage
import torchvision.transforms as transforms
from time import time

def get_file_index_list(path):
    return  [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def get_classes(path):
    return [f for f in os.listdir(path)]


class ImageDataset(Dataset): 
    

    def __init__(self, root_dir, index_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.classes = get_classes(root_dir)
        self.root_dir = root_dir
        self.index_list = index_list
        self.transform = transform
        self.file_index_list = {}
        self.file_list_size = {}
        self.total_elem = 0
        
        for class_id,cls in enumerate(self.classes):
            self.file_index_list[class_id] = get_file_index_list(self.root_dir + "/" +  cls)
            self.file_list_size[class_id] = len(self.file_index_list[class_id])
            self.total_elem += self.file_list_size[class_id]


    def __len__(self):
        return len(self.index_list)
    
    def get_class_id_file_id(self, idx):
        index_value = self.index_list[idx]
        count = 0
        for i in range(len(self.classes)):
            if index_value < count + self.file_list_size[i]:
                class_id = i
                break
            else:
                count = count + self.file_list_size[i]
            
        file_id = index_value - count
        return class_id, file_id

    def __getitem__(self, idx):        
        
        #print("file_index_list: ",indx_list)
        valid = False
        
        while not valid:
            class_id, file_id = self.get_class_id_file_id(idx)
            indx_list =  self.file_index_list[class_id]
            img = None
            #print("idx=", idx, " class_id=",  class_id, " file_id: ",file_id, "count: ", count)
        
            try:
                img = PILImage.open(self.root_dir + "/" + self.classes[class_id] + "/" + indx_list[file_id])
            
            except:
                #print("Train Excetpion caught while opening file:" + self.root_dir + "/" + self.classes[class_id] + "/" + indx_list[file_id])
                #print("Error:", sys.exc_info()[0])
                pass
            
            if img is not None:        
                if self.transform is not None:
                    img = self.transform(img)
                    #print("Inside transform")
                    #print(" img shape: ", img.shape)
                
            if img is None or img.shape[0] != 3 or img.shape[1] != 224 or img.shape[2] != 224:
                #print(classes[class_id] + "/" + indx_list[file_id] + " is not good")
                #file_id = (file_id + 1) %  self.file_list_size[class_id]
                idx += 1
            else: 
                valid = True       
  
        
        return {"X" : img, "Y": class_id}



