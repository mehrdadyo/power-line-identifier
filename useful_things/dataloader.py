import torch
import numpy as np
from skimage import io 
from torch.utils.data import Dataset
import os
from transforms.hog import *
from transforms.hough import *
from torchvision import datasets, transforms

import shutil

def removeAllFiles(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

class UtilityDataset(Dataset):
    'Defines a utility dataset for classifiying streetview images'
    def __init__(self, labels_ids, use_Image=True, use_HoG=False, use_HOUGH=False, setName='none', performTransforms=False):
        self.labels_ids = labels_ids
        self.use_Image = use_Image
        self.use_HoG = use_HoG
        self.use_HOUGH = use_HOUGH
        self.setName = setName
        
        if performTransforms:
            self.saveTransforms()
        
    def saveTransforms(self):
        save_folder = 'data/' + self.setName + '/'
        removeAllFiles(save_folder + 'img/')
        removeAllFiles(save_folder + 'hog/')
        removeAllFiles(save_folder + 'hough/')
        
        for label, ID in self.labels_ids:
            if label == 0:
                category = 'noUtility'
            elif label == 1:
                category = 'noVeg'
            else:
                category = 'veg'
            im_folder = 'images/sorted_data/' + category + '/'
                
            filename = os.listdir(im_folder)[ID]
            img_np = io.imread(im_folder+filename).astype(np.uint8)
            
            img_hog = hog_transform(img_np).float().reshape(-1,224,224)
            img_hough =  hough_transform(img_np).float().reshape(-1,224,224)
            img_pt = torch.from_numpy(img_np).float().permute(2,0,1)
                
            torch.save(img_pt,save_folder + 'img/' + category + '_' + str(ID) + '.pt')
            torch.save(img_hog,save_folder + 'hog/' + category + '_' + str(ID) + '.pt')
            torch.save(img_hough,save_folder + 'hough/' + category + '_' + str(ID) + '.pt')
        
    def __len__(self):
        return len(self.labels_ids)
    
    def __getitem__(self,index):
        label, ID = self.labels_ids[index]
        folder = 'data/' + self.setName + '/'
            
        if label == 0:
            category = 'noUtility_'
        elif label == 1:
            category = 'noVeg_'
        else:
            category = 'veg_'
            
        if self.use_Image:    
            img = torch.load(folder + '/img/' + category + str(ID) + '.pt')
        if self.use_HoG:
            hog = torch.load(folder + '/hog/' + category + str(ID) + '.pt')
            try:
                img = torch.cat((img,hog), dim=0)
            except UnboundLocalError:
                img = hog
        if self.use_HOUGH:
            hough = torch.load(folder + '/hough/' + category + str(ID) + '.pt')
            try:
                img = torch.cat((img,hough), dim=0)
            except UnboundLocalError:
                img = hough
        
        return img, label
    '''
    def __getitem__(self,index):
        'Generates sample of data' 
        # Select samples
        label, ID = self.labels_ids[index]
        
        if label == 0:
            folder = 'images/sorted_data/noUtility/'
        elif label == 1:
            folder = 'images/sorted_data/noVeg/'
        else:
            folder = 'images/sorted_data/veg/'
        
        filename = os.listdir(folder)[ID]
        name_split = filename.split('_')
        
        lat = name_split[1]
        long = name_split[2]
        cat = name_split[4]
        
        img = torch.from_numpy(io.imread(folder+filename).astype(np.uint8)).float().permute(2,0,1)

        return img, label, lat, long, cat
        '''
    