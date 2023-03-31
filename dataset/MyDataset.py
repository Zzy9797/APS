# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
import pandas as pd
import random
import cv2

class MyDataset(data.Dataset):
    def __init__(self, path, transform=None,phase='train'):
        self.path = path
        self.transform = transform
        self.phase=phase
        self.classes=len(os.listdir(self.path))
        self.images=[]
        self.label=[]
        for i in range(self.classes):
            self.img_dir=os.path.join(self.path,str(i))
            self.imageslist=os.listdir(self.img_dir)
            for item in self.imageslist:
                # item_withlabel=str(i)+'-'+item
                self.images.append(os.path.join(self.path,str(i),item))
                self.label.append(i)          

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label_7 = self.label[idx]
        image = cv2.imread(self.images[idx])
        name=self.images[idx].split('/')[-2]+'/'+self.images[idx].split('/')[-1]

        if self.transform is not None:
            image = self.transform(image)
        if self.phase=='test':
            return image,label_7,name
        return image, label_7
    def get_labels(self):
        return self.label
