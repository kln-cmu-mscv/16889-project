import numpy as np
import os

import torch
import torch.nn
from PIL import Image
from torch.utils.data import Dataset

import random
import torchvision.transforms as transforms

import scipy.io

class PoseDataset(Dataset):
    def __init__(self, split='train', image_size=224, data_dir='16889_pose_dataset'):
        super().__init__()
        self.split      = split     #'train' or 'test'
        self.data_dir   = data_dir
        self.size       = image_size
        self.index_list = self.get_index_list()
        self.poses      = self.get_poses()

    def get_poses(self):
        poses_path = os.path.join(self.data_dir, 'pose.npy')
        load_poses = np.load(poses_path)[self.index_list,:]
        poses = {}
        poses['theta'] = torch.from_numpy(load_poses[:,0])
        poses['phi'] = torch.from_numpy(load_poses[:,1])
        poses['r'] = torch.from_numpy(load_poses[:,2])
        return poses

    def get_index_list(self):
        return np.load(os.path.join(self.data_dir,self.split + '_indices.npy'))

    def __len__(self):
        return len(self.index_list)
        
    def __getitem__(self, index):

        findex = self.index_list[index]
        image_name = 'image_{0:06d}.png'.format(findex) 
        fpath = os.path.join(self.data_dir, image_name)
        
        img = Image.open(fpath)

        transform = transforms.Compose([
              transforms.Resize((self.size, self.size)),
              transforms.PILToTensor(),
              transforms.ConvertImageDtype(torch.float),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        og_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float)
          ])

        transformed_img = transform(img)

        item = {}
        item['og_image'] = og_transform(img)
        item['image'] = transformed_img
        item['theta'] = self.poses['theta'][index]
        item['phi'] = self.poses['phi'][index]
        item['r'] = self.poses['r'][index]
        
        return item