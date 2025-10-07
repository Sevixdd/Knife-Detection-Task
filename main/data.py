from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2
import pyarrow.parquet as pq

# create dataset class
# class knifeDataset(Dataset):
#     def __init__(self,images_df,mode="train"):
#         self.images_df = images_df.copy()
#         self.images_df.Id = self.images_df.Id
#         self.mode = mode

#     def __len__(self):
#         return len(self.images_df)

#     def __getitem__(self,index):
#         X,fname = self.read_images(index)
#         if not self.mode == "test":
#             labels = self.images_df.iloc[index].Label
#         else:
#             y = str(self.images_df.iloc[index].Id.absolute())
#         if self.mode == "train":
#             X = T.Compose([T.ToPILImage(),
#                     T.Resize((config.img_weight,config.img_height)),
#                     T.ColorJitter(brightness=0.2,contrast=0,saturation=0,hue=0),
#                     T.RandomRotation(degrees=(0, 180)),
#                     T.RandomVerticalFlip(p=0.5),
#                     T.RandomHorizontalFlip(p=0.5),
#                     T.ToTensor(),
#                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
#         elif self.mode == "val":
#             X = T.Compose([T.ToPILImage(),
#                     T.Resize((config.img_weight,config.img_height)),
#                     T.ToTensor(),
#                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
#         return X.float(),labels, fname

#     def read_images(self,index):
#         row = self.images_df.iloc[index]
#         filename = str(row.Id)
#         im = cv2.imread(filename)[:,:,::-1]
#         return im, filename


class knifeDataset(Dataset):
    def __init__(self, parquet_file, mode="train"):
        self.parquet_file = parquet_file
        self.parquet_data = self.read_parquet(parquet_file)
        self.mode = mode

    def read_parquet(self, parquet_file):
        parquet_data = pq.read_table(parquet_file)
        # Assuming 'Label' and 'Id' columns are present in the Parquet file
        labels = parquet_data.column('Label').to_pylist()
        ids = parquet_data.column('Id').to_pylist()
        return list(zip(ids, labels))

    def __len__(self):
        return len(self.parquet_data)

    def __getitem__(self, index):
        img_id, label = self.parquet_data[index]
        X, fname = self.read_images(img_id)
        
        if not self.mode == "test":
            labels = label
        else:
            y = str(img_id.absolute())
            
        if self.mode == "train":
            X = T.Compose([T.ToPILImage(),
                           T.Resize((config.img_weight, config.img_height)),
                           T.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
                           T.RandomRotation(degrees=(0, 180)),
                           T.RandomVerticalFlip(p=0.5),
                           T.RandomHorizontalFlip(p=0.5),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "val":
            X = T.Compose([T.ToPILImage(),
                           T.Resize((config.img_weight, config.img_height)),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        
        return X.float(), labels, fname

    def read_images(self, img_id):
        # Assuming the image files are named as per their ids
        filename = str(img_id)
        im = cv2.imread(filename)[:,:,::-1]
        return im, filename

