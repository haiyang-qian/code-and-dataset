from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy
from matplotlib import pyplot as plt

class CornDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.label_dir = os.listdir(self.root_dir)
        self.img_path = []
        self.encodemap = {"H":0,"SCLB":1,"SR":2,"GLS":3}
        self.transform = transforms.Compose(transforms.RandomRotation(45))
        for label in self.label_dir:
            for img in os.listdir(os.path.join(self.root_dir,label)):
                self.img_path.append((img,label));


    def __getitem__(self, idx):
        img_item_path  =os.path.join(self.root_dir,self.img_path[idx][1],self.img_path[idx][0])
        img = cv2.imread(img_item_path)

        # dim = (224,224)
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)
        # img = cv2.resize(img,(3,2000,2000))

        label = self.img_path[idx][1]
        label = int(self.encodemap[label])
        return img , label
    def __len__(self):
        return len(self.img_path)
