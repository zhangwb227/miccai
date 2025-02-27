import os
from options.train_options import TrainOptions
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

import natsort
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import cv2
# import random
from skimage import transform
class CubeDataset(Dataset):
    def __init__(self, data_dir,data_id,data_size,modality,is_dataaug=False):
        self.is_dataaug=is_dataaug
        self.datanum=data_id[1]-data_id[0]
        self.modality=modality
        self.data_size=data_size
        self.modalitynum=len(modality)-1
        self.datasetlist={'data':{},'label':{}}
        for modal in modality:
            if modal != modality[-1]:
                self.datasetlist['data'].update({modal: {}})
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                print(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['data'][modal].update({img: {}})
                    imgadress= os.path.join(data_dir, modal, img)
                    self.datasetlist['data'][modal][img] = imgadress
            else:
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['label'].update({img: {}})
                    labeladdress = os.path.join(data_dir, modal, img)
                    self.datasetlist['label'][img] = labeladdress

    def __getitem__(self, index):
        data=np.zeros((3,self.data_size[0],self.data_size[1]))
        label = np.zeros((self.data_size[0],self.data_size[1]))
        for i,modal in enumerate(self.modality):
            if modal != self.modality[-1]:
                name=list(self.datasetlist['data'][modal])[index]
                data[:,:,:] = cv2.imread(self.datasetlist['data'][modal][name]).transpose(2,0,1).astype(np.float32)
            else:
                name = list(self.datasetlist['label'])[index]
                label[:,:]=cv2.imread(self.datasetlist['label'][name], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        data = torch.from_numpy(np.ascontiguousarray(data))
        label = torch.from_numpy(np.ascontiguousarray(label))
        return data,label,name

    def __len__(self):
        return self.datanum