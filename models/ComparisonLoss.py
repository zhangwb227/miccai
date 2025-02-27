import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
import cv2 as cv
from models.ComparisonUtils import soft_skel
from utils import DiceLoss
from scipy.ndimage import distance_transform_edt

def encode_mask(ground_truth,prediction):
    encode_tensor=F.one_hot(ground_truth.to(torch.int64), num_classes=3)
    encode_tensor=encode_tensor.permute(0, 3, 1, 2).contiguous()     
    encode_tensor = encode_tensor.to(device=torch.device('cuda'), dtype=torch.float32)
    masks_pred_softmax = F.softmax(prediction,dim=1)
    return encode_tensor,masks_pred_softmax



class CF_Loss(nn.Module):

    def __init__(self, img_size,beta,alpha,gamma):
        super(CF_Loss, self).__init__()

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.p = torch.tensor(img_size[-1], dtype = torch.float)
        self.n = torch.log(self.p)/torch.log(torch.tensor([2]).to('cuda'))
        self.n = torch.floor(self.n)
        self.sizes = 2**torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)
        self.CE = nn.CrossEntropyLoss()
    
    def get_count(self,sizes,p,masks_pred_softmax):
    
        counts = torch.zeros((masks_pred_softmax.shape[0], len(sizes),2))
        index = 0

        for size in sizes:

            stride = (size, size)
            pad_size = torch.where((p%size) == 0, torch.tensor(0,dtype = torch.int), (size - p%size).to(dtype=torch.int))
            pad = nn.ZeroPad2d((0,pad_size, 0, pad_size))
            pool = nn.AvgPool2d(kernel_size = (size, size), stride = stride)

            S = pad(masks_pred_softmax)
            S = pool(S)
            S = S*((S> 0) & (S < (size*size)))
            counts[...,index,0] = (S[:,0,...] - S[:,2,...]).abs().sum()/(S[:,2,...]>0).sum()
            counts[...,index,1] = (S[:,1,...] - S[:,3,...]).abs().sum()/(S[:,3,...]>0).sum()        

            index += 1

        return counts

    def forward(self, prediction, ground_truth):
        

        encode_tensor,masks_pred_softmax = encode_mask(ground_truth,prediction)
        
        loss_CE = self.CE(prediction, ground_truth)
        
        Loss_vd = (torch.abs(masks_pred_softmax[:,1,...].sum()-encode_tensor[:,1,...].sum())+torch.abs(masks_pred_softmax[:,2,...].sum()-encode_tensor[:,2,...].sum()))/(masks_pred_softmax.shape[0]*masks_pred_softmax.shape[2]*masks_pred_softmax.shape[3])
        
        masks_pred_softmax = masks_pred_softmax[:,1:3,...]
        encode_tensor = encode_tensor[:,1:3,...]
        masks_pred_softmax = torch.cat((masks_pred_softmax, encode_tensor), 1)
        counts = self.get_count(self.sizes,self.p,masks_pred_softmax)

        artery_ = torch.sqrt(torch.sum(self.sizes*((counts[...,0])**2)))
        vein_ = torch.sqrt(torch.sum(self.sizes*((counts[...,1])**2)))
        size_t = torch.sqrt(torch.sum(self.sizes**2))
        loss_FD = (artery_+vein_)/size_t/masks_pred_softmax.shape[0]
        
        loss_value = self.beta*loss_CE + self.alpha*loss_FD + self.gamma*Loss_vd
        
        return loss_value

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred)[:,1:,...])
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true[:,1:,...]) + torch.sum(y_pred[:,1:,...]) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        
        y_true,y_pred = encode_mask(y_true,y_pred)
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice

    

"""
交叉熵损失函数和DC损失函数
"""
class CE_DC_Loss(nn.Module):
    def __init__(self, alpha=1.0,bite=0.6):
        super(CE_DC_Loss, self).__init__()
        self.alpha = alpha
        self.bite = bite
        self.Loss_CE = nn.CrossEntropyLoss()
        self.Loss_DSC= DiceLoss()

    def forward(self, pred, lab):
        return self.alpha*self.Loss_CE(pred, lab)+self.bite*self.Loss_DSC(pred, lab)

"""
曲率损失函数
Multi-Scale Pathological Fluid Segmentation in OCT With a Novel Curvature Loss in Convolutional Neural Network
"""
"""
分类动静脉，返回动静脉tensor
参数列表：
    lab:真实\预测标签, H*W
    num_class:类别数, int
返回值：
    lab_tensor：转换后的tensor
        维度：b*n*H*W, b:batch size, n:num_class
"""
def getLabTensor(lab, num_class):
    lab_tensor = torch.zeros((1,num_class-1,lab.shape[0],lab.shape[1]),dtype=torch.float32)
    for i in range(1, num_class):
        now_lab = lab.clone()
        now_lab[now_lab!=i]=0
        now_lab[now_lab==i]=1
        lab_tensor[:,i-1,:,:] = now_lab.unsqueeze(0).unsqueeze(0)
    return lab_tensor

class CUR_Loss(nn.Module):
    def __init__(self, num_class=3, e=1e-5, alpha=0.5):
        super(CUR_Loss, self).__init__()
        self.num_class = num_class
        self.e = e
        k = (torch.tensor([[-1,5,-1],[5,-16,5],[-1,5,-1]])/16).unsqueeze(0).unsqueeze(0)
        for i in range(1, num_class-1):
            k = torch.cat((k, k),dim=1)
        self.k = k
        self.Loss_CE = nn.CrossEntropyLoss()
        self.Loss_DSC= DiceLoss()
        self.alpha = alpha
    def forward(self, pred, lab):
        cur_sum = 0
        pred_softmax = torch.argmax(pred,dim=1)
        for i in range(pred_softmax.shape[0]):
            pred_tensor = getLabTensor(pred_softmax[i],self.num_class)
            lab_tensor = getLabTensor(lab[i],self.num_class)
            
            pred_ol = F.conv2d(pred_tensor, self.k, bias=None, stride=1, padding=1)
            lab_ol = F.conv2d(lab_tensor, self.k, bias=None, stride=1, padding=1)
            
            cur = torch.abs((pred_ol*lab_tensor)/(lab_ol+self.e)).sum()/(pred_softmax.shape[-1]*pred_softmax.shape[-2])
            cur_sum += cur
        return self.Loss_CE(pred, lab)+self.Loss_DSC(pred, lab)+self.alpha*cur_sum.mean()
    

"""
距离加权损失
"""
"""
分离动静脉
参数列表：
    label:待分离标签，H*W
    num_class:标签总数
返回值列表：
    bav:num_class*H*W的分离结果
"""
def getBAV(label,num_class):
    # 构造一个num_class*H*W维度的全0张量
    bav = torch.zeros((num_class,label.shape[0],label.shape[1]))
    # 标签0特殊分离
    now_lab = label.clone()
    now_lab[now_lab!=0]=255
    now_lab[now_lab==0]=1
    now_lab[now_lab==255]=0
    bav[0] = now_lab.clone()
    # 分离剩余的标签
    for i in range(1,num_class):
        now_lab = label.clone()
        # 不等于i的等于0，等于i的等于1
        now_lab[now_lab!=i]=0
        now_lab[now_lab==i]=1
        bav[i] = now_lab.clone()
    return bav
"""
根据距离计算权重并归一化权重
参数列表：
    lab_bav:真实标签动静脉分离结果
    pred_bav:预测标签动静脉分离结果
返回值列表：
    lab_bav:真实标签归一化距离权重
    pred_bav:预测标签归一化真实权重
"""
def getVE(lab_bav,pred_bav,e=1e-5):
    
    for i in range(lab_bav.shape[0]):
        lab_bav[i] = torch.FloatTensor(distance_transform_edt(lab_bav[i]))
        lab_bav[i] = (lab_bav[i]/(torch.max(lab_bav[i])+e))
    
    for i in range(pred_bav.shape[0]):
        pred_bav[i] = torch.FloatTensor(distance_transform_edt(pred_bav[i]))
        pred_bav[i] = (pred_bav[i]/(torch.max(pred_bav[i])+e))
    return lab_bav, pred_bav

class VE_Loss(nn.Module):
    def __init__(self, device, num_class=3,alpha=1.0,beta=0.6,mu=0.5):
        super(VE_Loss, self).__init__()
        self.num_class = num_class
        self.device = device
        self.Loss_CE = nn.CrossEntropyLoss()
        self.Loss_DSC= DiceLoss()
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
    def forward(self, pred, lab):
        pred_softmax = torch.argmax(pred,dim=1)
        loss_sum = torch.zeros(1, requires_grad=True).to(device=self.device)
        for i in range(pred_softmax.shape[0]):
            
            lab_bav = getBAV(pred_softmax[i],self.num_class)
            pred_bav = getBAV(lab[i],self.num_class)
            
            lab_bav, pred_bav = getVE(lab_bav, pred_bav)
            
            lab_one_hot = F.one_hot(lab[i].to(torch.int64),self.num_class).permute(2,0,1)
            pred_one_hot = F.one_hot(pred_softmax[i].to(torch.int64),self.num_class).permute(2,0,1)
            
            lab_feat = torch.cat((torch.sum(lab_bav,dim=0,keepdim=True).to(device=self.device),lab_one_hot),dim=0)
            pred_feat = torch.cat((torch.sum(pred_bav,dim=0,keepdim=True).to(device=self.device),pred_one_hot),dim=0)
            
            pred_result = pred_feat[0][lab[i]>0].unsqueeze(-1)
            for j in range(1,pred_feat.shape[0]):
                pred_result = torch.cat((pred_result,pred_feat[j][lab[i]>0].unsqueeze(-1)),dim=-1)

            lab_result = lab_feat[0][lab[i]>0].unsqueeze(-1)
            for j in range(1,lab_feat.shape[0]):
                lab_result = torch.cat((lab_result,lab_feat[j][lab[i]>0].unsqueeze(-1)),dim=-1)
            
            loss_sum = loss_sum+1-torch.cosine_similarity(pred_result,lab_result,dim=-1).mean()

        return self.alpha*self.Loss_CE(pred, lab)+self.beta*self.Loss_DSC(pred, lab)+self.mu*loss_sum