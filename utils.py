import os
from options.train_options import TrainOptions
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from sklearn.metrics import confusion_matrix

def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir, name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir', '\'' + dir + '\'', 'is created.')


def cal_Dice(img1, img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i, j] >= 1 and img2[i, j] >= 1:
                I += 1
            if img1[i, j] >= 1 or img2[i, j] >= 1:
                U += 1
    return 2 * I / (I + U + 1e-5)


def cal_acc(img1, img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i, j] == img2[i, j]:
                acc += 1
    return acc / (shape[0] * shape[1])


def cal_miou(img1, img2):
    classnum = img2.max()
    if classnum==0:
        return 0
    iou = np.zeros((int(classnum), 1))
    for i in range(int(classnum)):
        imga = img1 == i + 1
        imgb = img2 == i + 1
        imgi = imga * imgb
        imgu = imga + imgb
        iou[i] = np.sum(imgi) / np.sum(imgu)
    miou = np.mean(iou)
    return miou


def cal_miou_s(img1, img2):
    iou = np.zeros((1, 1))
    for i in range(1):
        imga = img1 == i + 1
        imgb = img2 == i + 1
        imgi = imga * imgb
        imgu = imga + imgb
        iou[i] = np.sum(imgi) / np.sum(imgu)
    miou = np.mean(iou)
    return miou


def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    # 热编码
    result = torch.zeros(shape)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # result.to(device=device)
    result.scatter_(1, input.cpu(), 1)
    # print(f"shape={shape}\tresult{result.shape}")
    # result = F.one_hot(input,shape[1])
    return result

"""
功能：
    计算混淆矩阵参数值
参数列表：
    t_labels：真实标签
    p_labels：预测标签
返回值：
    TP,TN,FP,FN
"""

def getConfusionMatrixInfomation(t_labels, p_labels):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(t_labels, p_labels)
    # print(f"混淆矩阵：\n{conf_matrix}")
    # 计算 TP, TN, FP, FN
    num_classes = len(np.unique(t_labels))
    # 获取对角元素
    TP = np.diag(conf_matrix)

    # 纵向FP
    FP = np.sum(conf_matrix, axis=0) - TP
    # 横向FN
    FN = np.sum(conf_matrix, axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(conf_matrix, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(np.sum(temp))
    return TP,TN,FP,FN

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape = predict.shape
        target = torch.unsqueeze(target, 1)
        # print(f"target={target.shape}")
        target = make_one_hot(target.long(), shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]