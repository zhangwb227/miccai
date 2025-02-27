import os
from options.train_options import TrainOptions
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import utils
import BatchDataReader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from models.AUNet import AUNet
from models.IGUNet import IGUNet
from models.IGAVNet import IGAVNet
from models.IGAVNet18 import IGAVNet18
from models.IGAVNet72 import IGAVNet72

# 损失函数对比实验
from models.ComparisonLoss import CF_Loss,CE_DC_Loss,CUR_Loss,VE_Loss

def train_net(net,device):
    #train setting
    val_num = opt.val_ids[1] - opt.val_ids[0]
    best_valid_miou=0
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    # Read Data
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,is_dataaug=False)
    print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    testid_dataset = BatchDataReader.CubeDataset(opt.dataroot_test, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    testid_loader = torch.utils.data.DataLoader(testid_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"数据加载完成！！！\ttrain: {len(train_dataset)}\tvalid{len(valid_loader)}")
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)

    if opt.loss_name == 'CF':
        now_loss = CF_Loss(opt.data_size,beta=1.1,alpha=0.5,gamma=0.08)
    elif opt.loss_name == 'CE_DC':
        now_loss = CE_DC_Loss()
    elif opt.loss_name == 'CUR':
        now_loss = CUR_Loss()
    elif opt.loss_name == 'VE':
        now_loss = VE_Loss(device=device,alpha=1.0,beta=0.6,mu=0.5)

    #Start train
    for epoch in range(1, opt.num_epochs + 1):
        net.train()
        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        
        for itr, (train_images, train_annotations, name) in pbar:
            train_images =train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)

            pred= net(train_images)

            loss = now_loss(pred, train_annotations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Start Val
        with torch.no_grad():
            # Save model
            val_miou_sum = 0
            net.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
            for itr, (test_images, test_annotations, cubename) in pbar:
                test_images = test_images.to(device=device, dtype=torch.float32)
                test_annotations = test_annotations.cpu().detach().numpy()
                pred = net(test_images)
                pred_argmax = torch.argmax(pred, dim=1)
                result= np.squeeze(pred_argmax).cpu().detach().numpy()
                val_miou_sum += utils.cal_miou(result, test_annotations)
            val_miou = val_miou_sum / val_num
            print("Step:{}, Valid_mIoU:{}".format(epoch, val_miou))
            # save best model
            if val_miou > best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                temp = temp.replace(".","_")+"_"+str(epoch)+".pth"
                torch.save(net.state_dict(), os.path.join(best_model_save_path, temp))
                logging.info(f'Checkpoint {epoch} saved !')
                best_valid_miou = val_miou
        model_name = str(epoch)+".pth"
        torch.save(net.state_dict(), os.path.join(best_model_save_path, model_name))

        



if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    logging.info(f'Using device {device}')
    #loading network
    # net = IGUNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet18(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet72(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    net = AUNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
        
    if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        net = torch.nn.DataParallel(net)#多gpu训练,自动选择gpu
        
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    #input the model into GPU
    net.to(device=device)
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




