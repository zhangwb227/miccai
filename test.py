import os
from options.train_options import TrainOptions
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
import torch
import logging
import sys
import numpy as np
from options.test_options import TestOptions
import cv2
import BatchDataReader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

from models.AUNet import AUNet
from models.IGAVNet import IGAVNet
from models.IGUNet import IGUNet
from models.IGAVNet18 import IGAVNet18
from models.IGAVNet72 import IGAVNet72
def test_net(net,device):

    test_dataset = BatchDataReader.CubeDataset(opt.dataroot_test, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_results = os.path.join(opt.saveroot, 'test_results')
    BGR=np.zeros((opt.data_size[0],opt.data_size[1],3))
    result = np.zeros((opt.data_size[0],opt.data_size[1]))
    net.eval()

    #test set
    pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
    for itr, (test_images,train_annotations,cubename) in pbar:
        test_images = test_images.to(device=device, dtype=torch.float32)

        pred= net(test_images)
        
        pred_argmax = torch.argmax(pred, dim=1)
        pred_argmax = pred_argmax.cpu().detach().numpy()

        cv2.imwrite(os.path.join(test_results, cubename[0]),pred_argmax[0,:,:])
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_softmax = pred_softmax.cpu().detach().numpy()
        BGR[:, :, 0] = 0 * pred_softmax[0, 0, :, :] + 0 * pred_softmax[0, 1, :, :] + 255 * pred_softmax[0, 2, :, :]
        BGR[:, :, 1] = 0 * pred_softmax[0, 0, :, :] + 0 * pred_softmax[0, 1, :, :] + 0 * pred_softmax[0, 2, :, :]
        BGR[:, :, 2] = 0 * pred_softmax[0, 0, :, :] + 255 * pred_softmax[0, 1, :, :] + 0 * pred_softmax[0, 2, :, :]
        cv2.imwrite(os.path.join(opt.saveroot, 'test_visuals', cubename[0]), BGR)


if __name__ == '__main__':
    # setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # loading options
    opt = TestOptions().parse()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    logging.info(f'Using device {device}')
    # loading network
    net = IGUNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet18(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = IGAVNet72(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
    # net = AUNet(in_size=opt.in_channels,out_size=opt.channels,n_classes=opt.n_classes,middle_size=opt.channels,device=device)
     
    
    #load trained model
    restore_path = os.path.join(opt.saveroot, 'best_model', '0_490603_59.pth')
    # 加载多GPU训练模型
    # state_dict = torch.load(restore_path, map_location=device)
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)

    # 加载单GPU训练的模型
    # print(net)
    net.load_state_dict(torch.load(restore_path, map_location=device))
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
