import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv

"""
双卷积模块
Dual Conv
    包含两个2d卷积，BN和LeakyReLU激活函数
    输入参数列表：
        in_size：输入通道数
        out_size：输出通道数
        middle_size：中间过程通道数
        rl_inplace: relu是否设置为True
        x:待处理特征图
"""
class DualConv(nn.Module):
    def __init__(self, in_size, out_size, middle_size, rl_inplace):
        super(DualConv, self).__init__()
        
        self.in_size = in_size
        self.out_size = out_size
        self.middle_size = middle_size
        self.rl_inplace = rl_inplace

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_size,self.middle_size,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.middle_size),
            nn.ReLU(inplace=self.rl_inplace),
            
            nn.Conv2d(self.middle_size,self.out_size,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.out_size),
            nn.ReLU(inplace=self.rl_inplace),
            )
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
"""
多选择注意力机制
Multi-feature Selection Attention Mechanism(MSAM)
选择激活两个特征图中置信度最大的特征
从空间维度和通道维度选择
参数列表：
    in_channels：输入通道数
    ratio：MLP缩减比率
    
    d, x：两个特征图
返回值：
    out：选择融合后的特征图
"""
class MSAM(nn.Module):
    def __init__(self, in_size, ratio):
        super(MSAM, self).__init__()
        self.in_size = in_size
        self.ratio = ratio
        
        # 通道选择：自适应池化
        self.max_pooling1 = nn.AdaptiveMaxPool2d(1)
        self.max_pooling2 = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling2 = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(self.in_size, self.in_size // self.ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.in_size // self.ratio, self.in_size, kernel_size=1),
            nn.Sigmoid()
            )
        
        self.conv_MLP = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3,bias=False),
            nn.Sigmoid()
            )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, d, x):
        # 通道维度选择
        d_max = self.max_pooling1(d)
        x_max = self.max_pooling2(x)
        d_avg = self.avg_pooling1(d)
        x_avg = self.avg_pooling2(x)
        
        d_max = self.shared_MLP(d_max)
        x_max = self.shared_MLP(x_max)
        d_avg = self.shared_MLP(d_avg)
        x_avg= self.shared_MLP(x_avg)

        d_x_ma = torch.cat([d_max.unsqueeze(1),d_avg.unsqueeze(1),x_max.unsqueeze(1),x_avg.unsqueeze(1)],dim=1)
        d_x_ma = self.softmax(d_x_ma)
        
        d_ma = d_x_ma[:,0,:,:]+d_x_ma[:,1,:,:]
        x_ma = d_x_ma[:,2,:,:]+d_x_ma[:,3,:,:]
        
        d = d_ma*d
        x = x_ma*x
        
        # 空间维度选择
        d_max,_ = torch.max(d, dim=1, keepdim=True)
        x_max,_ = torch.max(x, dim=1, keepdim=True)
        d_avg = torch.mean(d, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        
        d_max = self.conv_MLP(d_max)
        x_max = self.conv_MLP(x_max)
        d_avg = self.conv_MLP(d_avg)
        x_avg = self.conv_MLP(x_avg)
        
        d_x_ma = torch.cat([d_max.unsqueeze(1),d_avg.unsqueeze(1),x_max.unsqueeze(1),x_avg.unsqueeze(1)],dim=1)
        d_x_ma = self.softmax(d_x_ma)
        
        d_ma = d_x_ma[:,0,:,:]+d_x_ma[:,1,:,:]
        x_ma = d_x_ma[:,2,:,:]+d_x_ma[:,3,:,:]
        
        d = d_ma*d
        x = x_ma*x
        
        return d+x


class GATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=[3, 5, 3]):
        super(GATBlock, self).__init__()
        num_hidden = in_channels*2
        # 初始化第一层
        self.gat1 = GATConv(in_channels, num_hidden, heads[0], activation=F.elu)
        # 当前层的输入特征维度等于上一层的头数量乘中间层输出特征维度
        self.gat2 = GATConv(num_hidden * heads[0], num_hidden, heads[1], activation=F.elu)
        # 最后一层输出
        self.gat3 = GATConv(num_hidden * heads[1], out_channels, heads[2])

    def forward(self, G):
        g, feat = G, G.ndata['nfeature']
        h = self.gat1(g,feat).flatten(1)
        h = self.gat2(g,h).flatten(1)
        h = self.gat3(g,h).mean(1)
        return h

"""
图卷积模块
"""
class GAMBlock(nn.Module):
    # 参数说明：in_channel：输入通道数, growth_rate：通道数增长速度, num_layers：每个块中包括几个层
    def __init__(self, in_channel, device, block_size=36, ratio=4):
        super(GAMBlock, self).__init__()
        self.block_size = block_size
        self.device = device

        self.conv = DualConv(in_channel, in_channel, in_channel,True)

        self.conv_am11 = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(),
            )
        self.max_pooling11 = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling11 = nn.AdaptiveAvgPool2d(1)
        self.MLP11 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel // ratio, in_channel, kernel_size=1),
            nn.Sigmoid()
            )
        self.conv12 = DualConv(in_channel, in_channel, in_channel,True)

        self.conv_am21 = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(),
            )
        self.gat = GATBlock(in_channel, in_channel)
        self.conv22 = DualConv(in_channel, in_channel, in_channel,True)

        self.conv_am_end = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(),
            )
    def forward(self, x):
        # 初次卷积
        x = self.conv(x)

        # 空域特征卷积
        x1 = self.conv_am11(x)
        x1 = x * x1
        x1_avg = self.avg_pooling11(x1)
        x1_max = self.max_pooling11(x1)
        x1_max_avg = x1_max+x1_avg
        x1_max_avg = self.MLP11(x1_max_avg)
        x1 = x1 * x1_max_avg
        x1 = self.conv12(x1)

        # 图卷积支路
        x2 = self.conv_am21(x)
        x2 = x * x2
        # 存储当前批次特征图的图集
        for i in range(x2.shape[0]):
            # 将特征图分块
            ci, xi, yi = x2[i].shape
            img_label = torch.mean(x2[i], dim=0)
            # 将图像分割成块
            img_xblock = xi // self.block_size
            img_yblock = yi // self.block_size
            for xk in range(img_xblock):  # 控制当前图像‘长’可以分为多少个块
                for yk in range(img_yblock):  # 控制当前图像‘宽’可以分为多少个块
                    # 计算图中的边
                    a = img_label[xk * self.block_size:(xk + 1) * self.block_size, yk * self.block_size:(yk + 1) * self.block_size]
                    a = a.reshape(1,-1)
                    a = a.T @ a
                    index = torch.argwhere(a > 0.25)
                    if index.numel()>=2:
                        # 创建图
                        g = dgl.graph((index[:,0], index[:,1]), num_nodes=self.block_size * self.block_size)
                        g = dgl.add_self_loop(g)
                    else:
                        # 创建图
                        g = dgl.graph(([], []), num_nodes=self.block_size * self.block_size)
                        g = dgl.add_self_loop(g)
                    # 计算节点特征
                    f = x2[i][:, xk * self.block_size:(xk + 1) * self.block_size, yk * self.block_size:(yk + 1) * self.block_size]
                    f = f.reshape(ci, self.block_size * self.block_size)
                    # 赋予节点特征
                    g = g.to(device=self.device)
                    g.ndata['nfeature'] = f.permute(1, 0).contiguous()
                    # GATv2计算
                    if yk==0 and xk==0:
                        h = self.gat(g)
                        h = h.unsqueeze(0)
                    else:
                        h1 = self.gat(g)
                        h1 = h1.unsqueeze(0)
                        h = torch.cat((h, h1), dim=0)
            h = h.reshape(xi, yi, -1).permute(2, 0, 1).contiguous()
            if i == 0:
                u = h.unsqueeze(0)
            else:
                u = torch.cat((u, h.unsqueeze(0)), dim=0)
        u = self.conv22(u)
        out = self.conv_am_end(torch.cat((x1, u),dim=1))
        return out

  
class IGAVNet72(nn.Module):
    def __init__(self, in_size, out_size, n_classes, middle_size, device):
        super(IGAVNet72, self).__init__()
        self.in_size = in_size 
        self.out_size = out_size
        self.n_classes = n_classes
        self.middle_size = middle_size


        # 下采样阶段
        self.u_d_conv1 = DualConv(self.in_size, self.out_size, self.middle_size,True)
        self.u_d_maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.u_d_conv2 = DualConv(self.out_size, self.out_size, self.middle_size,True)
        self.u_d_maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fgat3 = GAMBlock(self.out_size,device,block_size=72)
        self.u_d_maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        self.fgat4 = GAMBlock(self.out_size,device,block_size=72)
        self.u_d_maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.fgat5 = GAMBlock(self.out_size,device,block_size=36)
        self.ucenter = DualConv(self.out_size, self.out_size, self.middle_size,True)


        # 上采样阶段
        self.u_up_conv1 = DualConv(self.out_size*2, self.out_size, self.middle_size,True)
        self.u_up_maxpool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.u_up_conv2 = DualConv(self.out_size*2, self.out_size, self.middle_size,True)
        self.u_up_maxpool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.u_up_conv3 = DualConv(self.out_size*2, self.out_size, self.middle_size,True)
        self.u_up_maxpool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.u_up_conv4 = DualConv(self.out_size*2, self.out_size, self.middle_size,True)
        self.u_up_maxpool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.u_up_conv5 = DualConv(self.out_size*2, self.out_size, self.middle_size,True)
        
        # 特征注意力阶段
        self.t_am1 = MSAM(self.out_size,4)
        self.t_am2 = MSAM(self.out_size,4)
        self.t_am3 = MSAM(self.out_size,4)
        self.t_am4 = MSAM(self.out_size,4)
        self.t_am5 = MSAM(self.out_size,4)

        # 最后一层预测结果
        self.endconv = nn.Conv2d(self.out_size, self.n_classes, kernel_size=1,padding=0)
    def forward(self, x):
        # 下采样过程
        u_d_conv1 = self.u_d_conv1(x)
        u_d_maxpool1 = self.u_d_maxpool1(u_d_conv1)
            
        u_d_conv2 = self.u_d_conv2(u_d_maxpool1)
        u_d_maxpool2 = self.u_d_maxpool2(u_d_conv2)
        
        u_d_conv3 = self.fgat3(u_d_maxpool2)
        u_d_maxpool3 = self.u_d_maxpool3(u_d_conv3)

        u_d_conv4 = self.fgat4(u_d_maxpool3)
        u_d_maxpool4 = self.u_d_maxpool4(u_d_conv4)

        u_d_conv5 = self.fgat5(u_d_maxpool4)
            
        # 上采样
        up5 = self.ucenter(u_d_conv5)
        up5_am = self.t_am5(u_d_conv5,up5)
        up5 = torch.cat([up5,up5_am],dim=1)
        up5 = self.u_up_conv1(up5)
            
        up4 = self.u_up_maxpool1(up5)
        up4_am = self.t_am4(u_d_conv4,up4)
        up4 = torch.cat([up4,up4_am],dim=1)
        up4 = self.u_up_conv2(up4)

        up3 = self.u_up_maxpool2(up4)
        up3_am = self.t_am3(u_d_conv3,up3)
        up3 = torch.cat([up3,up3_am],dim=1)
        up3 = self.u_up_conv3(up3)

        up2 = self.u_up_maxpool3(up3)
        up2_am = self.t_am2(u_d_conv2,up2)
        up2 = torch.cat([up2,up2_am],dim=1)
        up2 = self.u_up_conv4(up2)

        up1 = self.u_up_maxpool4(up2)
        up1_am = self.t_am1(u_d_conv1,up1)
        up1 = torch.cat([up1,up1_am],dim=1)
        up1 = self.u_up_conv5(up1)
        
        end = self.endconv(up1)
        
        return end