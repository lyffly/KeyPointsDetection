#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint

from data_loader import KFDataset
from models import KFSGNet

config = dict()
config['lr'] = 0.0000009
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch_num'] = 100
config['batch_size'] = 40
config['sigma'] = 5.0
config['debug_vis'] = False 

config['fname'] = 'images_kp/'
config['is_test'] = False

config['save_freq'] = 10
config['checkout'] = 'kd_epoch_429_model.ckpt'
config['start_epoch'] = 429
config['load_pretrained_weights'] = True
config['eval_freq'] = 5
config['debug'] = False
config['featurename2id'] = {
    'point1_x':0,
    'point1_y':1,
    'point2_x':2,
    'point2_y':3,
    'point3_x':4,
    'point3_y':5,
    'point4_x':6,
    'point4_y':7
    }


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,4,256,256)
    :return:numpy array (N,4,2) # 
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=4 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred_points,gts,indices_valid=None):
    """

    :param pred_points: numpy (N,4,2)
    :param gts: numpy (N,4,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss

# 计算mask ？？ 
def calculate_mask(heatmaps_target):
    """

    :param heatmaps_target: Variable (N,4,256,256)
    :return: Variable (N,4,256,256)
    """
    N,C,_,_ = heatmaps_targets.size()  #N =8 C = 4
    N_idx = []
    C_idx = []
    for n in range(N):      # 0-7
        for c in range(C):  # 0-3
            max_v = heatmaps_targets[n,c,:,:].max().item()
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.0
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]

if __name__ == '__main__':
    pprint.pprint(config)
    torch.manual_seed(0)
    cudnn.benchmark = True
    net = KFSGNet()
    net.float().cuda()
    net.train()
    criterion = nn.MSELoss()
    #criterion2 = nn.P()
    #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    #optimizer = optim.Adam(net.parameters(),lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer = optim.RMSprop(net.parameters(),lr=config['lr'],
                                    weight_decay=config['weight_decay'],
                                    momentum=config['momentum'])
    # 定义 Dataset
    trainDataset = KFDataset(config)
    #trainDataset.load()
    # 定义 data loader
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True,num_workers=8)
    sample_num = len(trainDataset)

    if config['load_pretrained_weights']:
        if (config['checkout'] != ''):
            net.load_state_dict(torch.load(config['checkout']))

    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        running_loss = 0.0
        for i, (inputs, heatmaps_targets, gts) in enumerate(trainDataLoader):
            inputs = Variable(inputs).cuda()
            heatmaps_targets = Variable(heatmaps_targets).cuda()
            mask,indices_valid = calculate_mask(heatmaps_targets)

            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs #* mask
            heatmaps_targets = heatmaps_targets #* mask
            loss = criterion(outputs, heatmaps_targets)
            loss.backward()
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)

            # 评估
            all_peak_points = get_peak_points(heatmaps_targets.cpu().data.numpy())
            loss_coor = get_mse(all_peak_points, gts.numpy(),indices_valid)

            print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} loss_coor : {:15} max : {:10} min : {}'.format(
                epoch, i * config['batch_size'],
                sample_num, loss.item(),loss_coor.item(),v_max.item(),v_min.item()))



        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(),'kd_epoch_{}_model.ckpt'.format(epoch))

