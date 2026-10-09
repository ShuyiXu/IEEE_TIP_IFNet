# -*- coding: utf-8 -*-
"""

@author: 13572
"""

import argparse
import torch
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoints',help='checkpoints')
parser.add_argument('--seed',type=int, default=30,help='初始化种子')
parser.add_argument('--model_name',type=str, default='IFNet',help='选择模型')
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--dataset',type=str, default='HVCD',help='选择数据')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoches', type=int, default=10, help='epoch number')
parser.add_argument('--batch_size', type=int, default=1, help='number of batch size') ## 8
parser.add_argument('--gumbel_temperature', type=float, default=0.1, help='gumbel_temperature')
parser.add_argument('--step_size', type=int, default=200, help='step_size')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--lamda', type=float, default=10, help='lamda')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')
args.device = device
args.expr_dir=os.path.join('checkpoints', args.dataset+'_'+str(args.model_name)+'_bs'+str(args.batch_size)+'_lr'+
                           str(args.learning_rate)+'_epo'+str(args.epoches)+'_gT'+str(args.gumbel_temperature)+'_la'+str(args.lamda)+
                           '_Thre'+str(args.threshold))

# if not os.path.exists(args.expr_dir):
#     os.makedirs(args.expr_dir)


