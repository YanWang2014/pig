#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:19:10 2017

@author: wayne
"""
import torch

TRAIN_ROOT = 'data/videos_folder/'
VALIDATION_ROOT = 'data/validation_folder/'


arch = 'resnet18' # preact_resnet50, resnet152
pretrained = 'imagenet' #imagenet
threshold_before_avg = True
evaluate = False
checkpoint_filename = arch + '_' + pretrained
save_freq = 2
try_resume = False
print_freq = 10
if_debug = False
start_epoch = 0
class_aware = False
AdaptiveAvgPool = True
SPP = False
num_levels = 1 # 1 = fcn
pool_type = 'avg_pool'
bilinear = {'use':False,'dim':16384}  #没有放进hyper_board
stage = 1 #注意，1只训练新加的fc层，改为2后要用try_resume = True
SENet = False
se_layers = [None,None,None,'7'] # 4,5,6,7 [3, 8, 36, 3]
print(se_layers)
se_stage = 1 # 1冻结前面几层[去模型那调]， 2全开放
input_size = 224#[224, 256, 384, 480, 640] 
train_scale = 224
test_scale = 224
train_transform = 'train2'
lr_decay = 0.5

# training parameters:
BATCH_SIZE = 32
INPUT_WORKERS = 8
epochs = 3
use_epoch_decay = True # 可以加每次调lr时load回来最好的checkpoint
lr = 0.01  #0.01  0.001
lr_min = 1e-6

if_fc = False #是否先训练最后新加的层，目前的实现不对。
lr1 = lr_min #if_fc = True, 里面的层先不动
lr2 = 0.2 #if_fc = True, 先学好最后一层
lr2_min = 0.019#0.0019 #lr2每次除以10降到lr2_min，然后lr2 = lr, lr1 = lr2/slow
slow = 1 #if_fc = True, lr1比lr2慢的倍数
print('lr=%.8f, lr1=%.8f, lr2=%.8f, lr2_min=%.8f'% (lr,lr1,lr2,lr2_min))

weight_decay=0 #.05 #0.0005 #0.0001  0.05太大。试下0.01?
optim_type = 'Adam' #Adam SGD http://ruder.io/optimizing-gradient-descent/
confusions = 'Entropic' #'Pairwise' 'Entropic'
confusion_weight = 0.5  #for pairwise loss is 0.1N to 0.2N (where N is the number of classes), and for entropic is 0.1-0.5. https://github.com/abhimanyudubey/confusion
betas=(0.9, 0.999)
eps=1e-08 # 0.1的话一开始都是prec3 4.几
momentum = 0.9

use_gpu = torch.cuda.is_available()
