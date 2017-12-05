#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
 *Epoch:[0] Prec@1 99.384 Prec@3 100.000 Loss 0.5274
'''

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import json
from model import load_model
from config import data_transforms
import pickle
import csv
from params import *
import torchvision.datasets as td
import numpy as np

phases = ['test_A']
batch_size = BATCH_SIZE

if phases[0] == 'test_A':
    test_root = 'data/pig_test_resize'
elif phases[0] == 'test_B':
    test_root = 'data/test_B'
elif phases[0] == 'val':
    test_root = 'data/validation_folder_full'

checkpoint_filename = arch + '_' + pretrained
multi_checks = []
'''
在这里指定使用哪几个epoch的checkpoint进行平均
'''
for epoch_check in ['5','7','9']:   # epoch的列表，如['10', '20']
    multi_checks.append('checkpoint/' + checkpoint_filename + '_' + str(epoch_check)+'.pth.tar')


'''
这是imagefolder的顺序
'''
if not triplet:
    aaa = ['1','10', '11','12','13','14', '15', '16', '17', '18','19', '2', '20', '21', '22','23', 
     '24', '25', '26', '27', '28', '29', '3', '30', '4', '5', '6', '7', '8','9']
else:
    aaa = [str(i+1) for i in range(0,30)]



best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar' 
model_conv = load_model(arch, pretrained, use_gpu=use_gpu, num_classes=num_classes,  AdaptiveAvgPool=AdaptiveAvgPool,
                       SPP=SPP, num_levels=num_levels, pool_type=pool_type, bilinear=bilinear, stage=stage, 
                       SENet=SENet,se_stage=se_stage,se_layers=se_layers, 
                       threshold_before_avg = threshold_before_avg, triplet = triplet)
for param in model_conv.parameters():
    param.requires_grad = False #节省显存

best_checkpoint = torch.load(best_check)
if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
    model_conv.features = nn.DataParallel(model_conv.features)
    model_conv.cuda()
    model_conv.load_state_dict(best_checkpoint['state_dict']) 
else:
    model_conv = nn.DataParallel(model_conv).cuda()
    model_conv.load_state_dict(best_checkpoint['state_dict']) 
    
with open(test_root+'/pig_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
    label_raw_test = json.load(f)
    
def write_to_csv(aug_softmax, epoch_i = None): #aug_softmax[img_name_raw[item]] = temp[item,:]

    if epoch_i != None:
        file = 'result/'+ phases[0] +'_1_'+ epoch_i.split('.')[0].split('_')[-1] + '.csv'
    else:
        file = 'result/'+ phases[0] +'_1.csv'
    with open(file, 'w', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile,dialect='excel')
        for item in aug_softmax.keys():
            the_sum = sum(aug_softmax[item])
            for c in range(0,30):
                if phases[0] != 'val':
                    spamwriter.writerow([int(item.split('.')[0]), c+1, aug_softmax[item][aaa.index(str(c+1))]/the_sum])
                else:
                    spamwriter.writerow([item, c+1, aug_softmax[item][aaa.index(str(c+1))]/the_sum])


class SceneDataset(Dataset):

    def __init__(self, json_labels, root_dir, transform=None):
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_raw)

    def __getitem__(self, idx):
#        if phases[0] == 'val':
#            img_name = self.root_dir+ '/' + str(self.label_raw[idx]['label_id']+1) + '/'+ self.label_raw[idx]['image_id']
#        else:
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        img_name_raw = self.label_raw[idx]['image_id']
        image = Image.open(img_name)
        label = self.label_raw[idx]['label_id']

        if self.transform:
            image = self.transform(image)

        return image, label, img_name_raw


transformed_dataset_test = SceneDataset(json_labels=label_raw_test,
                                        root_dir=test_root,
                                           transform=data_transforms('test',input_size, train_scale, test_scale)
                                           )           
dataloader = {phases[0]:DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)
             }
dataset_sizes = {phases[0]: len(label_raw_test)}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: logits
    target: labels
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        

    pred_list = pred.tolist()  #[[14, 13], [72, 15], [74, 11]]
    return res, pred_list


def test_model (model, criterion):
    since = time.time()

    mystep = 0    

    for phase in phases:
        
        model.eval()  # Set model to evaluate mode

        top1 = AverageMeter()
        top3 = AverageMeter()
        loss1 = AverageMeter()
        aug_softmax = {}

        # Iterate over data.
        for data in dataloader[phase]:
            # get the inputs
            mystep = mystep + 1
#            if(mystep%10 ==0):
#                duration = time.time() - since
#                print('step %d vs %d in %.0f s' % (mystep, total_steps, duration))

            inputs, labels, img_name_raw= data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = model(inputs)
            crop_softmax = nn.functional.softmax(outputs)
            temp = crop_softmax.cpu().data.numpy()
            for item in range(len(img_name_raw)):
                aug_softmax[img_name_raw[item]] = temp[item,:] #防止多线程啥的改变了图片顺序，还是按照id保存比较保险
                
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            
#            # statistics
            res, pred_list = accuracy(outputs.data, labels.data, topk=(1, 3))
            prec1 = res[0]
            prec3 = res[1]
            top1.update(prec1[0], inputs.size(0))
            top3.update(prec3[0], inputs.size(0))
            loss1.update(loss.data[0], inputs.size(0))
            

        print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f} Loss@1 {loss1.avg:.6f}'.format(top1=top1, top3=top3, loss1=loss1))

    return aug_softmax



criterion = nn.CrossEntropyLoss()


######################################################################
# val and test
total_steps = 1.0  * len(label_raw_test) / batch_size * len(multi_checks)
print(total_steps)

class Average_Softmax(object):
    """for item in range(len(img_name_raw)):
        aug_softmax[img_name_raw[item]] = temp[item,:]
    """
    def __init__(self, inits):
        self.reset(inits)
    def reset(self, inits):
        self.val = inits
        self.avg = inits
        self.sum = inits
        self.total_weight = 0
    def update(self, val, w=1):
        self.val = val
        self.sum_dict(w)
        self.total_weight += w
        self.average()
    def sum_dict(self, w):
        for item in self.val.keys():
            self.sum[item] += (self.val[item] * w) 
    def average(self):
        for item in self.avg.keys():
            self.avg[item] = self.sum[item]/self.total_weight

image_names = [item['image_id'] for item in label_raw_test]
inits = {}
for name in image_names:
    inits[name] = np.zeros(30)
aug_softmax_multi = Average_Softmax(inits)


for i in multi_checks:
    i_checkpoint = torch.load(i)
    print(i)
    if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
        #model_conv.features = nn.DataParallel(model_conv.features)
        #model_conv.cuda()
        model_conv.load_state_dict(i_checkpoint['state_dict']) 
    else:
        #model_conv = nn.DataParallel(model_conv).cuda()
        model_conv.load_state_dict(i_checkpoint['state_dict']) 
    aug_softmax = test_model(model_conv, criterion)
    write_to_csv(aug_softmax, i)
    aug_softmax_multi.update(aug_softmax)

'''
输出融合的结果，并计算融合后的loss和accuracy
'''
def cal_loss(aug_softmax, label_raw_test):
    loss1 = 0
    for row in label_raw_test:
        loss1 -= np.log(aug_softmax[row['image_id']][row['label_id']])
    loss1 /= len(label_raw_test)
    print('Loss@1 {loss1:.6f}'.format(loss1=loss1))
write_to_csv(aug_softmax_multi.avg)
cal_loss(aug_softmax_multi.avg, label_raw_test)  