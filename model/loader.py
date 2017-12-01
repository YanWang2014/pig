'''
https://github.com/pytorch/vision/tree/master/torchvision/models

各种卷积的动态图
http://www.sohu.com/a/159591827_390227


fcn
    densenet: 好改
    vgg, alexnet: https://github.com/pytorch/vision/pull/184
dilation
    resnet(dilation): https://github.com/pytorch/vision/pull/190
    dilation卷积本身有接口
Depthwise Separable Convolution
    https://discuss.pytorch.org/t/separable-convolutions-in-pytorch/3407
    https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315
attention
    stn
    SENet
    Residual attention network (商汤)
    RL
Deformable Convolution
    https://github.com/1zb/deformable-convolution-pytorch 可能更快？
    https://github.com/oeway/pytorch-deform-conv
'''

import os
from functools import partial
import pickle
import torch
import torch.nn as nn
import torchvision
from .Preact_resnet50_places365 import Preact_resnet50_places365
from .resnet152_places365 import resnet152_places365
import torchvision.models
from .spp_layer import SPPLayer
from .compact_bilinear_pooling import CompactBilinearPooling
from .se_resnet152_places365 import give_se_resnet152_places365
from .mask_relu import Mask_relu


support_models = {
    'places': ('alexnet', 'densenet161', 'resnet18', 'resnet50', 'preact_resnet50', 'resnet152'),
    'imagenet': tuple(filter(lambda x: (x.lower() == x) and (not x.startswith('_') and \
                                       (not x in ['densenet', 'resnet', 'vgg', 'inception', 'squeezenet'])) , \
                             dir(torchvision.models)))
}

model_file_root = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'places365')


def load_model(arch, pretrained, use_gpu=True, num_classes=30, AdaptiveAvgPool=False, SPP=False, num_levels=3, pool_type='avg_pool', bilinear={'use':False,'dim':16384}, stage=2, SENet=False, se_stage=2, se_layers=None, threshold_before_avg = False):
    num_mul = sum([(2**i)**2 for i in range(num_levels)])
    if SPP and (AdaptiveAvgPool or threshold_before_avg):
        raise ValueError("Set AdaptiveAvgPool = False and threshold_before_avg = False when using SPP = True")
    if bilinear['use'] and (SPP or AdaptiveAvgPool or threshold_before_avg):
        raise ValueError("Set AdaptiveAvgPool, SPP and threshold_before_avg = False when using bilinear")
    if AdaptiveAvgPool or SPP or SENet or threshold_before_avg:
        if not 'resnet' in arch:
            raise NotImplementedError("Currently AdaptiveAvgPool, SPP, SE and threshold_before_avg only support resnets")
    if threshold_before_avg and pretrained == 'places':
        raise NotImplementedError("Currently threshold_before_avg only support resnets pretrained on imagenet")
    
    if not arch in support_models[pretrained]:
        raise ValueError("No such places365 or imagenet pretrained model found")

    if pretrained == 'imagenet':
        model = torchvision.models.__dict__[arch](pretrained=True)
        if stage == 1: #第一阶段只训练新加的层
            for param in model.parameters():
                param.requires_grad = False
    elif pretrained == 'places':
        if arch == 'preact_resnet50':
            if SENet == True:
                raise NotImplementedError("Currently SE does not support preact_resnet50")
            model = Preact_resnet50_places365
            model.load_state_dict(torch.load(os.path.join(model_file_root, 'Preact_resnet50_places365.pth')))
            model._modules['12']._modules['1'] = nn.Linear(2048, num_classes)
            if AdaptiveAvgPool:
                model._modules['10'] = nn.AdaptiveAvgPool2d(1)
            if SPP:
                model._modules['10'] = SPPLayer(num_levels, pool_type) #1时应该等价于adaptiveavgpool
                model._modules['12']._modules['1'] = nn.Linear(2048*num_mul, num_classes)
            if bilinear['use']:
                input_C = 2048
                if stage == 1: 
                    for param in model.parameters():
                            param.requires_grad = False
                model._modules['10'] = CompactBilinearPooling(input_C, input_C, bilinear['dim']) 
                model._modules['12']._modules['1'] = nn.Linear(int(model._modules['12']._modules['1'].in_features/input_C*bilinear['dim']), num_classes) 
            return model
        elif arch == 'resnet152':
            if SENet == True:
                model = give_se_resnet152_places365(16, se_stage, se_layers)
                #use pretrained weights from places 365
#                original = resnet152_places365
#                original.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
#                or_dict = original.state_dict()
#                se_dict = model.state_dict()
#                for key, value in or_dict.items():
#                    if key in se_dict:
#                        se_dict[key] = value
#                print(or_dict)
#                print('============================')
#                print(se_dict)
            else:
                model = resnet152_places365
                model.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
                
            model._modules['10']._modules['1'] = nn.Linear(2048, num_classes)
            if AdaptiveAvgPool:
                model._modules['8'] = nn.AdaptiveAvgPool2d(1)
            if SPP:
                model._modules['8'] = SPPLayer(num_levels, pool_type)
                model._modules['10']._modules['1'] = nn.Linear(2048*num_mul, num_classes)
            if bilinear['use']:
                input_C = 2048
                if stage == 1: 
                    for param in model.parameters():
                            param.requires_grad = False
                model._modules['8'] = CompactBilinearPooling(input_C, input_C, bilinear['dim']) 
                model._modules['10']._modules['1'] = nn.Linear(int(model._modules['10']._modules['1'].in_features/input_C*bilinear['dim']), num_classes) 
            return model
        else:
            model_file = os.path.join(model_file_root, 'whole_%s_places365.pth.tar' % (arch))

            ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            if use_gpu:
                model = torch.load(model_file, pickle_module=pickle)
            else:
                # model trained in GPU could be deployed in CPU machine like this!
                model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle) 

    if arch.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if AdaptiveAvgPool:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        if SPP:
            model.avgpool = SPPLayer(num_levels, pool_type)
            model.fc = nn.Linear(model.fc.in_features*num_mul, num_classes)
        if bilinear['use']:
            if arch == 'resnet18' or arch == 'resnet34':
                input_C = 512# resnet fc之前的通道数， https://github.com/KaimingHe/deep-residual-networks
            else:
                input_C = 2048
            if stage == 1: #第一阶段只训练新加的层
                for param in model.parameters():
                        param.requires_grad = False
            model.avgpool = CompactBilinearPooling(input_C, input_C, bilinear['dim']) #(input_C, input_C, output_C)
            model.fc = nn.Linear(int(model.fc.in_features/input_C*bilinear['dim']), num_classes) #实际上就是batch_size * dim，因为resnet本来就是pool成1*1了，所以in_features = batch_size * C
        if threshold_before_avg:
            model.avgpool = Mask_relu()
            model.fc = nn.Linear(model.fc.in_features * 2, num_classes)
    elif arch.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch.startswith('inception'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith('vgg') or arch == 'alexnet':
        model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, num_classes)
    else:
        raise NotImplementedError('This pretrained model has not been adapted to the current tast yet.')

    return model