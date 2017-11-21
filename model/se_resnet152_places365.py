'''
https://github.com/moskomule/senet.pytorch
https://github.com/moskomule/senet.pytorch/issues/3 
    use ResNet's weight for "SENet's ResNet part" and use arbitrary weights for "SENet's SE blocks"
https://github.com/KaimingHe/deep-residual-networks  
    另外需要查看ResNet的通道数
    
http://pytorch.org/docs/master/nn.html#module
'''

import torch
from torch import nn
from .resnet152_places365 import resnet152_places365
from .se_module import SELayer
import copy 
import os

print_net = False
test_layer4 = False
test_layer5 = False
test_layer6 = False
test_layer7 = False


def give_se_resnet152_places365(reduction = 16, se_stage = 2, se_layers = None):

    model_file_root = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'places365')

    se_resnet152_places365 = copy.deepcopy(resnet152_places365)
    se_resnet152_places365.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
    
    if se_stage == 1: # 1冻结前面几层， 2全开放
        for layer in ['0','1','2','3','4','5','6','7']:
            for param in se_resnet152_places365._modules[layer].parameters():
                param.requires_grad = False
#        for param in se_resnet152_places365.parameters():
#            param.requires_grad = False


        
    #sum up all the necessary changes for se
    layers = se_layers #se_layers: 对应官方code的四个make_layer.  4,5,6,7 [3, 8, 36, 3]
    channels = [256, 512, 1024, 2048]
    for layer,C in zip(layers, channels):
        if layer != None:
            for param in se_resnet152_places365._modules[layer].parameters():
                param.requires_grad = True
            
            for ids in se_resnet152_places365._modules[layer]._modules:
#                if ids == ['2']:
                #continue
                se_resnet152_places365._modules[layer]._modules[ids]._modules['0']._modules['0'].add_module('se', SELayer(C, C//reduction))
                
    return se_resnet152_places365
        


# 按照每个redidual module的channel数增加se通路
#se_resnet152_places365._modules['10'].add_module('se',SELayer(planes * 4, reduction))


#    if print_net:
#        print(se_resnet152_places365._modules['10']._modules['1'])
#        print(se_resnet152_places365._modules['10']._modules['1'].in_features)
#        print(se_resnet152_places365._modules['10'].children())
#        
#        print(se_resnet152_places365._modules['0'])
#        print(se_resnet152_places365._modules['0'].in_channels)
#        
#        print(se_resnet152_places365.modules())
#        print(se_resnet152_places365.named_modules())
#        
#        print(se_resnet152_places365._modules['3']) 
#        
#        print(se_resnet152_places365._modules['4']) #从这里开始加 se
#        
#        x = torch.FloatTensor([1, 2, 3])
#        y = torch.FloatTensor([4, 5, 6])
#        print(x*y)
#        
#    if test_layer4:
#        layer = '4'
#        print(se_resnet152_places365._modules[layer]) #look at each residual module
#        print('===========================')
#        print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'])
#        # reduction之前的C, 256
#        print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0']._modules['7'].num_features)
#        
#    #    se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'].add_module('se', SELayer(4, reduction))
#    #    print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'])
#        
#        print('###########################')
#        for ids in se_resnet152_places365._modules[layer]._modules:
#            print(ids)
#            se_resnet152_places365._modules[layer]._modules[ids]._modules['0']._modules['0'].add_module('se', SELayer(256, 256//reduction))
#        print(se_resnet152_places365._modules[layer])
#    
#    if test_layer5:
#        layer = '5'
#        print(se_resnet152_places365._modules[layer]) #look at each residual module
#        print('===========================')
#        print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'])
#    
#        print('###########################')
#        for ids in se_resnet152_places365._modules[layer]._modules:
#            print(ids)
#            se_resnet152_places365._modules[layer]._modules[ids]._modules['0']._modules['0'].add_module('se', SELayer(512, 512//reduction))
#        print(se_resnet152_places365._modules[layer])
#        
#    if test_layer6:
#        layer = '6'
#        print(se_resnet152_places365._modules[layer]) #look at each residual module
#        print('===========================')
#        print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'])
#    
#        print('###########################')
#        for ids in se_resnet152_places365._modules[layer]._modules:
#            print(ids)
#            se_resnet152_places365._modules[layer]._modules[ids]._modules['0']._modules['0'].add_module('se', SELayer(1024, 1024//reduction))
#        print(se_resnet152_places365._modules[layer])
#        

if test_layer7:
    se_resnet152_places365 = copy.deepcopy(resnet152_places365)
    layer = '7'
    print(se_resnet152_places365._modules[layer]) #look at each residual module
    print('===========================')
    print(se_resnet152_places365._modules[layer]._modules['0']._modules['0']._modules['0'])

    print('###########################')
    for ids in se_resnet152_places365._modules[layer]._modules:
        print(ids)
#            se_resnet152_places365._modules[layer]._modules[ids]._modules['0']._modules['0'].add_module('se', SELayer(2048, 2048//reduction))
#        print(se_resnet152_places365._modules[layer])
    
        