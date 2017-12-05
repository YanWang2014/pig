'''
adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py

Pytorch提供的TripletMarginLoss计算如下，注意似乎没有进行归一化，所以我们在网络中进行归一化:
    http://pytorch.org/docs/master/nn.html?highlight=normalize#torch.nn.functional.normalize

torch.nn.functional.pairwise_distance(x1, x2, p=2, eps=1e-06)
Computes the batchwise pairwise distance between vectors v1,v2
Input: (N,D) where D = vector dimension
Output: (N,1)


>>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
>>> input1 = autograd.Variable(torch.randn(100, 128))
>>> input2 = autograd.Variable(torch.randn(100, 128))
>>> input3 = autograd.Variable(torch.randn(100, 128))
>>> output = triplet_loss(input1, input2, input3)
>>> output.backward()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y=None, z=None):
        logits_x = self.embeddingnet(x)  #这里是logits，供softmax使用

        if y==None and z==None:
            #print("Tripletnet working in test mode")
            return logits_x
        else:
            #print("Tripletnet working in test mode")
            embedded_x = F.normalize(logits_x)  #这里是l2归一化的logits，供triplet loss使用。甚至可以使用和softmax不同层的特征？
            embedded_y = F.normalize(self.embeddingnet(y))
            embedded_z = F.normalize(self.embeddingnet(z))
    
            return logits_x, embedded_x, embedded_y, embedded_z
