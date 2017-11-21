import torch
import torch.nn as nn

class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):

        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = nn.functional.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            elif self.pool_type == 'avg_pool':
                tensor = nn.functional.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            elif self.pool_type == 'mix_pool':
                if i == 0:
                    tensor = nn.functional.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
                else:
                    tensor = nn.functional.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                raise NotImplementedError("Currently SPP only supports max, avg and mix _pool")
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

#
#https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
#def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
#    '''
#    previous_conv: a tensor vector of previous convolution layer
#    num_sample: an int number of image in the batch
#    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
#    out_pool_size: a int vector of expected output size of max pooling layer
#    
#    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
#    '''    
#    # print(previous_conv.size())
#    for i in range(len(out_pool_size)):
#        # print(previous_conv_size)
#        h_wid = previous_conv_size[0] // out_pool_size[i]
#        w_wid = previous_conv_size[1] // out_pool_size[i]
#        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
#        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
#        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
#        x = maxpool(previous_conv)
#        if(i == 0):
#            spp = x.view(num_sample,-1)
#            # print("spp size:",spp.size())
#        else:
#            # print("size:",spp.size())
#            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
#    return spp
