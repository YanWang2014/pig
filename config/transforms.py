# https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
# https://github.com/ncullen93/torchsample/tree/master/torchsample/transforms
# https://keras.io/preprocessing/image/
'''
https://zhuanlan.zhihu.com/p/29513760

resize
rescale
noise
flip
rotate
shift
zoom
shear
contrast
channel shift
PCA
gamma
'''

import torch
from torchvision import transforms
import random
from PIL import Image
from .transforms_master import ColorJitter, scale, ten_crop, to_tensor
import collections

#input_size = 224 
#train_scale = 256 
#test_scale = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rerange them to [-1, +1]

def my_transform(img, input_size, train_scale, test_scale):
    img = scale(img, test_scale)
    imgs = ten_crop(img, input_size)  # this is a list of PIL Images
    return torch.stack([normalize(to_tensor(x)) for x in imgs], 0) # returns a 4D tensor
class my_ten_crops(object):
    def __init__(self, input_size, train_scale, test_scale):
        self.input_size = input_size
        self.train_scale = train_scale
        self.test_scale = test_scale
    def __call__(self, img):
        return my_transform(img, self.input_size, self.train_scale, self.test_scale)

# following ResNet paper, note that center crop should be removed if we can handle different image sizes in a batch
def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
class HorizontalFlip(object):
    def __init__(self, flip_flag):
        self.flip_flag = flip_flag
    def __call__(self, img):
        if self.flip_flag:
            return hflip(img)
        else:
            return img
def my_transform_multiscale_test(varied_scale, flip_flag):  
    return transforms.Compose([
        transforms.Scale(varied_scale),  
        transforms.CenterCrop(varied_scale), 
        HorizontalFlip(flip_flag),
        transforms.ToTensor(),
        normalize
    ])

#https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/#more
            transforms.Scale_larger(test_scale),   # 将长边scale到test_scale，保持长宽比
        transforms.Pad2Square(input_size)
def my_resize(img, size, interpolation=Image.BILINEAR):
    """ Adapted from but opposite to the oficial resize function, i.e.
        If size is an int, the larger edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size, size * width / height)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w > h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)
class my_Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return my_resize(img, self.size, self.interpolation)
    
class Pad2Set(object):
    """ Adapted from but different to the oficial Pad class, i.e.
    Pad the given PIL Image on all sides to the target value while keeping the original image
    in the center.

    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on w and h respectively. 
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, collections.Sequence) and len(padding) not in [2]:
            raise ValueError("Padding must be an int or a 2 element tuple, not a " +
                             "{} element tuple".format(len(padding)))
        self.padding = padding
        self.fill = fill
    def __call__(self, img):
        #own code here, note that self.padding refers to the target size first, and
        #then goes back to its original meaning defined in the official pad class or function.
        #i.e. if a tuple of length 4 is provided
        #    this is the padding for the left, top, right and bottom borders respectively.
        w, h = img.size
        if len(self.padding) > 1:
            raise ValueError("Currently Pad2Set only support input of a single int")
        self.padding = ((self.padding-w)//2, (self.padding-h)//2, 
                        self.padding-(self.padding-w)//2, self.padding-(self.padding-h)//2)
        return pad(img, self.padding, self.fill)

composed_data_transforms = {}
def data_transforms(phase, input_size = 224, train_scale = 256, test_scale = 256):
    print('input_size %d, train_scale %d, test_scale %d' %(input_size,train_scale,test_scale))
    
    composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'train2': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        ColorJitter(),
        transforms.ToTensor(), 
        normalize
    ]),
    'multi_scale_train': transforms.Compose([   ## following ResNet paper, but not include the standard color augmentation from AlexNet
        transforms.Scale(random.randint(384, 640)),  # May be adjusted to be bigger
        transforms.RandomCrop(input_size),  # not RandomSizedCrop
        transforms.RandomHorizontalFlip(), 
        ColorJitter(), # different from AlexNet's PCA method which is adopted in the ResNet paper?
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'ten_crop': my_ten_crops(input_size, train_scale, test_scale),#todo: merge my_transform
    'scale_pad': transforms.Compose([ 
        transforms.my_Resize(test_scale),   # 将长边scale到test_scale，保持长宽比
        transforms.Pad2Set(input_size), # pad 成正方形，边长为input_size
        transforms.ToTensor(),
        normalize
    ]),
    }
    return composed_data_transforms[phase]
