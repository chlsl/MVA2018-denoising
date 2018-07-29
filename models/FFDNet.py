"""
Parametrable FFDNet model (https://github.com/cszn/FFDNet.git)

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""


import torch
import torch.nn as nn

from .DnCNN import CONV_BN_RELU, DnCNN

##
class CONV_BN_RELU(nn.Module):
    '''
    model for a layer with: 2D CONV + BatchNorm + ReLU activation
    the parameters indicate the input and output channels, 
    the kernel size, the padding, and the stride 
    '''
    def __init__(self,in_channels=128, out_channels=128, kernel_size=7, 
                 stride=1, padding=3):
        super(CONV_BN_RELU, self).__init__()

        #self.pad  = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return(out)



class FFDNet(nn.Module):
    '''
    model for a FFDNet network build using CONV_BN_RELU units 
    the parameters indicate the input and output channels, 
    the kernel size (default 3), the number of layers (15 for grayscale),  
    and number of features (default 64)
    '''
    def __init__(self, inchannels=1, outchannels=1, num_of_layers=15, 
                 features=64, kernel_size=3):
        super(FFDNet, self).__init__()
        
        # DnCNN object
        self.dncnn = DnCNN(5,4,num_of_layers,features,kernel_size,residual=False)

        
# FOLD - UNFOD SEMANTICS
#########################
#from torch import nn
#x = dtype(np.random.randn(3,1,4,4))
#y = nn.Unfold(kernel_size=(2, 2),stride=2)(x) 
#print(y.numpy().shape)
#print(x,y.reshape((3,4,2,2)).shape)
#
#_ = nn.Fold(output_size=(4,4), kernel_size=(2, 2), stride=2)(y)


        
    def forward(self, x, sigma_map):
        ''' forward declaration '''
        
        # subsample sigma_map
        sig = sigma_map[:,:,::2,::2]
        
        sh = x.shape
            
        # unfold the image in 4 planes
        #x = dtype(np.random.randn(1,1,4,4))
        dx,dy=0,0
        if sh[3]%2==1:         dx=1 
        if sh[2]%2==1:         dy=1
        x = nn.functional.pad(x, (0, dx, 0, dy) )
        shtmp = x.shape
            
        y = nn.Unfold(kernel_size=(2, 2),stride=2)(x) 
        #print(y.numpy().shape)
        #print(x,y.reshape((1,4,2,2)))
        y = y.reshape(sh[0],4,shtmp[2]//2,shtmp[3]//2)
        
        # concatenate sigma_map with image
        y = torch.cat([y, sig], dim=1)
        
        # call dncnn
        out = self.dncnn(y)
        
        # fold the output
        shout = out.shape
        out = out.reshape(shout[0],4,shout[2]*shout[3])
        out = nn.Fold(output_size=(shout[2]*2, shout[3]*2), kernel_size=(2, 2), stride=2)(out)
        
        return(out[:,:,:sh[2],:sh[3]])




def FFDNet_pretrained_grayscale(savefile=None, verbose=False):
    '''
    Loads the pretrained weights of DnCNN for grayscale images from 
    https://github.com/cszn/FFDNet.git
    
    savefile: is the .pt file to save the model weights 
    returns the FFDNet(1,1) model with 15 layers with the pretrained weights
    '''
    
 
    # download the pretained weights
    import os
    import subprocess

    here = os.path.dirname(__file__)
    try:
        os.stat(here+'/FFDNet')
    except OSError:
        print('downloading pretrained models')
        subprocess.run(['git', 'clone',  'https://github.com/cszn/FFDNet.git'],cwd=here)
        
        
    # read the weights
    import numpy as np
    import hdf5storage
    import torch

    dtype = torch.FloatTensor
   
    m = FFDNet()
        
    mat = hdf5storage.loadmat(here+'/FFDNet/models/FFDNet_gray.mat')

    TRANSPOSE_PATTERN = [3, 2, 0, 1]

    # copy first 16 layers
    t=2
    for r in range(14):
        x = mat['net'][0][0][0][t]
        if verbose:
            print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
            print(r, m.dncnn.layers[r].conv.weight.shape, m.dncnn.layers[r].conv.bias.shape)

        w = np.array(x[0][1][0,0])
        b = np.array(x[0][1][0,1]).squeeze()

        m.dncnn.layers[r].conv.weight = torch.nn.Parameter( dtype( np.reshape(w.transpose(TRANSPOSE_PATTERN) , m.dncnn.layers[r].conv.weight.shape  )  ) ) 
        m.dncnn.layers[r].conv.bias   = torch.nn.Parameter( dtype( b ) )
        m.dncnn.layers[r].bn.bias     = torch.nn.Parameter( m.dncnn.layers[r].bn.bias    *0 )
        m.dncnn.layers[r].bn.weight   = torch.nn.Parameter( m.dncnn.layers[r].bn.weight  *0 +1)
        t+=2

    # copy last layer 
    r=14
    x = mat['net'][0][0][0][t]
    if verbose:
        print('a', t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
        print(r, m.dncnn.layers[r].weight.shape, m.dncnn.layers[r].bias.shape)

    w = np.array(x[0][1][0,0])
    b = np.array(x[0][1][0,1]).squeeze()
    
    m.dncnn.layers[r].weight = torch.nn.Parameter( dtype( w.transpose(TRANSPOSE_PATTERN)  )  )  
    m.dncnn.layers[r].bias   = torch.nn.Parameter( dtype( b ) )

    
    if savefile:
        torch.save(m, savefile)

    return m

    