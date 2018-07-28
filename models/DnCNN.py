"""
Parametrable DnCNN model (https://github.com/cszn/DnCNN.git)

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Inspired on:
    https://github.com/SaoYan/DnCNN-PyTorch/
    https://github.com/Ourshanabi/Burst-denoising
"""


import torch
import torch.nn as nn


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



class DnCNN(nn.Module):
    '''
    model for a DnCNN network build using CONV_BN_RELU units 
    the parameters indicate the input and output channels, 
    the kernel size (default 3), the number of layers (17 for grayscale),  
    and number of features (default 64)
    The residual option allows to deactivate the residual learning
    '''
    def __init__(self, inchannels=1, outchannels=1, num_of_layers=17, 
                 features=64, kernel_size=3, residual=True):
        super(DnCNN, self).__init__()
        
        self.residual = residual
        
        self.layers = []  
        
        # first layer 
        self.layers.append(CONV_BN_RELU(in_channels=inchannels, 
                                        out_channels=features, 
                                        kernel_size=kernel_size, 
                                        stride=1, padding=kernel_size//2))
        # intermediate layers
        for _ in range(num_of_layers-2):
            self.layers.append(CONV_BN_RELU(in_channels=features, 
                                            out_channels=features, 
                                            kernel_size=kernel_size, 
                                            stride=1, padding=kernel_size//2))
        # last layer 
        self.layers.append(nn.Conv2d(in_channels=features, 
                                     out_channels=outchannels, 
                                     kernel_size=kernel_size, 
                                     stride=1, padding=kernel_size//2))
        # chanin the layers
        self.dncnn = nn.Sequential(*self.layers)

        
    def forward(self, x):
        ''' forward declaration '''
        out = self.dncnn(x)
        
        if self.residual: # residual learning    
            out = x - out 
        
        return(out)




def DnCNN_pretrained_grayscale(sigma=30, savefile=None, verbose=False):
    '''
    loads the pretrained weights of DnCNN for grayscale images from 
    https://github.com/cszn/DnCNN.git
    
    sigma: is the level of noise in range(10,70,5)
    savefile: is the .pt file to save the model weights 
    returns the DnCNN(1,1) model with 17 layers with the pretrained weights
    '''
    
    if sigma  not in list(range(10,70,5)):
        print ('pretained sigma %d is not available'%sigma)
   
 
    # download the pretained weights
    import os
    import subprocess

    here = os.path.dirname(__file__)
    try:
        os.stat(here+'/DnCNN')
    except OSError:
        print('downloading pretrained models')
        subprocess.run(['git', 'clone',  'https://github.com/cszn/DnCNN.git'],cwd=here)
        
        
    # read the weights
    import numpy as np
    import hdf5storage
    import torch

    dtype = torch.FloatTensor
   
    m = DnCNN(1,1)
        
    mat = hdf5storage.loadmat(here+'/DnCNN/model/specifics/sigma=%d.mat'%sigma)

    TRANSPOSE_PATTERN = [3, 2, 0, 1]

    # copy first 16 layers
    t=0
    for r in range(16):
        x = mat['net'][0][0][0][t]
        if verbose:
            print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
            print(r, m.layers[r].conv.weight.shape, m.layers[r].conv.bias.shape)

        w = np.array(x[0][1][0,0])
        b = np.array(x[0][1][0,1]).squeeze()

        m.layers[r].conv.weight = torch.nn.Parameter( dtype( np.reshape(w.transpose(TRANSPOSE_PATTERN) , m.layers[r].conv.weight.shape  )  ) ) 
        m.layers[r].conv.bias   = torch.nn.Parameter( dtype( b ) )
        m.layers[r].bn.bias     = torch.nn.Parameter( m.layers[r].bn.bias    *0 )
        m.layers[r].bn.weight   = torch.nn.Parameter( m.layers[r].bn.weight  *0 +1)
        t+=2

    # copy last layer 
    r=16
    x = mat['net'][0][0][0][t]
    if verbose:
        print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
        print(r, m.layers[r].weight.shape, m.layers[r].bias.shape)

    w = np.array(x[0][1][0,0])[:,:,:,np.newaxis]
    b = np.array(x[0][1][0,1])[:,0]
    
    m.layers[r].weight = torch.nn.Parameter( dtype( w.transpose(TRANSPOSE_PATTERN)  )  )  
    m.layers[r].bias   = torch.nn.Parameter( dtype( b ) )

    
    if savefile:
        torch.save(m, savefile)

    return m

    