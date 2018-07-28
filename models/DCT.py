"""
DCT-like CNN models

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

import torch
import torch.nn as nn

class DCT(nn.Module):
    def __init__(self, ksize=7):
        super(DCT, self).__init__()
        #self.sigma = nn.Parameter(torch.Tensor(1).fill_(1))
        
        channels = ksize**2
        self.direct_trans = nn.Conv2d(in_channels=1,out_channels=channels,
                                      kernel_size=ksize,stride=1,padding=ksize//2)
        
        self.shrinkage = nn.Softshrink()
        
        self.inverse_trans_and_aggregation = nn.Conv2d(in_channels=channels,out_channels=1,
                                      kernel_size=ksize,stride=1,padding=ksize//2)
 
    def forward(self, x):
        out = self.direct_trans(x)
        out = self.shrinkage(out)      
        out = x + self.inverse_trans_and_aggregation(out) # residual is faster to train
        
        return(out)
    
    
    
class DCTlike(nn.Module):
    def __init__(self, ksize=7):
        super(DCTlike, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor(1).fill_(1))
        
        
        ch = ksize**2
        
        self.direct_trans = nn.Conv2d(in_channels=1,out_channels=ch,
                                      kernel_size=ksize,stride=1,padding=ksize//2, bias=False)
        
        self.shrinkage = nn.Softshrink()
        
        self.inverse_trans_and_aggregation = nn.ConvTranspose2d(in_channels=ch,out_channels=1,
                                      kernel_size=ksize,stride=1,padding=ksize//2, bias=True)
 
    def forward(self, x):
        out = self.direct_trans(x)
        out = self.shrinkage(out)      
        out = x - self.inverse_trans_and_aggregation(out)
        
        return(out)
    
    
    
    
    

   
class DCTlike(nn.Module):
    def __init__(self, ksize=7, sigma=30, initializeDCT=True):
        super(DCTlike, self).__init__()       
        from scipy.fftpack import dct, idct
        import numpy as np
        
        dtype = torch.FloatTensor
        if torch.cuda.is_available():  dtype   = torch.cuda.FloatTensor
            
        self.sigma = sigma
            
        ch = ksize**2 *2
                
        self.direct_trans = nn.Conv2d(in_channels=1,out_channels=ch,
                                      kernel_size=ksize,stride=1,padding=ksize//2, bias=False)

        self.thr = nn.Parameter(dtype(np.ones((1,ch,1,1))), requires_grad=True)

        self.inverse_trans_and_aggregation = nn.ConvTranspose2d(in_channels=ch,out_channels=1,
                                      kernel_size=ksize,stride=1,padding=ksize//2, bias=False)
        
        if initializeDCT:
            # initialize the isometric DCT transforms        
            for i in range(ch):
                a=np.zeros((ksize,ksize)); a.flat[i] = 1
                a1 = dct(dct(a.T,norm='ortho', type=3).T,norm='ortho', type=3)
                a2 = idct(idct(a.T,norm='ortho', type=2).T,norm='ortho', type=2)

                self.direct_trans.weight.data[i,0,:,:] = nn.Parameter(dtype(a1));
                self.inverse_trans_and_aggregation.weight.data[i,0,:,:] = 1/(ch) *nn.Parameter(dtype(a2));
                #        vistools.display_imshow(dct(dct(a.T).T))

            
                
    def forward(self, x):
        # direct transform
        out = self.direct_trans(x)

        # hard thresholding (the not so easy way)
        out = out*self.thr
        th = 3.0*self.sigma/255     #  3*sigma
        tmp =  out * (torch.abs(out) > th).float() 
        #tmp[:,0,:,:] = out[:,0,:,:]   # fix the DC
        out = out/self.thr

      #  out = self.shrinkage(out)     
        # inverse transform
        out =  self.inverse_trans_and_aggregation(tmp)
        
        return(out)