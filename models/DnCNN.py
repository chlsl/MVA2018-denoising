import torch
import torch.nn as nn



class CONV_BN_RELU(nn.Module):
    
    def __init__(self,in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3, onlyconv=False):
        super(CONV_BN_RELU, self).__init__()

        #self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        #self.bn = nn.BatchNorm2d(out_channels)
        #self.relu = nn.ReLU(inplace=True)
        self.onlyconv = onlyconv
        
        self.pad  = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=True)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        if not self.onlyconv:
            out = self.bn(out)
            out = self.relu(out)

        return(out)
    
    
class DnCNN(nn.Module):
  
    def __init__(self, inchannels, outchannels, num_of_layers=17, features=64, kernel_size=3):
        super(DnCNN, self).__init__()
        padding = 0
        ll=0
        layers = []
        layers.append(nn.ReflectionPad2d(kernel_size//2))
        layers.append(nn.Conv2d(in_channels=inchannels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        #self.modules_['layer%d'%ll] = layers[0]
        #ll=ll
        
        for _ in range(num_of_layers-2):
            layers.append(CONV_BN_RELU(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2))
            
            #layers.append(nn.ReflectionPad2d(kernel_size//2))
            #layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            #layers.append(nn.BatchNorm2d(features))
            #layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ReflectionPad2d(kernel_size//2))
        layers.append(nn.Conv2d(in_channels=features, out_channels=outchannels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out