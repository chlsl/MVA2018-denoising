import torch
import torch.nn as nn

#plot 
%matplotlib inline 
import matplotlib.pylab as plt

##### Declare a denoising CNN
# the network stored in the mat files does not have an explicit batchnorm layer
# hence we set it to Id during the loading of the weights


class CONV_BN_RELU(nn.Module):
    def __init__(self,in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3):
        super(CONV_BN_RELU, self).__init__()
        
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return(out)
  

        
class CDNCNN(nn.Module):
    def __init__(self):
        super(CDNCNN, self).__init__()
        
        self.layer1=CONV_BN_RELU(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer2=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer3=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer4=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer5=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer6=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer7=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer8=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer9=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer10=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer11=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer12=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer13=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer14=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer15=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer16=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
       # self.layer17=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
       # self.layer18=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
       # self.layer19=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
                
        self.layer17=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1)
      #  self.layer18=nn.BatchNorm2d(1)     # is this necessary? 
 
  
    
    
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.layer5(out)
        out=self.layer6(out)
        out=self.layer7(out)
        out=self.layer8(out)
        out=self.layer9(out)
        out=self.layer10(out)
        out=self.layer11(out)
        out=self.layer12(out)
        out=self.layer13(out)
        out=self.layer14(out)
        out=self.layer15(out)
        out=self.layer16(out)
 #       out=self.layer17(out)
 #       out=self.layer18(out)
 #       out=self.layer19(out)
 #       out=self.layer17(out)
        out=x-self.layer17(out)
        
        return(out)

    
    
    
    
##### Load pretrained weights into the network   
    
# read the weights
import numpy as np
import hdf5storage

import torch

dtype = torch.FloatTensor

mat = hdf5storage.loadmat('DnCNN/model/specifics/sigma=50.mat')

m = CDNCNN()



TRANSPOSE_PATTERN = [3, 2, 0, 1]

# copy first 16 layers
t=0
for r in range(1,17):

    x = mat['net'][0][0][0][t]
    print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
    
    print(r, m.state_dict()['layer%d.conv.weight'%r].shape, m.state_dict()['layer%d.conv.bias'%r].shape)

    w = np.array(x[0][1][0,0])
    b = np.array(x[0][1][0,1]).squeeze()
    
    getattr(m,'layer%d'%r).conv.weight = torch.nn.Parameter( dtype( np.reshape(w.transpose(TRANSPOSE_PATTERN) , getattr(m,'layer%d'%r).conv.weight.shape  )  ) ) 
    getattr(m,'layer%d'%r).conv.bias   = torch.nn.Parameter( dtype( b ) )
    getattr(m,'layer%d'%r).bn.bias     = torch.nn.Parameter( getattr(m,'layer%d'%r).bn.bias  *0 )
    getattr(m,'layer%d'%r).bn.weight   = torch.nn.Parameter(getattr(m,'layer%d'%r).bn.weight  *0 +1)
    t+=2

# copy last layer 
r=17
x = mat['net'][0][0][0][t]
print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)

print(r, m.state_dict()['layer%d.weight'%r].shape, m.state_dict()['layer%d.bias'%r].shape)

w = np.array(x[0][1][0,0])[:,:,:,np.newaxis]
b = np.array(x[0][1][0,1])[:,0]
getattr(m,'layer%d'%r).weight = torch.nn.Parameter( dtype( w.transpose(TRANSPOSE_PATTERN)  )  )  
getattr(m,'layer%d'%r).bias = torch.nn.Parameter( dtype( b ) )

 
torch.save(m, 'dncnnmodel_50.pt')








#####  DISPLAY THE STRUCTURE OF THE NET

if 0:

    m = CDNCNN()

    # STRUCTURE FROM PYTORCH
    #m.state_dict().items()

    for i in range (1,17):
        print(m.state_dict()['layer%d.conv.weight'%i].shape)
    print(m.state_dict()['layer%d.weight'%17].shape)


    # STRUCTURE FROM MAT 
    import hdf5storage
    mat = hdf5storage.loadmat('DnCNN/model/specifics/sigma=50.mat')

    t=0
    for i in range(1,17):

        x = mat['net'][0][0][0][t]
        print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
        t+=2
    x = mat['net'][0][0][0][t]
    print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)

    
    
    
    
    
    
#### Test the network  
    
from PIL import Image


dtype = torch.FloatTensor
gpu_dtype=torch.cuda.FloatTensor


sigma = 50
img_clean = np.array(Image.open('DnCNN/testsets/BSD68/test001.png'), dtype='float32') / 255.0
img_test = img_clean + np.random.normal(0, sigma/255.0, img_clean.shape)
img_test = img_test.astype('float32')


img = img_test[np.newaxis,np.newaxis,:,:]
img = dtype(img)
m.eval()
with torch.no_grad():
    out = m.forward(img)





plt.figure()
plt.imshow(img.numpy().squeeze(),vmin=0,vmax=1)
plt.figure()
plt.imshow(out.data.numpy().squeeze() ,vmin=0,vmax=1)






#### Test the network  

m = CDNCNN()
m = torch.load('dncnnmodel_50.pt')


img = np.zeros((256,256))
img[:,0:128] += 1
img = img + np.random.normal(0, sigma/255.0, img.shape)

img = (dtype(img).reshape([1,1,256,256]))
m.eval()
with torch.no_grad():
    out = m.forward(img)



plt.figure()
plt.imshow(img.numpy().squeeze(),vmin=0,vmax=1)
plt.figure()
plt.imshow(out.data.numpy().squeeze() ,vmin=0,vmax=1)