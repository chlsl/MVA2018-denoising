       
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