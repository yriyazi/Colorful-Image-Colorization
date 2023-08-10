import torch 
import torch.nn as nn
import utils

utils.padding=1
utils.stride=1
# Mixing the convolution and bath normalization
class TwoLayerConv(nn.Module):
    def __init__(self,
                 in_channels    :int,
                 out_channels   :int,
                 kernel_size    :int=3,
                 stride         :int=1,
                 padding        :int=1,
                #  groups         :int=1,
                 bias           :bool=True)-> None:
        
        super().__init__()
        self.relu = nn.ReLU()
        
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=padding,
                            #   groups=groups,
                              bias=bias)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=2, 
                              padding=padding,
                            #   groups=groups,
                              bias=bias)
        
        self.BatchNormalization = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return self.BatchNormalization(out)
    
class ThreeLayerConv(nn.Module):
    def __init__(self,
                 in_channels    :int ,
                 out_channels   :int ,
                 dilation       :list = [1,1,1],
                 padding        :int = [1,1,1] ,
                 stride         :list = [1,1,1] ,
                 groups         :int = 1 ,
                 kernel_size    :int = 3 ,
                 bias           :bool = True)-> None:
        
        super().__init__()
        """
        Dilated Convolutions are a type of convolution that “inflate” the kernel by inserting holes
        between the kernel elements. An additional parameter (dilation rate) indicates how much the
        kernel is widened. There are usually

        spaces inserted between kernel elements.

        Note that concept has existed in past literature under different names, for instance the 
        algorithme a trous, an algorithm for wavelet decomposition
        (Holschneider et al., 1987; Shensa, 1992).
        https://paperswithcode.com/method/dilated-convolution
        
        
        
        """
        self.relu = nn.ReLU()
            
        self.conv1 = nn.Conv2d(in_channels  = in_channels,
                              out_channels  = out_channels,
                              kernel_size   = kernel_size,
                              
                              dilation      = dilation[0],
                              stride        = stride[0], 
                              padding       = padding[0],
                              
                              groups        = groups,
                              bias          = bias)
        
        self.conv2 = nn.Conv2d(in_channels  = out_channels,
                              out_channels  = out_channels,
                              kernel_size   = kernel_size,
                              
                              dilation      = dilation[1],
                              stride        = stride[1], 
                              padding       = padding[1],
                              
                              groups        = groups,
                              bias          = bias)
        
        self.conv3 = nn.Conv2d(in_channels  = out_channels,
                              out_channels  = out_channels,
                              kernel_size   = kernel_size,
                              
                              dilation      = dilation[2],
                              stride        = stride[2], 
                              padding       = padding[2],
                              
                              groups        = groups,
                              bias          = bias)
              
        self.BatchNormalization = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        return self.BatchNormalization(out)