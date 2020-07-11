import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob


###Patch Dimension = N X C X W X H , Original Image Dimension = N1 X C1 X W1 X H1

class bn_relu_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(bn_relu_conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(nin)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.relu(out)
        out = self.conv(out)

        return out

class bottleneck_layer(nn.Sequential):
  def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(bottleneck_layer, self).__init__()
      
      self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=growth_rate*4, kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('conv_3x3', bn_relu_conv(nin=growth_rate*4, nout=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
  def forward(self, x):
      bottleneck_output = super(bottleneck_layer, self).forward(x)
      if self.drop_rate > 0:
          bottleneck_output = F.dropout(bottleneck_output, p = self.drop_rate, training=self.training)
          
      bottleneck_output = torch.cat((x, bottleneck_output), 1)
      
      return bottleneck_output

class Transition_layer(nn.Sequential):
  def __init__(self, nin, theta=0.5):    
      super(Transition_layer, self).__init__()
      
      self.add_module('conv_1x1', bn_relu_conv(nin=nin, nout=int(nin*theta), kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=1, padding=0))

class DenseBlock(nn.Sequential):
  def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
      super(DenseBlock, self).__init__()
                        
      for i in range(num_bottleneck_layers):
          nin_bottleneck_layer = nin + growth_rate * i
          self.add_module('bottleneck_layer_%d' % i, bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=48, num_layers=40, theta=0.5, drop_rate=0.2, num_classes=2):
        super(DenseNet, self).__init__()
        
        assert (num_layers - 4) % 6 == 0
        
        # (num_layers-4)//6 
        num_bottleneck_layers = (num_layers - 4) // 6
        
        # 96 x 96 x 3 --> 96 x 96 x (growth_rate*2) --> 96 X 96 X 96
	# 24 x 24 x 3 --> 24 x 24 x (growth_rate*2) --> 24 X 24 X 12
        self.dense_init = nn.Conv2d(3, growth_rate*2, kernel_size=3, stride=1, padding=1, bias=True)
                
        # 96 x 96 x (growth_rate*2) --> 96 x 96 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)] ---> 96 X 96 X 864
        self.dense_block_1 = DenseBlock(nin=growth_rate*2, num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 96 x 96 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)] --> 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]*theta ---> 32 X 32 X 432 or 12 X 12 X 9
        nin_transition_layer_1 = (growth_rate*2) + (growth_rate * num_bottleneck_layers)  ###864 or 18
        self.transition_layer_1 = Transition_layer(nin=nin_transition_layer_1, theta=theta)
        
        # 32 x 32 x nin_transition_layer_1*theta --> 32 x 32 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]  ---> 32*32*1200 or 12*12*15
        self.dense_block_2 = DenseBlock(nin=int(nin_transition_layer_1*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 32 x 32 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)] --> 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]*theta ---> 16*16*600 /6*6*8
        nin_transition_layer_2 = int(nin_transition_layer_1*theta) + (growth_rate * num_bottleneck_layers) ###1200, 15
        self.transition_layer_2 = Transition_layer(nin=nin_transition_layer_2, theta=theta)
        
        # 16 x 16 x nin_transition_layer_2*theta --> 16 x 16 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)] ----> 16*16*1368
        self.dense_block_3 = DenseBlock(nin=int(nin_transition_layer_2*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        nin_fc_layer = int(nin_transition_layer_2*theta) + (growth_rate * num_bottleneck_layers) 
        
        # [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)] --> num_classes
        self.fc_layer = nn.Linear(nin_fc_layer, num_classes)
        
    def forward(self, x):
        dense_init_output = self.dense_init(x) #Dense Block initialization with 6 botteleneck layers each
        
        dense_block_1_output = self.dense_block_1(dense_init_output)
        transition_layer_1_output = self.transition_layer_1(dense_block_1_output)
        
        dense_block_2_output = self.dense_block_2(transition_layer_1_output)
        transition_layer_2_output = self.transition_layer_2(dense_block_2_output)
        
        dense_block_3_output = self.dense_block_3(transition_layer_2_output)

        return dense_block_3_output

###DEFINE ATTENTION NETWORK###

###Channel and Spatial attention code from https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py###

# It consisits of Convolution + Batch_normalization + ReLU
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
           x = self.bn(x)
        if self.relu is not None:
           x = self.relu(x)
        return x

# This will make [batch_size, channel, height, width] to [batch_size, channel*height*width]
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
        
# it will provide the concatenation of max pool and average pool across the channels
# if x is [batch_sz, C, H, W] then output is [batch_sz, 1, H, W]+[batch_sz, 1, H, W], that is [batch_sz, 2, H, W]        
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        
                
# Channel Attention module which will return channel attentive feature
class Channel_Attention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types=['avg', 'max']): ##Changing the reduction ratio from 16 to 2
        super(Channel_Attention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
        
# Spatial Attention module which will return spatial attentive feature        
class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        ###kernel_size is selected according to the formula, out_size = ((in_size-kernel_size+2xpadding)/stride)+1
        # To keep the in_size and out_size same, we are getting kernel_size 3
        kernel_size = 3 
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

###Patch Attention Model which is supposed to highlight the best patch
class Patch_Attention(nn.Module):
    def __init__(self, patches=49, reduction_ratio=10):
        super(Patch_Attention, self).__init__()
        self.patches = patches
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(patches, patches // reduction_ratio),
            nn.ReLU(),
            nn.Linear(patches // reduction_ratio, patches)
            )
    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = self.mlp( x )
        if channel_att_sum is None:
          channel_att_sum = channel_att_raw
        else:
          channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

  
class Attention_Network(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=12, pool_types=['avg', 'max'], spatial=True):
        super(Attention_Network, self).__init__()
        self.Channel_Attention = Channel_Attention(gate_channels, reduction_ratio, pool_types)
        self.spatial=spatial
        if spatial:
           self.Spatial_Attention = Spatial_Attention()
        self.Patch_Attention = Patch_Attention(gate_channels, reduction_ratio)
    def forward(self, x):
        x_out = self.Channel_Attention(x)
        if self.spatial:
           x_out = self.Spatial_Attention(x_out)
	return x_out


##########COMBINED MODEL CODE####################################################

class Model(nn.Module):
    def __init__(self,growth_rate=48, num_layers=40, theta=0.5, drop_rate=0.2, num_classes=2,patches=49,reduction_ratio =12):
        super(Model, self).__init__()
        assert (num_layers - 4) % 6 == 0
	num_classes = 2
        self.f_1 = DenseNet(growth_rate=growth_rate, num_layers= num_layers, theta=theta, drop_rate=drop_rate, num_classes=num_classes)
        self.f_2 = DenseNet(growth_rate=growth_rate//4, num_layers= num_layers//4, theta=theta, drop_rate=drop_rate, num_classes=num_classes)
        num_bottleneck_layers = ((num_layers//4) - 4) // 6
        nin_transition_layer_1 = ((growth_rate//4)*2) + ((growth_rate//4) * num_bottleneck_layers) 
        nin_transition_layer_2 = int(nin_transition_layer_1*theta) + ((growth_rate//4) * num_bottleneck_layers) 
	gate_channels = int(nin_transition_layer_2*theta) + ((growth_rate//4) * num_bottleneck_layers) 
	self.Attention_Network = Attention_Network(gate_channels, reduction_ratio=reduction_ratio)
	self.Patch_Attention = Patch_Attention(patches = patches,reduction_ratio =(reduction_ratio//2))
	int_size = int(528+gate_channels)
        self.lin_layer = nn.Linear(int_size, num_classes)

    def forward(self,x,y) :
        dense_model1_output = self.f_1(x) ###Original Output size (1X528X94X94)
        dense_model2_output = self.f_2(y)
        output_size = dense_model2_output.size()
	attention_input = dense_model2_output[0].unsqueeze(0)
	attention_output = self.Attention_Network(attention_input)
	attention_output =  F.adaptive_avg_pool3d(attention_output, (1,1, 1))
	patches = int((dense_model2_output.size())[0])
	for i in range(1,patches):
		attention_input = dense_model2_output[i].unsqueeze(0)
        	attention_output_patch = self.Attention_Network(attention_input) ###gate channels?? ---> output will be of dimension NXCXHXW (63 X 15 X 22 X 22) in our case
		attention_output_patch =  F.adaptive_avg_pool3d(attention_output_patch, (1,1, 1))
		attention_output = torch.cat((attention_output,attention_output_patch),0)
	##Now adding the patch attention network after we compute the channel and spatial attention for all the N patches and we get a outpt vector size of 160X3X94X94
	patch_attention_input = torch.reshape((attention_output),(1,patches,1,1))
	patch_attention_output = self.Patch_Attention(patch_attention_input)
	patch_attention_output = torch.reshape((patch_attention_output),(patches,1,1,1))
	final_attention_output = torch.mul(dense_model2_output,patch_attention_output) ###Multiplying the output of Patch attention with the input
        ###Combining both the original model and the patch based network### ---> Implies combining (1X528X94X94) with (63 X 15 X 22 X 22) we want ----> (64 X 543 X 2 X 2)
        model1_avg_pool_output = F.adaptive_avg_pool2d(dense_model1_output, (1, 1)) ## 1 X 528 X 1 X 1 
        patch_avg_pool_output =  F.adaptive_avg_pool2d(final_attention_output, (1, 1)) ## 63 X 15 X 1 X 1 
        patch_model_output = (torch.mean(patch_avg_pool_output,0)).unsqueeze(0) ### 1 X 15 X 1 X 1
        patch_model_output = patch_model_output.view(patch_model_output.size(0), -1) ## 1 X 15
        original_output = model1_avg_pool_output.view(model1_avg_pool_output.size(0), -1) ## 1 X 528
        combined_out = torch.cat((original_output,patch_model_output),dim = 1)
	out_flat = combined_out.view(combined_out.size(0), -1)
	in_size = out_flat.size()[1]
	num_classes = 2
	out = self.lin_layer(out_flat)
        return out


