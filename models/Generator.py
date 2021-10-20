import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, ks=3, stride=1, padding=0, dilation=1, pad_mode='zeros', norm='batch'):
    if norm == 'batch':
        normalization = nn.BatchNorm2d
    elif norm == 'instance':
        normalization = nn.InstanceNorm2d
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, ks, stride, padding, dilation, padding_mode=pad_mode),
                         normalization(out_channels),
                         nn.ReLU())

def conv_trans_block(in_channels, out_channels, ks=3, stride=1, padding=0, out_padding=0, dilation=1, pad_mode='zeros', norm='batch'):
    if norm == 'batch':
        normalization = nn.BatchNorm2d
    elif norm == 'instance':
        normalization = nn.InstanceNorm2d
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, ks, stride, padding, out_padding,
                                            dilation=dilation, padding_mode=pad_mode),
                         normalization(out_channels),
                         nn.ReLU())

def res_block(channels, pad_mode='zeros', norm='batch'):
    ks = 3; p = 1
    if norm == 'batch':
        normalization = nn.BatchNorm2d
    elif norm == 'instance':
        normalization = nn.InstanceNorm2d
    return nn.Sequential(nn.Conv2d(channels, channels, kernel_size=ks, padding=p, padding_mode=pad_mode),
                         normalization(channels),
                         nn.ReLU(),
                         nn.Conv2d(channels, channels, kernel_size=ks, padding=p, padding_mode=pad_mode),
                         normalization(channels))

class CycleGen(nn.Module):
    def __init__(self, in_channels_t, out_channels_t, pipe_channels, norm):
        super(CycleGen, self).__init__()
        self.conv1 = conv_block(in_channels_t, pipe_channels, padding=3, pad_mode='reflect', norm=norm)
        self.conv2 = conv_block(pipe_channels, pipe_channels*2, padding=1, norm=norm)
        self.conv3 = conv_block(pipe_channels*2, pipe_channels*4, padding=1, norm=norm)
        self.res1 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.res2 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.res3 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.res4 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.res5 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.res6 = res_block(pipe_channels*4, pad_mode='reflect', norm=norm)
        self.deconv1 = conv_trans_block(pipe_channels*4, pipe_channels*2, padding=1, out_padding=1)
        self.deconv2 = conv_trans_block(pipe_channels*2, pipe_channels, padding=1, out_padding=1)
        self.final_conv = nn.Conv2d(pipe_channels, out_channels_t, kernel_size=3, padding=3, padding_mode='reflect')
        self.final_act = nn.Tanh()
    
    def forward(self, xb):
        yb = self.conv1(xb)
        yb = self.conv2(yb)
        yb = self.conv3(yb)
        yb = self.res1(yb) + yb
        yb = self.res2(yb) + yb
        yb = self.res3(yb) + yb
        yb = self.res4(yb) + yb
        yb = self.res5(yb) + yb
        yb = self.res6(yb) + yb
        yb = self.deconv1(yb)
        yb = self.deconv2(yb)
        yb = self.final_conv(yb)
        yb = self.final_act(yb)
        return yb