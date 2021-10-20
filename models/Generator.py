import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, ks=3, stride=1, padding=0, dilation=1, pad_mode='zeros', norm=nn.BatchNorm2d):
    return [nn.Conv2d(in_channels, out_channels, ks, stride, padding, dilation, padding_mode=pad_mode),
            norm(out_channels),
            nn.ReLU()]

def conv_trans_block(in_channels, out_channels, ks=3, stride=1, padding=0, out_padding=0, dilation=1, pad_mode='zeros', norm=nn.BatchNorm2d):
    return [nn.ConvTranspose2d(in_channels, out_channels, ks, stride, padding, out_padding,
                                            dilation=dilation, padding_mode=pad_mode),
            norm(out_channels),
            nn.ReLU()]

class ResBlock(nn.Module):
    def __init__(self, channels, pad_mode='zeros', norm=nn.BatchNorm2d, use_dropout=False):
        super(ResBlock, self).__init__()
        ks = 3; p = 1 # Kernel size and padding
        
        block = [nn.Conv2d(channels, channels, kernel_size=ks, padding=p, padding_mode=pad_mode),
                 norm(channels),
                 nn.ReLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.Conv2d(channels, channels, kernel_size=ks, padding=p, padding_mode=pad_mode),
                  norm(channels)]
        self.conv_block = nn.Sequential(*block)
    
    def forward(self, xb):
        out = self.conv_block(xb) + xb
        return out

class CycleGen(nn.Module):
    def __init__(self, in_channels_t, out_channels_t, pipe_channels, num_res, norm, use_dropout):
        super(CycleGen, self).__init__()
        if norm == 'batch':
            normalization = nn.BatchNorm2d
        elif norm == 'instance':
            normalization = nn.InstanceNorm2d
        sampling = 2

        # first convolutional block
        net = conv_block(in_channels_t, pipe_channels, ks=7, padding=3, pad_mode='reflect', norm=normalization)
        
        # downsampling
        for i in range(sampling):
            ratio = 2 ** i
            net += conv_block(pipe_channels * ratio, pipe_channels * ratio * 2, stride=2, padding=1, norm=normalization)
        
        # resnet blocks
        ratio = 2 ** sampling
        for i in range(num_res):
            net += [ResBlock(pipe_channels * ratio, pad_mode='reflect', norm=normalization, use_dropout=use_dropout)]

        # upsampling
        for i in range(sampling):
            ratio = 2 ** (sampling - i)
            net += conv_trans_block(pipe_channels * ratio, pipe_channels * ratio // 2, stride=2, padding=1, out_padding=1)

        # final convolutional block
        net += [nn.Conv2d(pipe_channels, out_channels_t, kernel_size=7, padding=3, padding_mode='reflect')]
        
        # activation
        net += [nn.Tanh()]

        self.net = nn.Sequential(*net)
    
    def forward(self, xb):
        return self.net(xb)