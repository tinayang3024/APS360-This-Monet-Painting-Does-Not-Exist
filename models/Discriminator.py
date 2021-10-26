def patch_size_cal(layer_info):
    n = 4 # from the last layer
    for kernal_size, stride in reversed(layer_info):
        n = kernal_size + stride * (n-1)

    n = 4 + 2 * (n-1) # the first layer
    return n
  
layers = [(4,2), (4,2), (4,1)]
print(patch_size_cal(layers))

import torch
import torch.nn as nn

class MonaiDiscriminator(nn.Module):
    """Adapted from a PatchGAN discriminator"""

    def __init__(self, input_nc, layer_info, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            input_nc (int)  -- the number of channels in input images
            layer_info(list)-- a list of 2 elements tuples (kernal_size, stride) for each conv layer.
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(MonaiDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        n = 1
        for kernal_size, stride in layer_info: 
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernal_size, stride=stride, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            n += 1

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

d = MonaiDiscriminator(3, layers)
