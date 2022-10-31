"""
Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation

Dae-Young Song†, Geonsoo Lee, HeeKyung Lee, Gi-Mun Um, and Donghyeon Cho*
Chungnam National University & Electronic and Telecommunications Research Institute (ETRI)
†: Source Code Author, *: Corresponding Author

Copyright 2022. ETRI Allright reserved.


3-Clause BSD License(BSD-3-Clause)

SPDX short identifier: BSD-3-Clause



Note: This license has also been called the "New BSD License" or "Modified BSD License". See also the 2-clause BSD License.



Copyright 2022 ETRI.



Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:



1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.



THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,

INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A  PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,

OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;

LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY  THEORY OF LIABILITY, WHETHER IN CONTRACT,

STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF  THIS SOFTWARE,

EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


You are eligible to use this source according to licenses described as above ONLY WHEN YOU ARE USING THIS CODE FOR NON-COMMERCIAL PURPOSE.
If you want to use and/or redistribute this source commercially, please consult lhk95@etri.re.kr for details in advance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import BaseNet


class UNet(BaseNet):
    """U-Net Metaclass"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c7 = None

    def getName(self) -> str:
        """Get the name of the network"""
        return self.__class__.__name__

    def downfeature_size(self):
        """Get the size of the encoding map"""
        return self.c7

    def get_downfeature(self):
        """Get the encoding map"""
        return self.downfeature

    def unet_step_init(self):
        """Initialize the encoding map stacks"""
        self.downfeature1 = self.empty_tensor()
        self.downfeature2 = self.empty_tensor()
        self.downfeature3 = self.empty_tensor()
        self.downfeature4 = self.empty_tensor()
        self.downfeature5 = self.empty_tensor()
        self.downfeature6 = self.empty_tensor()
        self.downfeature = self.empty_tensor()

    @staticmethod
    def encode(image, conv_layer, store):
        """Encode once the feature"""
        feature = conv_layer(image)
        store = torch.cat([store, feature], dim=1)
        return feature, store

    @staticmethod
    def decode(feature, upconv, iconv, downfeature):
        """Decode once the feature"""
        f = upconv(feature)
        cat_f = torch.cat([downfeature, f], dim=1)
        return iconv(cat_f)

    def downsample(self, inputs):
        """Downsample the input image"""
        for image in inputs:
            f, self.downfeature1 = self.encode(image, self.downconv_1, self.downfeature1)  # [c1 * N X h/2 X w/2]
            f, self.downfeature2 = self.encode(f, self.downconv_2, self.downfeature2)  # [c2 * N X h/4 X w/4]
            f, self.downfeature3 = self.encode(f, self.downconv_3, self.downfeature3)  # [c3 * N X h/8 X w/8]
            f, self.downfeature4 = self.encode(f, self.downconv_4, self.downfeature4)  # [c4 * N X h/16 X w/16]
            f, self.downfeature5 = self.encode(f, self.downconv_5, self.downfeature5)  # [c5 * N X h/32 X w/32]
            f, self.downfeature6 = self.encode(f, self.downconv_6, self.downfeature6)  # [c6 * N X h/64 X w/64]
            _, self.downfeature = self.encode(f, self.downconv_7, self.downfeature)  # [c7 * N X h/128 X w/128]

    def upsample(self):
        """
        must be located after self.downsample
        upsampling only for getting "weight" values
        final value will be memorized.
        """
        f = self.decode(self.downfeature, self.upconv_7, self.conv_7, self.downfeature6)
        f = self.decode(f, self.upconv_6, self.conv_6, self.downfeature5)
        f = self.decode(f, self.upconv_5, self.conv_5, self.downfeature4)
        f = self.decode(f, self.upconv_4, self.conv_4, self.downfeature3)
        f = self.decode(f, self.upconv_3, self.conv_3, self.downfeature2)
        f = self.decode(f, self.upconv_2, self.conv_2, self.downfeature1)
        iconv_1 = self.decode(f, self.upconv_1, self.conv_1, self.empty_tensor())
        return iconv_1

    def forward(self, images):
        """U-Net Forward Pass"""
        self.unet_step_init()
        self.downsample(images)
        iconv_1 = self.upsample()
        return iconv_1

    @staticmethod
    def downconv_single(in_dim, out_dim, kernel, stride=2):
        """Get Downsampling Convolutional Layer"""
        conv_down_block = []
        # [H, W] -> [H/2, W/2]
        conv_down_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride,
                                      padding=int((kernel - 1) / 2), bias=False),
                            nn.BatchNorm2d(out_dim, momentum=0.1),
                            nn.ELU()]
        return nn.Sequential(*conv_down_block)

    @staticmethod
    def upconv_single(in_dim, out_dim, kernel, ratio=2, interpolation='bilinear'):
        """Get Upsampling Convolutional Layer"""
        conv_up_block = []
        # [H, W] -> [2H, 2W]
        conv_up_block += [Upsize(ratio, interpolation, False),
                          nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=1),
                          nn.BatchNorm2d(out_dim, momentum=0.1),
                          nn.ELU()]
        return nn.Sequential(*conv_up_block)

    @staticmethod
    def downconv_double(in_dim, out_dim, kernel):
        """Get Doubled Downsampling Convolutional Layers"""
        conv_down_block = []
        # [H, W] -> [H/2, W/2]
        conv_down_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=1, padding=int((kernel - 1) / 2),
                                      bias=False),
                            nn.BatchNorm2d(out_dim, momentum=0.1),
                            nn.ELU()]
        conv_down_block += [nn.Conv2d(out_dim, out_dim, kernel_size=kernel, stride=2, padding=int((kernel - 1) / 2),
                                      bias=False),
                            nn.BatchNorm2d(out_dim, momentum=0.1),
                            nn.ELU()]
        return nn.Sequential(*conv_down_block)

    @staticmethod
    def upconv_double(in_dim, out_dim, kernel, ratio=2, interpolation='bilinear'):
        """Get Doubled Upsampling Convolutional Layers"""
        conv_up_block = []
        # [H, W] -> [2H, 2W]
        conv_up_block += [Upsize(ratio, interpolation),
                          nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=1),
                          nn.BatchNorm2d(out_dim, momentum=0.1),
                          nn.ELU()]
        conv_up_block += [nn.Conv2d(out_dim, out_dim, kernel, stride=1, padding=1),
                          nn.BatchNorm2d(out_dim, momentum=0.1),
                          nn.ELU()]
        return nn.Sequential(*conv_up_block)

    @staticmethod
    def identity_conv(in_dim, out_dim):
        """Get Identity Convolutional Layer"""
        conv_block = []
        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(out_dim, momentum=0.1),
                       nn.ELU()]
        return nn.Sequential(*conv_block)


class Upsize(nn.Module):
    """Upsizing Module"""
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Upsize, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, img):
        """Upsizing Forward Pass"""
        _, _, h, w = img.shape
        nh, nw = h*2, w*2
        return F.interpolate(img, [nh, nw], mode=self.mode, align_corners=self.align_corners)
