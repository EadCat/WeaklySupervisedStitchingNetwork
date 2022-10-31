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
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    Homography Estimation Module
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.homography = opt.homography
        self.input_direction = 3
        if opt.unet.lower() == 'large':
            self.first_layer = self.input_direction * 512  # used in first layer input channel

    @staticmethod
    def conv_block(in_dim, out_dim, kernel_size=3, stride=1):
        """
        Build Convolutional Block [Conv, BatchNorm, ELU]
        """
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
                          nn.BatchNorm2d(out_dim, momentum=0.1),
                          nn.ELU()]

        return nn.Sequential(*conv_up_block)

    def _forward(self, feat):
        """
        feat: [B x 512 x 2 x 4]
        Get Parameters of Homography
        """
        x = feat
        # x: [Batch, 512, 2, 4]
        x = self.conv45(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv5a(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv5b(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv56(x)
        # x: [Batch, 512, 1, 1]
        x = self.conv6a(x)
        # x: [Batch, 512, 1, 1]
        x = self.conv6b(x)
        # x: [Batch, 512, 1, 1]
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))

        # x: [Batch, 512]
        x = x.view(-1, x.shape[1])

        # theta: [Batch, 6 * 2]
        theta = self.fc(x)

        return theta

    def forward(self, feat):
        """
        feat: [B x 512 x 2 x 4]
        theta -> [B x homography*direction x 2 x 3]
        """
        theta = self._forward(feat)
        # theta: [Batch X 6 X input_direction X homography]
        theta = theta.view(-1, self.homography * 3, 2, 3)
        # theta: [Batch X homography * input_direction X 2 X 3]
        return theta
