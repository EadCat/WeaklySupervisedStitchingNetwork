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
import torch.nn.functional as F

from networks.base import BaseNet
from networks.unet.builder import build_unet
from networks.homography.builder import build_regressor
from normalization.manipulator import *


class MetaStitcher(BaseNet):
    """
    Stitcher Module to be inherited
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_direction = 3
        self.homography = self.opt.homography
        self.UNet = build_unet(self.opt, self.device)
        self.Regressor = build_regressor(self.opt)
        self.weight_block = self.get_weight_block(16)

        # self.denormalizer = ImageDenormalize(self.opt, self.opt.mean, self.opt.std)
        self.normalizer = ImageNormalize(self.opt, self.opt.mean, self.opt.std)

    def forward(self, data: dict, infer: bool):
        """
        forward method
        set infer = False during inference
        """
        if not infer:
            return self.learn(data)  # Dictionary
        else:
            return self.test(data)   # torch.Tensor [B x C x H x W]

    def learn(self, *args, **kwargs):
        """
        Abstract method
        """
        raise NotImplementedError

    def test(self, *args, **kwargs):
        """
        Abstract method
        """
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        """
        Abstract method
        """
        raise NotImplementedError

    def get_weight_block(self, in_dim):
        """
        Get Weight Map Generation Sub-Module
        """
        weight_block = []
        weight_block += [nn.Conv2d(in_dim, self.homography * self.input_direction,
                                   kernel_size=3, stride=1, padding=1),
                         nn.Softmax(dim=1)]
        return nn.Sequential(*weight_block)

    def get_displace_block(self, in_dim):
        """
        Get Local-Adjustment Regression Sub-Module
        """
        disp_block = []
        disp_block += [nn.Conv2d(in_dim, 2 * self.homography * self.input_direction,
                                 kernel_size=3, stride=1, padding=1),
                       nn.Tanh()]
        return nn.Sequential(*disp_block)

    @staticmethod
    def get_correct_block(in_dim, out_dim):
        """
        Get Color Correction Sub-Module
        """
        # For Pixel-wise Color Correction
        correct_block = [nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(in_dim, momentum=0.1),
                         nn.ELU(),

                         nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(out_dim, momentum=0.1),
                         nn.ELU(),

                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.Tanh()]
        return nn.Sequential(*correct_block)

    @staticmethod
    def flow_estimation(theta: torch.Tensor, imgsize):
        """
        Get Flow-Map from Warping Map
        """
        flow_list = []
        h, w = imgsize
        n, c, *_ = theta.shape

        for i in range(c):
            flow = F.affine_grid(theta[:, i, :, :], size=[n, c, h, w], align_corners=True)
            flow_list.append(flow)

        current_flow = torch.cat(flow_list, dim=3)  # [B x H x W x Homography * 2(XY)]

        return current_flow

    @staticmethod
    def warp(flow_map, image) -> torch.Tensor:
        """
        :param flow_map: [B x H x W x 2(XY) * Homography]
        :param image: [B x C x H x W]
        :return: Warped Images [Homography x B x C x H x W]
        """
        warped_list = []
        *_, h = flow_map.shape
        homo_num = h // 2

        for i in range(homo_num):
            warped_image = F.grid_sample(image, flow_map[..., 2*i: 2*i+2], padding_mode='border', align_corners=True)
            warped_list.append(warped_image)
        warped_list = torch.stack(warped_list)
        return warped_list

    def weighted_sum(self, images, weights):
        """
        :param images: [Homography x B x C x H x W]
        :param weights: [B x Homography x H x W]
        :return: Blended Image [B x C x H x W]
        """
        _, *size = images.shape
        output = torch.zeros(size).to(self.device)
        for i, img in enumerate(images):
            output += img * weights[:, [i], ...].repeat(1, 3, 1, 1)
        return output

    def correct(self, img, color_map):
        """
        img: [B x C x H x W]
        color_map: [B x C x H x W]
        """
        # img = self.denormalizer(img)
        img = denormalizer(img, 1.)
        img = img + color_map * img * (1 - img)  # I + aI(1-I)
        img = self.normalizer(img)
        if 'normalize' in self.opt.transform:
            img = normalizer(img, self.opt.mean, self.opt.std)
        return img





