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
from .HomoDispNet import HomoDispNet
from networks.base import finish_merge


class DoubleColorCorNet(HomoDispNet):
    """
    Stitching Network with Pre & Post Color Correction
    """
    @finish_merge
    def __init__(self, *args, **kwargs):
        """
        The Final Model for WSSN
        """
        super().__init__(*args, **kwargs)
        self.pre_correct_block = self.get_correct_block(16, 3 * self.input_direction)
        self.post_correct_block = self.get_correct_block(16, 3)

    def generate(self, images: torch.Tensor) -> dict:
        """
        :param images: [input direction x B x C x H x W]
        :return: Panorama Image [B x C x H x W]
        """
        _, b, c, h, w = images.shape
        iconv_1 = self.UNet(images)
        downfeature = self.UNet.get_downfeature()

        weight = self.weight_block(iconv_1)
        theta = self.Regressor(downfeature)  # theta: [Batch X homography * input_direction X 2 X 3]
        disp = self.local_limit * self.local_adj_block(iconv_1)  # [B x 2 * homography * input_direction x H x W]
        disp = disp.permute(0, 2, 3, 1)  # [B x H x W x 2 * homography * input_direction]
        pre_correct_map = self.pre_correct_block(iconv_1)
        post_correct_map = self.post_correct_block(iconv_1)

        panorama = torch.zeros([b, c, h, w]).to(self.device)

        # warped_image_list = []
        for i, img in enumerate(images):
            start, end = i * self.homography, i * self.homography + self.homography
            flow = self.flow_estimation(i, disp, theta[:, start: end, :, :], imgsize=[h, w])
            img = self.correct(img, pre_correct_map[:, i*self.input_direction: i*self.input_direction+3])
            warped_images = self.warp(flow, img)
            # warped_image_list.append(warped_images)
            panorama += self.weighted_sum(warped_images, weight[:, start: end, ...])

        # warped_image_list = torch.stack(warped_image_list)
        panorama = self.correct(panorama, post_correct_map)

        return {'panorama': panorama}

