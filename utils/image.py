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
import cv2
from normalization.manipulator import *
from typing import Union, List, Tuple


def grayscaler(img: torch.Tensor, keepdim=True, chn_dim=1) -> torch.Tensor:
    """
    make a grayscale image
    :param img: target RGB
    :param keepdim: return [H, W, 1]:True or [H, W]:False
    :return: grayscale image
    """
    gray = torch.mean(img, dim=chn_dim)
    if keepdim:
        gray = torch.unsqueeze(gray, dim=chn_dim)
    return gray


@torch.no_grad()
def save_image(opt, image: torch.Tensor, filename: Union[List, Tuple], ext='png'):
    """
    :param opt: Engine Option
    :param image: [B x C x H x W]
    :param filename: List of Filename
    :return:
    """
    sample_num = opt.sample_num
    batch_size = image.shape[0]
    # denormalizer = ImageDenormalize(opt, opt.mean, opt.std, conditional=True)

    if sample_num > batch_size:
        sample_num = batch_size
    if sample_num > len(filename):
        sample_num = len(filename)

    # image = denormalizer(image) * 255
    image = denormalizer(image, 255)
    samples = image.cpu()
    samples = np.transpose(np.array(samples), (0, 2, 3, 1))

    for i, (sample, name) in enumerate(zip(samples, filename)):
        if i == sample_num: break
        cv2.imwrite(name + f'.{ext}', sample[..., ::-1])
