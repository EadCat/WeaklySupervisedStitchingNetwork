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
from typing import Union
import numpy as np
import torch
from termcolor import colored

from .inspection import inspect_grayscale


def zero_masking(img: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray],
                 minimum=(0, 0, 0)):
    """
    * assumptions:
    img: range(x, y)
    mask: "0" or somevalue, binary. "0" will be assumed as False

    if np.ndarray: [Height X width X Channels]
    if torch.Tensor: [* X Channels X Height X Width]

    * inputs
    :param img: target image to be masked (RGB)
    :param mask: masking area (RGB or Grayscale)
    :param minimum: theoretically possible minimum value of distribution (default 0) -> masking value
    """
    assert(type(img) == type(mask)), \
        f"input array's types are must be equal. img:{colored(type(img), 'red')} | mask:{colored(type(mask), 'red')}"
    minimum = np.array(minimum)
    min_ndim = len(minimum)

    if isinstance(img, np.ndarray):
        cimg = np.copy(img)  # Copy because of reference call
        if cimg.ndim > mask.ndim:  # [H X W] -> [H X W X C]
            mask = np.expand_dims(mask, axis=2).repeat(min_ndim, axis=2)
        gray = True if inspect_grayscale(mask) else False
        for i, mini in enumerate(minimum):
            if gray: m = 0
            else: m = i
            cimg[..., i][mask[..., m] == 0] = mini

    else:
        cimg = img.clone()  # Copy because of reference call
        gray = True if inspect_grayscale(mask) else False
        for i, mini in enumerate(minimum):
            if gray: m = 0
            else: m = i
            if cimg.ndim == 3:  # [C X H X W]
                cimg[i, ...][mask[m, ...] == 0] = mini
            elif cimg.ndim == 4:  # [B X C X H X W]
                cimg[:, i, ...][mask[:, m, ...] == 0] = mini
            elif cimg.ndim == 5:  # [# of img X B X C X H X W]
                cimg[:, :, i, ...][mask[:, :, m, ...] == 0] = mini
            elif cimg.ndim == 6:  # [# of img X homo X B X C X H X W]
                cimg[:, :, :, i, ...][mask[:, :, :, m, ...] == 0] = mini

        return cimg


def region_divider(masks: torch.Tensor):
    """
    :param masks: [num of inputs x B x Gray x H x W]
    :return:
    """
    # [1, 3, 5] -> [3, 5, 1]
    rollin_mask = torch.roll(masks, -1, dims=0)

    # [num of inputs x B x Gray x H x W]
    overlaps = torch.stack([torch.logical_and(mask1, mask2) for mask1, mask2 in zip(masks, rollin_mask)])
    nonoverlaps = torch.stack([torch.logical_xor(mask, overlap) for mask, overlap in zip(masks, overlaps)])

    return overlaps, nonoverlaps
