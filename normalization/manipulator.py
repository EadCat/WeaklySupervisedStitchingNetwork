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
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as funct


def normalizer(target: Union[torch.Tensor, np.ndarray], mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Normalize Distribution"""
    if isinstance(target, np.ndarray):
        h, w, _ = target.shape
        mean = np.array(mean)  # (c, )
        std = np.array(std)  # (c, )
        target = (target - mean) / std
    else:  # torch.Tensor
        target = funct.normalize(target, mean, std)

    return target


def denormalizer(target: Union[torch.Tensor, np.ndarray],
                 # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device='cpu',
                 magnification=255.0):
    """Stretch Distribution"""
    if isinstance(target, np.ndarray): # Numpy Processing
        if target.ndim == 3:  # Gray or RGB
            h, w, c = target.shape
        elif target.ndim == 2:  # Gray
            h, w = target.shape
            c = 1
        else:
            import sys
            print(f"target ndim: {target.ndim}. illegal format. please arange in [2(gray) ~ 3(rgb)]")
            print(f'target shape: {target.shape}')
            sys.exit()

        # Get Channel-wise min & max
        tmin, tmax = [], []
        for i in range(c):  # get channel(depth)-wise minimum & maximum
            tmin.append(target[..., i].min())
            tmax.append(target[..., i].max())
        tmin, tmax = np.array(tmin), np.array(tmax)  # shape: (c, )

    else:  # PyTorch Tensor Processing
        if target.ndim == 5:  # [Homography x Batch x Channel x Height x Width]
            _, _, c, h, w = target.shape
        elif target.ndim == 4:  # [Batch x Channel x Height x Width]
            _, c, h, w = target.shape
        elif target.ndim == 3:  # [Channel x Height x Width]
            c, h, w = target.shape
        else:
            import sys
            print(f"target ndim: {target.ndim}. illegal format. please arange in [3 ~ 5]")
            print(f'target shape: {target.shape}')
            sys.exit()
        tmin, tmax = [], []
        for i in range(c):  # get channel-wise minmax
            tmin.append(target[..., i, :, :].min())
            tmax.append(target[..., i, :, :].max())
        tmin, tmax = torch.tensor(tmin), torch.tensor(tmax)  # shape: (c, )
        tmin = tmin.unsqueeze(1).unsqueeze(2).repeat(1, h, w).to(target.device)  # shape: (c, h, w)
        tmax = tmax.unsqueeze(1).unsqueeze(2).repeat(1, h, w).to(target.device)  # shape: (c, h, w)

    return (target - tmin) / (tmax - tmin + 1e-8) * magnification  # finish


class ImageDenormalize(nn.Module):
    """Denormalize Distribution"""
    def __init__(self, opt, mean, std, conditional=True):
        super().__init__()
        self.opt = opt
        self.mean = [-m/s for m, s in zip(mean, std)]
        self.std = [1/s for s in std]
        self.conditional = conditional
        self.function = transforms.Normalize(mean=self.mean, std=self.std, inplace=False)

    def forward(self, input_tensor):
        """Conditional Denormalize"""
        if self.conditional:
            if 'normalize' in self.opt.transform:
                return self.function(input_tensor)
            else:
                return input_tensor
        else:
            return self.function(input_tensor)


class ImageNormalize(nn.Module):
    """Normalize Distribution"""
    def __init__(self, opt, mean, std, conditional=True):
        super().__init__()
        self.opt = opt
        self.mean = mean
        self.std = std
        self.conditional = conditional
        self.function = transforms.Normalize(mean=self.mean, std=self.std, inplace=False)

    def forward(self, input_tensor):
        """Conditional Normalize"""
        if self.conditional:
            if 'normalize' in self.opt.transform:
                return self.function(input_tensor)
            else:
                return input_tensor
        else:
            return self.function(input_tensor)
