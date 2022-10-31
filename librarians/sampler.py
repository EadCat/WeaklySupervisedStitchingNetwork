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
import os
from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from .librarian import model_check


class ValidSampler(nn.Module):
    """
        * ====== PyTorch Experiment Support ====== *
        * ============= Sample Agent ============= *
        Dae-Young Song from Chungnam National University
        2022. 05. 08.
        """
    def __init__(self, opt, rank, model, sample_function, fmt='png'):
        super().__init__()
        self.opt = opt
        self.device = torch.device(rank)
        self.save_fmt = fmt

        self.to(self.device)
        # Link Parameters (call by reference)
        if isinstance(model, DistributedDataParallel) or isinstance(model, DataParallel):
            self.model = model.module
        else: self.model = model
        self.name = self.model.get_name()
        self.saveroot = os.path.join(self.opt.saveroot, self.opt.name)

        self.iter_sample_fq = opt.iter_sample_freq
        self.ep_sample_fq = opt.epoch_sample_freq

        self.sample_function = sample_function

    @model_check
    @torch.no_grad()
    def iter_sample(self, i, epoch, data: dict):
        """
        output sampling during iteration
        :param i: [0 ~ (data number - 1)]
        :param epoch: [1 ~ total_epoch]
        :param image: [C x H x W]
        :param Volume: Volume from Voxelization
        :param name: name of data
        :return: None
        """
        if self.iter_sample_fq > 0 and (i + 1) % self.iter_sample_fq == 0:
            output = self.model(data, infer=True)
            if output is not None:
                name = data['name']
                fname = self.sample_save_dir(epoch, name, i)
                self.sample_function(self.opt, output, fname)

    @model_check
    @torch.no_grad()
    def epoch_sample(self, epoch, data: dict):
        """
        output sampling during epoch
        :param epoch: [1 ~ total_epoch]
        :param image: [C x H x W]
        :param name: name of data
        :return: None
        """
        if self.ep_sample_fq > 0 and epoch % self.ep_sample_fq == 0:
            output = self.model(data, infer=True)
            if output is not None:
                name = data['name']
                fname = self.sample_save_dir(epoch, name)
                self.sample_function(self.opt, output, fname)

    @model_check
    def sample_save_dir(self, epoch, name: Union[List, Tuple], i=None):
        """Get Sampling directory"""
        sample_root = os.path.join(self.saveroot, 'samples')
        os.makedirs(sample_root, exist_ok=True)
        list_of_name = []
        for n in name:
            if i is None:
                target_name = os.path.join(sample_root, f'{self.name}-{str(epoch).zfill(3)}-{n}.{self.save_fmt}')
            else:
                target_name = os.path.join(
                    sample_root, f'{self.name}-{str(epoch).zfill(3)}-{str(i+1).zfill(4)}-{n}.{self.save_fmt}')
            list_of_name.append(target_name)
        return list_of_name
