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
import math
import torch
import torch.nn as nn


def finish_build(init_function):
    """
    Decorator for initializing a network
    """
    def init_wrapper(self, *args, **kwargs):
        init_function(self, *args, **kwargs)
        self.count_params()
        self.init_weights(self.opt)
        self.to(self.device)
    return init_wrapper


def finish_merge(init_function):
    """
    Decorator for merging a network
    """
    def init_wrapper(self, *args, **kwargs):
        init_function(self, *args, **kwargs)
        self.count_params()
        # self.init_weights(self.opt)
        self.to(self.device)
    return init_wrapper


class BaseNet(nn.Module):
    """
    Base Network to be inherited
    """
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.prime = True if device == 'cuda:0' or device == 'cpu' else False

        self.__num_params = 0
        self.__trainable_params = 0

    def get_name(self) -> str:
        """
        Get a name of the network
        """
        return self.__class__.__name__

    def count_params(self):
        """
        Count a number of parameters
        """
        self.__num_params = sum(p.numel() for p in self.parameters())
        self.__trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_params(self):
        """
        Get a number of parameters
        """
        return self.__num_params

    def get_tparams(self):
        """
        Get a number of trainable parameters
        """
        return self.__trainable_params

    def empty_tensor(self):
        """
        Pass Empty Tensor
        """
        return torch.tensor([]).to(self.device)

    def init_weights(self, opt):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        init_type = opt.init_model
        gain = opt.init_gain

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, *args, **kwargs):
        """
        Forward pass
        """
        raise NotImplementedError
