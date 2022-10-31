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


def to_cuda(data: dict, device) -> dict:
    """
    Move data to cuda device
    """
    new_dict = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            new_dict.update({k: v.to(device)})
        elif isinstance(v, dict):
            sub_dict = to_cuda(v, device)
            new_dict.update({k: sub_dict})
        else:
            new_dict.update({k: v})

    del data
    return new_dict


def permute_direction(data: dict) -> dict:
    """
    [Batch x Direction x Channel x Height x Width] -> [Direction x Batch x Channel x Height x Width]
    """
    new_dict = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            new_dict.update({k: detect_5dim(v)})
        elif isinstance(v, dict):
            sub_dict = permute_direction(v)
            new_dict.update({k: sub_dict})
        else:
            new_dict.update({k: v})

    del data
    return new_dict


def detect_5dim(data: torch.Tensor) -> torch.Tensor:
    """
    [Batch x Direction x Channel x Height x Width] -> [Direction x Batch x Channel x Height x Width]
    """
    if data.ndim == 5:
        return data.permute(1, 0, 2, 3, 4).contiguous()
