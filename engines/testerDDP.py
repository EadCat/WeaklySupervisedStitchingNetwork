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
# Custom Library
import torch

from engines.tester import Tester

# PyTorch for multi-processing
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


class DistributedTester(Tester):
    """
    Launch Multi-Processes (DDP) for inference
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unused_params = False

    def build_generator(self, rank, verbose):
        """
        Wrap Stitcher with DistributedDataParallel
        """
        model = super().build_generator(rank, verbose)
        if model:
            return DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=self.unused_params)
        else:
            return None

    @staticmethod
    def distributed_test(rank, engine):
        """
        Launch a Unit Process
        """
        engine.distribution_setup(rank)
        engine.test(rank)

    @torch.no_grad()
    def main(self):  # Spawn Multi-Process
        """
        Trigger Multi-Processes
        """
        mp.spawn(self.distributed_test, nprocs=self.num_gpu, args=(self,))