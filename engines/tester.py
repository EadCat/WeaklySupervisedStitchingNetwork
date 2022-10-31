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
import datetime
from termcolor import colored
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from engines.metaclass import MetaEngine, model_construction
from dataset.kandao_test import KandaoTestDataset
from librarians.record import snapshot_maker
from utils.time import *
from utils.image import save_image
from utils.cuda import to_cuda, permute_direction


class Tester(MetaEngine):
    """Inference Engine for a single process"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 1
        self.mode = 'test'

        self.note = dict()  # for Recording
        os.makedirs(self.saveroot, exist_ok=True)
        if self.opt.smart:
            self.gen_smart_build(self.opt.load_dir)

    def init(self, rank):
        """Initiate Engine"""
        # Set device for this process
        self.device_assignment(rank)

        # Option Record
        self.note = self.record(self.note)

        # Build Model
        self.generator = self.build_generator(rank, self.note)

        # Load Model
        self.model_load(rank, self.generator, None, self.opt.load_dir, False)

    @torch.no_grad()
    def main(self):
        """Launch a process"""
        self.test(0)

    def test(self, rank=0):
        """
        A definition of the engine's main execution.
        :param rank: rank of process (for DDP)
        :return: Nothing
        """
        if self.gpu: torch.cuda.set_device(rank)
        self.test_start_format = nowtime()
        self.init(rank)

        # load Dataset
        test_dataloader = self.load_data(rank)

        # Prime Process (rank 0)
        prime = True if rank == 0 else False
        if prime:
            print(f"Test Start. [{colored('Panoramic Image Generation', 'blue')}]")
            snapshot_maker(self.note, self.history_save_dir('snapshot'))

        test_start_time = time.perf_counter()
        self.opt.sample_num = self.opt.batch_size
        for i, data in enumerate(tqdm(test_dataloader) if prime else test_dataloader):
            # data move into the device
            data = to_cuda(data, self.device)
            data = permute_direction(data)

            # Forward
            panorama = self.generator(data, infer=True)  # [B x 3 x H x W]
            dst_dir = [os.path.join(self.saveroot, 'test', name) for name in data['name']]
            save_image(self.opt, panorama, dst_dir)

        if prime:
            print(f"{time.perf_counter() - test_start_time:.3f} seconds elapsed.")
            print("Test Finished.")
            time_spent = strfdelta(datetime.timedelta(seconds=time.perf_counter() - test_start_time),
                                   fmt='{d}d {h}h {m}m {s}s')
            self.note['Time Spent'] = time_spent
            print(f"{colored(time_spent, 'cyan')} spent for the training.")
            snapshot_maker(self.note, self.history_save_dir('snapshot'))

    def load_data(self, rank):
        """Load Test Dataset"""
        dataset = KandaoTestDataset(self.opt, 'test')
        pin = True if self.opt.pin_memory else False
        # sampler will be used in distributed running, else be None
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank,
                                     shuffle=False) if self.num_gpu > 1 else None
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=False, pin_memory=pin,
                                num_workers=self.opt.num_workers, sampler=sampler, drop_last=False)
        if rank == 0:
            print(f"{len(dataset)} files for test detected.")
            self.note['Test Dataset Length'] = len(dataset)

        return dataloader

    @model_construction
    def build_generator(self, *args, **kwargs):
        """Build Stitching Network"""
        return super().build_generator(*args, **kwargs)

    def record(self, note: dict):
        """Record Option"""
        if self.gpu:  # Using GPU
            for i, num in enumerate(self.opt.gpu):
                note[f'GPU Index: {num} | Device'] = torch.cuda.get_device_name(i)
        else:
            note['Device'] = 'CPU'

        if self.opt.gpu != ['-1']:  # Using GPU
            for i, num in enumerate(self.opt.gpu):
                note[f'GPU Index: {num} | Device'] = torch.cuda.get_device_name(i)
        else:
            note['Device'] = 'CPU'
            print('GPU Unabled.')
        note['Test Batch'] = self.opt.batch_size
        note['Num_workers'] = self.opt.num_workers
        note['Load'] = self.opt.load_dir
        return note

    def history_save_dir(self, keyword: str) -> str:
        """Control History Save Directory"""
        history_dir = os.path.join(self.saveroot, 'test')
        os.makedirs(history_dir, exist_ok=True)
        return {'root': history_dir,
                'snapshot': os.path.join(self.saveroot, 'test', f'G-model-snapshot.txt'),
                }[keyword]



