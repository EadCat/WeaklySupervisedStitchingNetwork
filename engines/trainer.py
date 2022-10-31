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
import time
import datetime
from termcolor import colored

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from engines.metaclass import MetaEngine, model_construction
from loss.control import RegionDivideLoss
from dataset.kandao_train import KandaoTrainDataset
from librarians.librarian import Librarian
from librarians.sampler import ValidSampler
from librarians.record import snapshot_maker
from utils.time import *
from utils.image import save_image
from utils.cuda import to_cuda, permute_direction


class Trainer(MetaEngine):
    """Training Engine for a single process"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 1
        self.mode = 'train'

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
        self.optimizer = self.optimizer_selection(self.generator, rank, self.opt.optim, self.opt.lr, 'Generator')

        # Load Model
        self.model_load(rank, self.generator, self.optimizer, self.opt.load_dir, True)

        # Loss Functions
        self.region_loss = RegionDivideLoss(self.opt, self.device)

        # Librarians for record
        self.lib = Librarian(self.opt, rank, loss_name=self.region_loss.__class__.__name__,
                             model=self.generator, optimizer=self.optimizer)
        self.sampler = ValidSampler(self.opt, rank, self.generator, save_image, fmt='png')

    def main(self):  # Single process unit
        """Launch a process"""
        self.train(0)

    def train(self, rank=0):
        """
        A definition of the engine's main execution.
        :param rank: rank of process (for DDP)
        :return: Nothing
        """
        # Time Setting
        train_start_time = time.perf_counter()
        if self.gpu: torch.cuda.set_device(rank)
        self.train_start_format = nowtime()
        self.init(rank)

        # load Dataset
        train_dataloader = self.load_data(rank)

        # Prime Process (rank 0)
        prime = True if rank == 0 else False
        if prime:
            print(f"Training Start. [{colored('Panoramic Image Generation', 'blue')}]")
            snapshot_maker(self.note, self.history_save_dir('snapshot'))

        # Training loop
        for epoch in range(self.epoch, self.opt.total_epochs + 1):
            epoch_start = time.perf_counter()
            if prime: print("================================ Epoch Start ==================================")
            if self.num_gpu > 1 and self.opt.shuffle: train_dataloader.sampler.set_epoch(epoch - 1)

            self.lib.loss_init()
            self.generator.train()
            for i, data in enumerate(train_dataloader):
                # data move into the device
                data = to_cuda(data, self.device)
                data = permute_direction(data)

                # Feed-Forward
                output = self.generator(data, infer=False)
                loss = self.region_loss(output, data)

                # Back-Propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record & Save Checkpoint during Iteration (Including All-Reduce)
                self.lib(loss)
                self.sampler.iter_sample(i, epoch, data)
                self.lib.iter_doing(i, epoch)

            # Record & Save Checkpoint during Epoch (Including All-Reduce)
            self.sampler.epoch_sample(epoch, data)
            self.lib.epoch_doing(epoch, len(train_dataloader))

            if prime:
                print(f"{time.perf_counter() - epoch_start:.3f} seconds elapsed for {epoch}th epoch.")

        # Training End
        if prime:
            print("Training Complete.")
            time_spent = strfdelta(datetime.timedelta(seconds=time.perf_counter() - train_start_time),
                                   fmt='{d}d {h}h {m}m {s}s')
            self.note['Time Spent'] = time_spent
            print(f"{colored(time_spent, 'cyan')} spent for the training.")
            snapshot_maker(self.note, log_dir=self.history_save_dir('snapshot'))

        self.lib.plot(0, 'r')

    def load_data(self, rank):
        """Load Training Dataset"""
        dataset = KandaoTrainDataset(self.opt, 'train')
        pin = True if self.opt.pin_memory and self.gpu else False
        seed = torch.randint(16384, size=(1, )).item()
        # sampler will be used in distributed training, else be None
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank,
                                     shuffle=self.opt.shuffle, seed=seed) if self.num_gpu > 1 else None
        dataloader = DataLoader(dataset, batch_size=self.opt.batch_size,
                                shuffle=self.opt.shuffle & (sampler is None), pin_memory=pin,
                                num_workers=self.opt.num_workers, sampler=sampler, drop_last=True)
        if rank == 0:
            print(f"{len(dataset)} files for training detected.")
            self.note['Train Dataset Length'] = len(dataset)

        return dataloader

    def optimizer_selection(self, model, rank, optim: str, lr: float, decorator: str):
        """Build Optimizer by Option"""
        verbose = True if rank == 0 else False
        lr = lr
        beta = self.opt.beta
        eps = self.opt.eps
        weight_decay = self.opt.weight_decay
        lr_decay = self.opt.lr_decay

        amsgrad = self.opt.amsgrad
        init_accu = self.opt.init_accumulator
        momentum = self.opt.momentum
        dampening = self.opt.dampening
        nesterov = self.opt.nesterov
        alpha = self.opt.alpha
        centered = self.opt.centered

        if verbose:
            print(f"{decorator} optimizer: {optim}")

        if optim.lower() == 'Adam'.lower():
            return torch.optim.Adam(model.parameters(), lr=lr, betas=beta, eps=eps, weight_decay=weight_decay,
                                    amsgrad=amsgrad)
        elif optim.lower() == 'Adagrad'.lower():
            return torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                                       initial_accumulator_value=init_accu, eps=eps)
        elif optim.lower() == 'SGD'.lower():
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                   weight_decay=weight_decay, nesterov=nesterov)

        elif optim.lower() == 'RMSprop'.lower():
            return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps,
                                       weight_decay=weight_decay, momentum=momentum, centered=centered)
        else:
            return None

    def history_save_dir(self, keyword: str) -> str:
        """Control History Save Directory"""
        history_dir = os.path.join(self.saveroot, 'history')
        epochs = str(self.opt.total_epochs).zfill(4)
        os.makedirs(history_dir, exist_ok=True)
        return {'root': history_dir,
                'trainloss': os.path.join(history_dir, f'G-train-loss-{epochs}.png'),
                'record_train_iter': os.path.join(history_dir, f'G-train-iter-loss-{epochs}.txt'),
                'record_train': os.path.join(history_dir, f'G-train-loss-{epochs}.txt'),
                'snapshot': os.path.join(self.saveroot, 'weights', f'G-model-snapshot-{epochs}.txt'),
                }[keyword]

    @model_construction
    def build_generator(self, *args, **kwargs):
        """Build Generator"""
        return super().build_generator(*args, **kwargs)

    def record(self, note: dict) -> dict:
        """Record Options"""
        option = vars(self.opt)

        for k, v in option.items():
            note[k] = v
        if self.gpu:  # Using GPU
            for i, num in enumerate(self.opt.gpu):
                note[f'GPU Index: {num} | Device'] = torch.cuda.get_device_name(i)
        else:
            note['Device'] = 'CPU'
        return note
