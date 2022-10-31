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
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

# Custom
from .record import csv_record
from .plot import PlotGenerator


def model_check(any_funct):
    """Check if model is not None"""
    def pass_check(self, *args, **kwargs):
        if self.model is None:
            pass
        else:
            return any_funct(self, *args, **kwargs)
    return pass_check


class Librarian(nn.Module):
    """
    * ====== PyTorch Experiment Support ====== *
    * ============= Record Agent ============= *
    Dae-Young Song from Chungnam National University
    2022. 04. 14.
    """
    def __init__(self, opt, rank, loss_name, model, optimizer=None):
        super().__init__()
        self.opt = opt
        self.prime = True if rank == 0 else False  # main proces
        self.device = torch.device(rank)

        self.to(self.device)
        if isinstance(model, DistributedDataParallel) or isinstance(model, DataParallel):
            self.model = model.module
        else: self.model = model
        self.name = self.model.get_name()
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.saveroot = os.path.join(self.opt.saveroot, self.opt.name)

        self.iter_fq = opt.iter_op_freq
        self.ep_fq = opt.epoch_op_freq
        self.iter_save_fq = opt.iter_save_freq
        self.ep_save_fq = opt.epoch_save_freq

        self.num_gpu = 0
        self.gpu = False
        self.get_device_settings()

        self.iter_loss = 0.
        self.epoch_loss = 0.

        self.decimal = opt.loss_decimal  # int

    @model_check
    def __call__(self, *args, **kwargs):
        self.insert(*args, **kwargs)

    @model_check
    def loss_init(self):
        """Initialize loss to zero."""
        self.iter_loss = 0.

    @model_check
    def insert(self, value):
        """
        self.loss_value += value
        :param value: tensor or some matrices
        :return: None
        """
        if self.num_gpu > 1 and isinstance(value, torch.Tensor):
            value = self.all_reduce(value)

        if isinstance(value, torch.Tensor): value = value.item()

        self.iter_loss += value
        self.epoch_loss += value

    def iter_doing(self, i, epoch):
        """Action Definition during iteration"""
        self.iter_calc_rec(i, epoch)
        self.iter_save(i, epoch)

    def epoch_doing(self, epoch, len_loader):
        """Action Definition during an epoch"""
        self.ep_calc_rec(epoch, len_loader)
        self.epoch_save(epoch, len_loader)

    @model_check
    def iter_calc_rec(self, i, epoch):
        """
        iteration loss calculation and record
        :param i: [0 ~ (data number - 1)]
        :param epoch: [1 ~ total_epoch]
        :return: None
        """
        if self.iter_fq > 0 and (i + 1) % self.iter_fq == 0:
            self.iter_loss /= self.iter_fq

            if self.prime:
                if self.opt.iter_print:
                    print(f"[Epoch: {epoch} | Iter: {i+1}]: [{self.name} | {self.loss_name}]: "
                          f"{self.iter_loss:.{self.decimal}}")

                dataframe = {'Epoch': epoch, 'Iteration': i+1, self.name: round(self.iter_loss, self.decimal)}
                csv_record(dataframe, self.history_save_dir()['loss-history'])

            self.iter_loss = 0.

    @model_check
    def ep_calc_rec(self, epoch, len_loader):
        """
        epoch loss calculation and record
        :param epoch: [1 ~ total_epoch]
        :param len_loader: maximum length of dataloader
        :return: None
        """
        if self.ep_fq > 0 and epoch % self.ep_fq == 0:
            self.epoch_loss /= (self.ep_fq * len_loader)

            if self.prime:
                if self.opt.epoch_print:
                    print(f"[Epoch: {epoch}]: [{self.name} | {self.loss_name}]: {self.epoch_loss:.{self.decimal}}")
                dataframe = {'Epoch': epoch, 'Iteration': len_loader, self.name: round(self.epoch_loss, self.decimal)}
                csv_record(dataframe, self.history_save_dir()['loss-history'])

            self.epoch_loss = 0.

    def save_model_info(self, epoch: int, i: int):
        """
        refine model value to save checkpoint
        :param i: [0 ~ (data number - 1)]
        :param epoch: [1 ~ total_epoch]
        :return: None
        """
        weights = self.model.state_dict()
        if self.optimizer: opckpt = self.optimizer.state_dict()
        else:              opckpt = None
        return {'epoch': epoch,
                'iter': i + 1,
                'ckpt': weights,
                'optim': opckpt,
                'option': self.opt}

    @model_check
    def iter_save(self, i, epoch):
        """
        model save during iteration
        :param i: [0 ~ (data number - 1)]
        :param epoch: [1 ~ total_epoch]
        :return: None
        """
        if self.iter_save_fq > 0 and (i + 1) % self.iter_save_fq == 0 and self.prime:
            torch.save(self.save_model_info(epoch, i), self.weight_save_dir(epoch, i))

    @model_check
    def epoch_save(self, epoch, len_loader):
        """
        model save during epoch
        :param epoch: [1 ~ total_epoch]
        :param len_loader: maximum length of dataloader
        :return: None
        """
        if self.ep_save_fq > 0 and epoch % self.ep_save_fq == 0 and self.prime:
            torch.save(self.save_model_info(epoch, len_loader), self.weight_save_dir(epoch))

    @model_check
    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        """
        all reduce to get mean value
        :param value: Tensor value (call by reference)
        :return: reduced tensor
        """
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= self.world_size
        return value

    @model_check
    def weight_save_dir(self, epoch, i=None) -> str:
        """Get Checkpoint save directory"""
        weight_root = os.path.join(self.saveroot, 'weights')
        os.makedirs(weight_root, exist_ok=True)
        if i is None:
            return os.path.join(weight_root, f'{self.name}-{str(epoch).zfill(3)}.pth')
        else:
            return os.path.join(weight_root, f'{self.name}-{str(epoch).zfill(3)}-{str(i+1).zfill(4)}.pth')

    @model_check
    def history_save_dir(self) -> dict:
        """Get History save directory"""
        history_dir = os.path.join(self.saveroot, 'history')
        total_epochs = str(self.opt.total_epochs).zfill(4)
        os.makedirs(history_dir, exist_ok=True)
        return {'root': history_dir,
                'loss-history': os.path.join(history_dir, f'{self.name}-Loss-History.csv'),
                'loss-graph': os.path.join(history_dir, f'{self.name}-{self.loss_name}-graph-{total_epochs}.png'),
                'snapshot': os.path.join(self.saveroot, 'weights', f'model-snapshot-{self.name}-{total_epochs}.txt'),
                }

    @model_check
    def get_device_settings(self):
        """Get number of GPU and device for all-reduce"""
        if self.opt.gpu != ['-1']:
            self.num_gpu = len(self.opt.gpu) if self.opt.gpu != ['-1'] else 0
            self.gpu = True if self.num_gpu > 0 and torch.cuda.is_available() else False
        else: pass

        if self.opt.world_size is not None:
            self.world_size = self.opt.world_size if self.num_gpu > 1 else 1
        else:
            self.world_size = self.num_gpu

    @model_check
    def plot(self, number, color: str):
        """Plot loss graph"""
        dataframe = pd.read_csv(self.history_save_dir()['loss-history'])
        epochs = dataframe['Epoch']
        iterations = dataframe['Iteration']
        temporal_axis = [epoch * iteration for iteration in iterations for epoch in epochs]

        plot = PlotGenerator(number, self.name, (20, 15), xlabel='epoch x iter', ylabel=self.loss_name)
        losses = dataframe[self.name]
        # refinement
        plot_data = {temp: loss for temp, loss in zip(temporal_axis, losses)}
        plot.add_data(plot_data)
        plot.add_set(name=self.loss_name, color=color)
        plot.plot()
        plot.save(self.history_save_dir()['loss-graph'])
