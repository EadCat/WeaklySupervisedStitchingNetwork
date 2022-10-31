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
from termcolor import colored

# PyTorch
import torch
import torch.distributed as dist
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

# Models
from networks.models.HomoDispNet import HomoDispNet
from networks.models.PreColorCorNet import PreColorCorNet
from networks.models.PostColorCorNet import PostColorCorNet
from networks.models.DoubleColorCorNet import DoubleColorCorNet

from utils.cuda import optimizer_to


def model_construction(build_function):
    """
    Need to inherit networks.base.BaseNet
    A record function & Sync Batch Norm Layers
    """
    def build_wrapper(engine, rank: int, note: dict, *args, **kwargs):
        mode = engine.mode
        verbose = True if rank == 0 else False
        model = build_function(engine, rank, verbose, *args, **kwargs)
        if model:
            class_name = model.get_name()
            engine.name = class_name
            note['model name'] = class_name
            note['params'] = model.get_params()
            note['tparams'] = model.get_tparams()
            if mode == 'train':
                if engine.num_gpu > 1:  # Must be disabled during Test
                    if verbose: print(f"{model.get_name()} SyncBN converting...")
                    model = SyncBatchNorm.convert_sync_batchnorm(model)

            if verbose:
                print(f"{class_name} model construction complete.")
                print(f"total parameters: {model.get_params()}")
                print(f"trainable parameters: {model.get_tparams()}")
        return model

    return build_wrapper


def load_previous_sets(smart_build):
    """
    A Decorator for Smart Build Option
    Automatically load the settings in the checkpoint(.pth) file.
    """
    def sbuild_wrapper(engine, load_dir):
        if load_dir is None or engine.opt.smart == False: return
        load_info = torch.load(load_dir, map_location='cpu')['option']
        smart_build(engine, load_info)
        del load_info
    return sbuild_wrapper


class MetaEngine:
    """Abstract Engine for Training & Testing"""
    def __init__(self, opt):
        self.opt = opt
        self.saveroot = os.path.join(self.opt.saveroot, self.opt.name)
        self.mode = 'test'

        # Device Settings
        self.num_gpu = 0
        self.device = torch.device('cpu')
        self.set_gpu()
        if self.opt.world_size is not None:
            self.world_size = self.opt.world_size if self.num_gpu > 1 else 1
        else:
            self.world_size = self.num_gpu

    def build_generator(self, rank, verbose):
        """
        :param rank: Need for DDP Wrapping
        :param verbose: print message or not
        :return: Network
        """
        choice = self.opt.generator
        if verbose:
            print(f"{colored(choice, 'yellow')} Generator building...")

        if choice.lower() in ['disp']:
            model = HomoDispNet(self.opt, self.device)
        elif choice.lower() in ['pre']:
            model = PreColorCorNet(self.opt, self.device)
        elif choice.lower() in ['post']:
            model = PostColorCorNet(self.opt, self.device)
        elif choice.lower() in ['double']:
            model = DoubleColorCorNet(self.opt, self.device)
        else:
            raise ValueError(f"{choice} is not supported.")
        return model

    def model_load(self, rank, model, optimizer=None, load_dir=None, timeset=False):
        """
        :param rank: Device Rank in DDP System
        :param model: Network
        :param optimizer: Optimizer linked with the model
        :param load_dir: Load Directory
        :param timeset: Set Epoch or Not
        """
        if load_dir is None: return
        if self.gpu: device = torch.device(rank)
        else: device = torch.device('cpu')
        load_dict = torch.load(load_dir, map_location=device)
        epoch = load_dict.get('epoch')
        iteration = load_dict.get('iter')

        if isinstance(model, DDP) or isinstance(model, DP):
            model.module.load_state_dict(load_dict['ckpt'], strict=self.opt.strict)
            model_name = model.module.get_name()
        else:
            model.load_state_dict(load_dict['ckpt'], strict=self.opt.strict)
            model_name = model.get_name()

        if rank == 0:
            print("===================================================")
            print(f"{model_name} model has been loaded from {load_dir}.")
            print(f"Strict Option: {self.opt.strict}")
            print(f"Model's Time Record - Epoch: {epoch} | iter: {iteration}")
            print("===================================================")

        if optimizer is not None:
            optimizer.load_state_dict(load_dict['optim'])
            if self.gpu: optimizer_to(optimizer, device)
            if rank == 0:
                print("===================================================")
                print(f"Optimizer has been loaded from {load_dir}.")
                print("===================================================")

        if timeset and epoch is not None:
            if rank == 0: print(f"Training Epoch Setting: {epoch + 1}")
            self.epoch = epoch + 1

    @load_previous_sets
    def gen_smart_build(self, load_info):
        """
        Generator Smart Build
        :param load_info: Option information in the checkpoint file
        """
        # Kind
        kind = ['generator', 'unet', 'reg']

        # General Option for Stitcher
        stitcher = ['homography', 'local_adj_limit']

        # Append Here if additional option need.
        setting_collections = [kind, stitcher]

        for setting in setting_collections:
            for sets in setting:
                setattr(self.opt, sets, getattr(load_info, sets))

    def device_assignment(self, rank):
        """
        Device Set for a process
        :param rank: Device Rank in DDP System
        """
        if self.gpu: self.device = torch.device(rank)
        else:        self.device = torch.device('cpu')

    def set_gpu(self):
        """
        Set GPU Environment
        Set Environment Variable: CUDA_VISIBLE_DEVICES
        """
        if self.opt.gpu != ['-1']:
            print(f"GPU CALL: {self.opt.gpu}")
            self.num_gpu = len(self.opt.gpu) if self.opt.gpu != ['-1'] else 0
            environ_value = ','.join(self.opt.gpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = environ_value  # Do not use cuda lib before this declaration
            self.gpu = True if self.num_gpu > 0 and torch.cuda.is_available() else False
            for i, num in enumerate(self.opt.gpu):
                print(f'GPU Index: {num} | Name: {colored(torch.cuda.get_device_name(i), "green")}')
        else:
            print("CPU set.")
            self.gpu = False
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

    def distribution_setup(self, gpu_idx):
        """
        Set DDP Environment
        initiate multi-process group
        """
        global_rank = self.opt.rank * self.opt.npgpu + gpu_idx  # node rank * node GPUs + index of this GPU
        if self.opt.world_size is None: self.opt.world_size = len(self.opt.gpu) * self.opt.node
        self.world_size = self.opt.world_size  # number of all GPUs
        os.environ['MASTER_ADDR'] = self.opt.master_addr
        os.environ['MASTER_PORT'] = self.opt.master_port
        dist.init_process_group(backend='nccl',
                                world_size=self.world_size,
                                rank=global_rank)
        return global_rank


































