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
import argparse


def str2bool(v):
    """Enable to set boolean value in the shell"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    """Copyright 2022. ETRI all rights reserved."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        """Initialize Options"""
        # Directory Control
        self.parser.add_argument('--name', type=str, default='ckpt', help='directory name for record')
        self.parser.add_argument('--saveroot', type=str, default='./archive', help='root directory of "name" option')

        # Training Options
        # 1. Frequency
        self.parser.add_argument('--total-epochs', type=int, dest='total_epochs', default=10,
                                 help='# of epochs for training')
        self.parser.add_argument('--iter-op-freq', type=int, dest='iter_op_freq', default=100,
                                 help='Loss calc. period during iteration, disable: -1')
        self.parser.add_argument('--epoch-op-freq', type=int, dest='epoch_op_freq', default=1,
                                 help='Loss calc. period during epoch, disable: -1')
        self.parser.add_argument('--iter-save-freq', type=int, dest='iter_save_freq', default=-1,
                                 help='Model Saving period during iteration, disable: -1')
        self.parser.add_argument('--epoch-save-freq', type=int, dest='epoch_save_freq', default=1,
                                 help='Model Saving period during epoch, disable: -1')
        self.parser.add_argument('--iter-sample-freq', type=int, dest='iter_sample_freq', default=-1,
                                 help='Output Sampling Frequency during iteration, disable: -1')
        self.parser.add_argument('--epoch-sample-freq', type=int, dest='epoch_sample_freq', default=1,
                                 help='Output Sampling Frequency during epoch, disable: -1')
        self.parser.add_argument('--sample-num', dest='sample_num', type=int, default=1,
                                 help='# of samples during training')

        # 2. Loss
        self.parser.add_argument('--loss', type=str, nargs='+',
                                 choices=['L1', 'L2', 'SSIM', 'PL1', 'PL2'],
                                 help='Loss Selection')
        self.parser.add_argument('--appearance-weight', type=float, dest='appearance_weight', default=0.0,
                                 help='Weight of appearance loss')
        self.parser.add_argument('--div-appear', type=str2bool, dest='div_appear', default=False,
                                 help="Divide application area of appearance loss")
        self.parser.add_argument('--ssim-weight', type=float, dest='ssim_weight', default=0.4,
                                 help='Weight of SSIM loss')
        self.parser.add_argument('--div-ssim', type=str2bool, dest='div_ssim', default=True,
                                 help='Divide application area of SSIM loss')
        self.parser.add_argument('--perceptual-weight', type=float, dest='perceptual_weight', default=0.6,
                                 help='Weight of perceptual loss')
        self.parser.add_argument('--div-perceptual', type=str2bool, dest='div_perceptual', default=False,
                                 help='Divide application area of perceptual loss')
        self.parser.add_argument('--vgg-loss-weight', type=float, nargs=5, dest='vgg_loss_weight',
                                 default=[0., 0., 0.2, 0.3, 0.5], help='Perceptual Loss Weight')

        # 3. Optimizer
        self.parser.add_argument('--optim', type=str, default='RMSProp', help='Reconstructor Optimizer Selection')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='Reconstructor learning rate')
        self.parser.add_argument('--beta', type=float, default=(0.9, 0.999), help='momentum for Adam')
        self.parser.add_argument('--momentum', type=float, default=0, help='PyTorch SGD, RMSprop parameter momentum')
        self.parser.add_argument('--dampening', type=float, default=0, help='PyTorch SGD, RMSprop parameter dampening')
        self.parser.add_argument('--nesterov', type=str2bool, default=False,
                                 help='for SGD, http://www.cs.toronto.edu/~hinton/absps/momentum.pdf')
        self.parser.add_argument('--weight-decay', type=float, default=0, dest='weight_decay',
                                 help='PyTorch SGD, RMSprop, Adagrad, Adam parameter, L2 penalty weight decay')
        self.parser.add_argument('--eps', type=float, default=1e-08, help='Pytorch RMSprop, Adam parameter')
        self.parser.add_argument('--alpha', type=float, default=0.99, help='Pytorch RMSprop parameter alpha')
        self.parser.add_argument('--amsgrad', type=str2bool, default=False,
                                 help='for Adam, https://openreview.net/forum?id=ryQu7f-RZ')
        self.parser.add_argument('--centered', type=str2bool, default=False,
                                 help='for RMSprop, https://arxiv.org/pdf/1308.0850v5.pdf')
        self.parser.add_argument('--lr-decay', dest='lr_decay', type=float, default=0,
                                 help='for Adagrad, Learning rate decay')
        self.parser.add_argument('--init-accumulator', dest='init_accumulator', type=float, default=0,
                                 help='term for PyTorch Adagrad')

        # Model
        self.parser.add_argument('--unet', type=str, default='large', help='UNet Model Selection')
        self.parser.add_argument('--reg', type=str, default='large', help='Regressor Model Selection')
        self.parser.add_argument('--homography', type=int, default=1, help='# of homography for the Regressor')
        self.parser.add_argument('--generator', type=str, choices=['disp', 'pre', 'post', 'double'],
                                 help='Model Selection')
        self.parser.add_argument('--local-adj-limit', dest='local_adj_limit', type=float, default=0.3,
                                 help="Maximum Limitation of Local Adjustment Layer's Output")
        self.parser.add_argument('--load-dir', dest='load_dir', type=str, default=None,
                                 help='Load model from this directory')
        self.parser.add_argument('--strict', type=str2bool, default=True, help='Strict Checkpoint Loading')
        self.parser.add_argument('--smart', type=str2bool, default=True, help='Activate Smart Build')
        self.parser.add_argument('--init-model', type=str, dest='init_model', default='xavier',
                                 choices=['normal', 'xavier', 'kaiming', 'orthogonal'],
                                 help='Initiation Selection of Model Weight')
        self.parser.add_argument('--init-gain', type=float, dest='init_gain', default=0.2,
                                 help='Initiation Gain of Model Weight')

        # Device
        self.parser.add_argument('--gpu', type=str, nargs='+', default=['-1'],
                                 help='GPU IDs, e.g.: 0 1 2, CPU: -1')
        self.parser.add_argument('--master-addr', dest='master_addr', type=str, default='127.0.0.1',
                                 help='IP Address for Dist-engine')
        self.parser.add_argument('--master-port', dest='master_port', type=str, default='12355',
                                 help='Port Num. for Dist-engine')
        self.parser.add_argument('--node', type=int, default=1, help='# of total nodes(CPUs) to use (Distribution)')
        self.parser.add_argument('--npgpu', type=int, default=4, help='# of GPUs per this Node (Distribution)')
        self.parser.add_argument('--rank', type=int, default=0, help='ranking within the nodes (Distribution)')
        self.parser.add_argument('--world-size', dest='world_size', type=int, default=None, help='# of all GPUs')

        # Dataloader
        self.parser.add_argument('--dataroot', type=str, help='Dataset Root Directory')
        self.parser.add_argument('--train-datalist', dest='train_datalist', type=str,
                                 help='Dataset txt file (split) for training')
        self.parser.add_argument('--valid-datalist', dest='valid_datalist', type=str,
                                 help='Dataset txt file (split) for validation')
        self.parser.add_argument('--test-datalist', dest='test_datalist', type=str,
                                 help='Dataset txt file (split) for testing')
        self.parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='Batch Size')
        self.parser.add_argument('--num-workers', dest='num_workers', type=int, default=2, help='Number of workers')
        self.parser.add_argument('--shuffle', type=str2bool, default=True, help='Activate shuffle or not')
        self.parser.add_argument('--pin-memory', dest='pin_memory', type=str2bool, default=True,
                                 help='Activate pin memory or not')

        # Data Transform
        self.parser.add_argument('--transform', nargs='+', help='Tensor Transform Selection',
                                 type=str, choices=['resize', 'normalize', 'augment'])
        self.parser.add_argument('--resize', type=int, nargs=2, default=(512, 1024), help='Resize [Height x Width]')
        self.parser.add_argument('--mean', type=float, nargs=3, default=None, help='Normalize Mean')
        self.parser.add_argument('--std', type=float, nargs=3, default=None, help='Normalize Standard Deviation')
        self.parser.add_argument('--aug-prob', dest='aug_prob', type=float, default=0.5,
                                 help='Augmentation Probability')

        # Permission
        self.parser.add_argument('--iter-print', dest='iter_print', type=str2bool, default=True,
                                 help='Print iteration progress or not.')
        self.parser.add_argument('--epoch-print', dest='epoch_print', type=str2bool, default=True,
                                 help='Print epoch progress or not')
        self.parser.add_argument('--loss-decimal', dest='loss_decimal', type=int, default=5,
                                 help='Decide where to round up the LOSS')

    def parse(self):
        """Parse Arguments"""
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt



