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
from .photometric import L1Loss, L2Loss
from .ssim import ssim as SSIM
from .vgg.vgg16_inter import VGG16
from normalization.ground import ground_zero
from normalization.manipulator import *


def activation_check(forward):
    """A Decorator for Activation Check by Option"""
    def function_wrapper(obj, *args, **kwargs):
        if obj.activate:
            return forward(obj, *args, **kwargs)
        else:
            return torch.tensor(0., requires_grad=True).to(obj.device)

    return function_wrapper


class MetaLoss(nn.Module):
    """Abstract Class for Loss"""
    def __init__(self, opt, device='cpu'):
        super().__init__()
        self.opt = opt
        self.groundzero = ground_zero(opt.transform, opt.mean, opt.std)
        self.loss_container = opt.loss
        self.device = device
        self.activate = False

    def forward(self, *args, **kwargs):
        """Abstract Method"""
        raise NotImplementedError


class AppearanceLoss(MetaLoss):
    """Calculate L1 or L2 Loss"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activate = True if 'L1' in self.loss_container or 'L2' in self.loss_container else False

    @activation_check
    def forward(self, gt: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Calculate L1 or L2 Loss"""
        loss = torch.tensor(0., requires_grad=True).to(self.device)

        if 'L1' in self.loss_container:
            l1 = L1Loss(gt, output)
            loss += l1

        if 'L2' in self.loss_container:
            l2 = L2Loss(gt, output)
            loss += l2

        return loss


class SSIMLoss(MetaLoss):
    """
    Calculate PyTorch SSIM Loss
    PyTorch SSIM from Po-Hsun Su
    https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.denormalizer = ImageDenormalize(self.opt, self.opt.mean, self.opt.std)
        self.activate = True if 'SSIM' in self.loss_container else False

    @activation_check
    def forward(self, gt: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        gt: [# x Batch x Channel x Height x Width]
        output: [# x Batch x Channel x Height x Width]
        """
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        # get minibatch
        for g, out in zip(gt, output):
            # g = self.denormalizer(g)  # rescaling [-1 ~ 1] -> [0 ~ 1]
            # out = self.denormalizer(out)  # rescaling [-1 ~ 1] -> [0 ~ 1]
            g = denormalizer(g, 1.)  # rescaling [-1 ~ 1] -> [0 ~ 1]
            out = denormalizer(out, 1.)  # rescaling [-1 ~ 1] -> [0 ~ 1]
            ssim = SSIM(g, out, is_train=True)
            loss += ssim

        return loss


class PerceptualLoss(MetaLoss):
    """Calculate Perceptual Loss"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = VGG16().to(self.device)
        self.loss_weight = self.opt.vgg_loss_weight
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.denormalizer = ImageDenormalize(self.opt, self.opt.mean, self.opt.std)
        self.renormalizer = ImageNormalize(self.opt, self.mean, self.std, conditional=False)
        self.eval()
        self.activate = True if 'PL1' in self.loss_container or 'PL2' in self.loss_container else False

    @activation_check
    def forward(self, gt: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        gt: [# x Batch x Channel x Height x Width]
        output: [# x Batch x Channel x Height x Width]
        """
        loss = torch.tensor(0., requires_grad=True).to(self.device)

        for g, out in zip(gt, output):
            # g = self.denormalizer(g)
            g = denormalizer(g, 1.)
            g = self.renormalizer(g)
            # out = self.denormalizer(out)
            out = denormalizer(out, 1.)
            out = self.renormalizer(out)

            gts = self.model(g)  # tuple
            outs = self.model(out)  # tuple

            if 'PL1' in self.loss_container:
                for weight, gt_feat, out_feat in zip(self.loss_weight, gts, outs):
                    loss += weight * L1Loss(gt_feat, out_feat)

            if 'PL2' in self.loss_container:
                for weight, gt_feat, out_feat in zip(self.loss_weight, gts, outs):
                    loss += weight * L2Loss(gt_feat, out_feat)

        return loss

