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
import torch.nn as nn
from utils.masking import region_divider, zero_masking
from normalization.ground import ground_zero, ground_max
from .modules import AppearanceLoss, SSIMLoss, PerceptualLoss


class RegionDivideLoss(nn.Module):
    """
    L1 L2
    SSIM
    VGG-16 Loss
    """
    def __init__(self, opt, device='cpu'):
        super().__init__()
        self.opt = opt
        self.device = device
        self.groundzero = ground_zero(opt.transform, opt.mean, opt.std)
        self.groundmax = ground_max(opt.transform, opt.mean, opt.std)

        self.region_div = [opt.div_appear, opt.div_ssim, opt.div_perceptual]
        self.loss_weights = [opt.appearance_weight, opt.ssim_weight, opt.perceptual_weight]

        self.appearance_loss = AppearanceLoss(opt, device)
        self.ssim_loss = SSIMLoss(opt, device)
        self.perceptual_loss = PerceptualLoss(opt, device)
        self.functions = [self.appearance_loss, self.ssim_loss, self.perceptual_loss]

    def forward(self, output: dict, data: dict) -> torch.Tensor:
        """ ===========================================
        gts: [# x Batch x Channel x Height x Width]
        gt_mask: [# x Batch x Channel x Height x Width]
        pred: [Batch x Channel x Height x Width]
        =========================================== """
        gts = data['gts']
        gt_mask = data['gt_masks']
        pred = output['panorama']

        loss = torch.tensor(0, dtype=torch.float32).to(self.device)

        # Devide region of masks
        overlap_masks, nonoverlap_masks = region_divider(gt_mask)

        # Devide region
        # [B x C x H x W] -> [# x Batch x Channel x Height x Width]
        overlap_pred = torch.stack([zero_masking(pred, mask, self.groundzero) for mask in gt_mask])
        seamline_pred = torch.stack([zero_masking(pred, mask, self.groundzero) for mask in nonoverlap_masks])
        seamline_gt = torch.stack([zero_masking(gt, mask, self.groundzero) for mask, gt in zip(nonoverlap_masks, gts)])
        # Complete Non-overlap GT & Inference
        for i, (maxi, zero) in enumerate(zip(self.groundmax, self.groundzero)):
            seamline_pred[:, :, i, ...][torch.roll(overlap_masks, 1, dims=0)[:, :, 0, ...] == maxi] = zero
            seamline_gt[:, :, i, ...][torch.roll(overlap_masks, 1, dims=0)[:, :, 0, ...] == maxi] = zero

        # Loss Calculation
        for div, weight, loss_func in zip(self.region_div, self.loss_weights, self.functions):
            if div:  # Non-overlap Region Loss
                loss += loss_func(seamline_gt, seamline_pred) * weight
            else:
                loss += loss_func(gts, overlap_pred) * weight

        return loss
