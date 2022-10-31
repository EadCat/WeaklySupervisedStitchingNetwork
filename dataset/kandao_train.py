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
import random

import torch

from .transform import *
from .augmentation import AugCompose
from utils.masking import zero_masking
from .kandao import KandaoDataset


class KandaoTrainDataset(KandaoDataset):
    """Dataset for Training"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs_1 = []
        self.inputs_2 = []
        self.inputs_3 = []
        self.gt_1 = []
        self.gt_2 = []
        self.gt_3 = []
        self.gt_1_mask = []
        self.gt_2_mask = []
        self.gt_3_mask = []

        target_file = open(self.datalist)
        lines = target_file.readlines()

        self.list_call(lines)

        self.image_transform = ImageTransform(self.opt)
        self.mask_transform = MaskTransform(self.opt)

    def list_call(self, lines):
        for line in lines:
            """
            data order
            input1 | input2 | input3 | gt1 | gt1_mask | gt2 | gt2_mask | gt3 | gt3_mask
            """
            objs = line.strip('\n').split(' ')
            self.inputs_1.append(os.path.join(self.dataroot, objs[0]))
            self.inputs_2.append(os.path.join(self.dataroot, objs[1]))
            self.inputs_3.append(os.path.join(self.dataroot, objs[2]))
            self.gt_1.append(os.path.join(self.dataroot, objs[3]))
            self.gt_2.append(os.path.join(self.dataroot, objs[5]))
            self.gt_3.append(os.path.join(self.dataroot, objs[7]))
            self.gt_1_mask.append(os.path.join(self.dataroot, objs[4]))
            self.gt_2_mask.append(os.path.join(self.dataroot, objs[6]))
            self.gt_3_mask.append(os.path.join(self.dataroot, objs[8]))

    def __len__(self):
        return len(self.inputs_1)

    def __getitem__(self, idx):
        input_name1 = self.inputs_1[idx]
        input_name2 = self.inputs_2[idx]
        input_name3 = self.inputs_3[idx]

        mask_name1 = self.gt_1_mask[idx]
        mask_name2 = self.gt_2_mask[idx]
        mask_name3 = self.gt_3_mask[idx]

        gt_name1 = self.gt_1[idx]
        gt_name2 = self.gt_2[idx]
        gt_name3 = self.gt_3[idx]

        input1 = Image.open(input_name1).convert('RGB')
        input2 = Image.open(input_name2).convert('RGB')
        input3 = Image.open(input_name3).convert('RGB')

        gt1 = Image.open(gt_name1).convert('RGB')
        gt2 = Image.open(gt_name2).convert('RGB')
        gt3 = Image.open(gt_name3).convert('RGB')

        mask1 = Image.open(mask_name1).convert('L')
        mask2 = Image.open(mask_name2).convert('L')
        mask3 = Image.open(mask_name3).convert('L')

        augmentation = AugCompose()

        # transforms
        if 'aug' in self.opt.transform and random.random() > (1. - self.opt.aug_prob) and self.mode == 'train':
            input1, input2, input3, gt1, gt2, gt3 = augmentation([input1, input2, input3, gt1, gt2, gt3])
        input1 = self.image_transform(input1)
        input2 = self.image_transform(input2)
        input3 = self.image_transform(input3)

        gt1 = self.image_transform(gt1)
        gt2 = self.image_transform(gt2)
        gt3 = self.image_transform(gt3)

        mask1 = self.mask_transform(mask1)
        mask2 = self.mask_transform(mask2)
        mask3 = self.mask_transform(mask3)

        # Remove Augmentation Effect in Empty Space
        gt1 = zero_masking(gt1, mask1, self.groundzero)
        gt2 = zero_masking(gt2, mask2, self.groundzero)
        gt3 = zero_masking(gt3, mask3, self.groundzero)

        input_name1 = os.path.splitext(os.path.basename(input_name1))[0]

        inputs = torch.stack([input1, input2, input3])
        gts = torch.stack([gt1, gt2, gt3])
        gt_masks = torch.stack([mask1, mask2, mask3])

        return {'images': inputs,
                'gts': gts,
                'gt_masks': gt_masks,
                'name': input_name1}





