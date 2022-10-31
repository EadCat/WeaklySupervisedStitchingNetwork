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
from PIL import Image
import torchvision.transforms as transforms


class BaseTransform:
    """
    A Base Class for inheritance
    """
    def __init__(self, opt):
        self.opt = opt
        self.functions = []
        self.collect()

    def collect(self):
        """
        Implemented in subclass
        """
        raise NotImplementedError

    def __call__(self, input_tensor):
        if callable(self.functions):
            return self.functions(input_tensor)
        else:
            raise ValueError('Transform functions is not callable.')


class ImageTransform(BaseTransform):
    """
    Transform Function for 2D Image [Resize, ToTensor, Normalize]
    """
    def __init__(self, opt):
        super(ImageTransform, self).__init__(opt)

    def __call__(self, input_tensor):
        if isinstance(input_tensor, Image.Image): input_tensor = input_tensor.convert('RGB')
        return super().__call__(input_tensor)

    def collect(self):
        if 'resize' in self.opt.transform:
            h, w = self.opt.resize
            self.functions.append(transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC))
        self.functions.append(transforms.ToTensor())
        if 'normalize' in self.opt.transform:
            self.functions.append(transforms.Normalize(self.opt.mean, self.opt.std, inplace=True))

        self.functions = transforms.Compose(self.functions)


class MaskTransform(BaseTransform):
    """
    Transform Function for 2D Mask [Resize, ToTensor]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_tensor):
        if isinstance(input_tensor, Image.Image): input_tensor = input_tensor.convert('L')
        return super().__call__(input_tensor)

    def collect(self):
        if 'resize' in self.opt.transform:
            h, w = self.opt.resize
            self.functions.append(transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC))

        self.functions.append(transforms.ToTensor())  # [0 ~ 1]

        self.functions = transforms.Compose(self.functions)



