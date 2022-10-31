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
from networks.base import finish_build
from .base import UNet


class LargeUNet(UNet):
    """
    Encoder-Decoder in the Paper
    """
    @finish_build
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ======= number of channel layer control =======
        """
        "U-Net 1"
        """
        c1 = 16
        c2 = 32
        c3 = 64
        c4 = 128
        c5 = 256
        c6 = 256
        self.c7 = 512
        cf = 32

        N = 3  # number of input directions
        # ===============================================

        self.downconv_1 = self.downconv_double(3, c1, 7)
        self.downconv_2 = self.downconv_double(c1, c2, 5)
        self.downconv_3 = self.downconv_double(c2, c3, 3)
        self.downconv_4 = self.downconv_double(c3, c4, 3)
        self.downconv_5 = self.downconv_double(c4, c5, 3)
        self.downconv_6 = self.downconv_double(c5, c6, 3)
        self.downconv_7 = self.downconv_double(c6, self.c7, 3)

        self.upconv_7 = self.upconv_single(self.c7 * N, c6 * N, 3, 2)
        self.upconv_6 = self.upconv_single(c6 * N, c5 * N, 3, 2)
        self.upconv_5 = self.upconv_single(c5 * N, c4 * N, 3, 2)
        self.upconv_4 = self.upconv_single(c4 * N, c3 * N, 3, 2)
        self.upconv_3 = self.upconv_single(c3 * N, c2 * N, 3, 2)
        self.upconv_2 = self.upconv_single(c2 * N, c1 * N, 3, 2)
        self.upconv_1 = self.upconv_single(c1 * N, cf, 3, 2)

        self.conv_7 = self.identity_conv(c6 * N * 2, c6 * N)
        self.conv_6 = self.identity_conv(c5 * N * 2, c5 * N)
        self.conv_5 = self.identity_conv(c4 * N * 2, c4 * N)
        self.conv_4 = self.identity_conv(c3 * N * 2, c3 * N)
        self.conv_3 = self.identity_conv(c2 * N * 2, c2 * N)
        self.conv_2 = self.identity_conv(c1 * N * 2, c1 * N)
        self.conv_1 = self.identity_conv(cf, 16)
