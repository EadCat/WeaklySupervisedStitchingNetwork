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
import pandas as pd
import torch


def snapshot_maker(param_dict, log_dir:str):
    """
    overwrite all contents
    record <.pth> model infomation snapshot.
    """
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(log_dir, 'w') as file:
        for key, value in param_dict.items():
            file.write(key + ' : ' + str(value) + '\n')
        time_now = datetime.datetime.now()
        file.write('Record Time : ' + time_now.strftime('%Y-%m-%d %H:%M:%S'))


def write_line(dict_in: dict, log_dir: str):
    """record loss in real time (TXT)."""
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    with open(log_dir, 'a') as file:
        for key, value in dict_in.items():
            if isinstance(key, torch.Tensor):
                key = float(key)
            if isinstance(value, torch.Tensor):
                value = float(value)
            if isinstance(key, float):
                key = round(key, 4)
            if isinstance(value, float):
                value = round(value, 6)
            file.write(str(key) + ' : ' + str(value) + '\n')


def csv_record(input_dict, csv_dir):
    """record loss in real time (CSV)."""
    dataframe = pd.DataFrame.from_records([input_dict])
    if not os.path.exists(csv_dir):  # writing mode with header
        dataframe.to_csv(csv_dir, index=False, mode='w', encoding='utf-8-sig')
    else:                            # append mode without header
        dataframe.to_csv(csv_dir, index=False, mode='a', encoding='utf-8-sig', header=False)
