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
import matplotlib.pyplot as plt


class SetInfo:
    """
    Structure for PlotGenerator
    """
    def __init__(self, legend, color:str, linewidth:float, alpha:float, animated:bool, linestyle:str,
                 visible:bool, dash_capstyle:str, dash_joinstyle:str):
        self.color = None
        self.name = None
        self.linewidth = None
        self.linestyle = None
        self.alpha = None
        self.animated = None
        self.visible = None
        self.dash_joinstyle = None
        self.dash_capstyle = None

        self.update(legend=legend, color=color, linewidth=linewidth, alpha=alpha, animated=animated,
                    linestyle=linestyle, visible=visible, dash_capstyle=dash_capstyle, dash_joinstyle=dash_joinstyle)

    def update(self, legend, color, linewidth, linestyle, alpha, visible, animated, dash_capstyle, dash_joinstyle):
        """Set the Plot Options"""
        self.color = color
        self.alpha = alpha
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.animated = animated
        self.visible = visible
        self.dash_capstyle = dash_capstyle
        self.dash_joinstyle = dash_joinstyle
        self.name = legend


class PlotGenerator:
    """Plot Generator for Loss History"""
    def __init__(self, number: int, title, size: tuple, xlabel='x', ylabel='y'):
        self.figure_num = number
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.size = size

        self.datalen = 0
        self.setlen = 0
        self.datalist = []  # a list of dictionaries
        self.setlist = []
        self.image = None

    def __len__(self):
        print(f'existing data : {self.datalen}')
        print(f'existing set  : {self.setlen}')
        return self.datalen

    def plot(self, reverse=False, legend=True):
        # plot graph.
        # at least you must run 'self.add_data', 'self.add_set' method.
        """
        :param reverse: swap [x axis data] and [y axis data]
        :param legend: show legend.
        :return: nothing.
        """
        assert self.datalen > 0, 'data container empty.'
        assert self.setlen > 0, 'setting container empty'
        assert self.datalen == self.setlen, 'The number of data and settings are not the same.'
        self.image = plt.figure(num=self.figure_num, figsize=self.size)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for idx in range(self.datalen):
            if not reverse:
                x_data = list(self.datalist[idx].keys())
                y_data = list(self.datalist[idx].values())
            else:
                y_data = list(self.datalist[idx].keys())
                x_data = list(self.datalist[idx].values())

            plt.plot(x_data, y_data, color=self.setlist[idx].color, animated=self.setlist[idx].animated,
                     label=self.setlist[idx].name, ls=self.setlist[idx].linestyle, lw=self.setlist[idx].linewidth,
                     visible=self.setlist[idx].visible, dash_capstyle=self.setlist[idx].dash_capstyle,
                     dash_joinstyle=self.setlist[idx].dash_joinstyle, alpha=self.setlist[idx].alpha)
        if legend:
            plt.legend()

    def show(self):
        """show plot figure."""
        self.image.show()

    def save(self, dir:str):
        """save plot figure."""
        base = os.path.dirname(dir)
        os.makedirs(base, exist_ok=True)
        self.image.savefig(dir)

    def clear(self):
        """Clear plot figure."""
        self.datalist.clear()
        self.setlist.clear()
        self.datalen = 0
        self.setlen = 0
        plt.clf()

    def add_data(self, data, std:str=' : '):
        """add a plot data to the end of instance's data container."""
        if isinstance(data, str):
            self.datalist.append(self.parsing(data, std))

        elif isinstance(data, dict):
            self.datalist.append(data)

        self.datalen += 1

    def add_set(self, name=None, color=None, linewidth=3.0, linestyle='-',
                visible=True, animated=False, alpha=1.0,
                dash_capstyle='round', dash_joinstyle='round',
                data=None):
        """
        add a plot setting(option) to the end of instance's setting container.
        if you add SetInfo Structure, give 'data' parameter.
        """
        if isinstance(data, SetInfo):
            self.setlist.append(data)
        else:
            if color is None:
                color = 'r'
            if name is None:
                name = 'dataset' + str(self.setlen+1)
            setting = SetInfo(legend=name, color=color, linewidth=linewidth, linestyle=linestyle,
                              visible=visible, animated=animated, alpha=alpha, dash_capstyle=dash_capstyle,
                              dash_joinstyle=dash_joinstyle)
            self.setlist.append(setting)

        self.setlen += 1

    def sub_data(self, idx=-1):
        """remove the plot data of the targeted index."""
        self.datalist.pop(idx)
        self.datalen -= 1

    def sub_set(self, idx=-1):
        """remove the plot setting of the targeted index."""
        self.setlist.pop(idx)
        self.setlen -= 1

    def fix_data(self, idx:int, data, std:str=' : '):
        """fix plot data."""
        assert idx >= 0, 'index number must be bigger than 0.'
        assert idx < self.datalen, 'out of index.'

        if isinstance(data, str):
            self.datalist[idx] = self.parsing(data, std)
        elif isinstance(data, dict):
            self.datalist[idx] = data

    def fix_set(self, idx:int, name=None, color=None, linewidth=3.0, linestyle='-',
                visible=True, animated=False, alpha=1.0,
                dash_capstyle='round', dash_joinstyle='round',
                data=None):
        """fix plot setting"""
        assert idx >= 0, 'index number must be bigger than 0.'
        assert idx < self.setlen, 'out of index.'
        if isinstance(data, SetInfo):
            self.setlist[idx] = data
        else:
            if color is None:
                color = 'r'
            if name is None:
                name = 'dataset' + str(idx+1)
            setting = SetInfo(legend=name, color=color, linewidth=linewidth, linestyle=linestyle,
                              visible=visible, animated=animated, alpha=alpha, dash_capstyle=dash_capstyle,
                              dash_joinstyle=dash_joinstyle)
            self.setlist[idx] = setting

    @staticmethod
    def parsing(file:str, std:str= ' : ') -> dict:
        """
        read text file.
        see functions.utils.write_line function.
        """
        assert os.path.isfile(file), f"There's no such file."
        with open(file, 'r') as f:
            lines = f.readlines()

        text_dict = {}

        for line in lines:
            key, value = line.split(std)
            value = float(value)
            text_dict[key] = value  # data type str

        return text_dict

    def interval_remove(self, interval:int, idx:int, update=False) -> dict:
        """
        :param interval: the number of data to be removed periodically.
        :param idx: index of data dictionary
        :param update: decide whether to update the original data dictionary. (overlay original plot data)
        :return: compressed dictionary
        """
        assert interval > 0, 'interval must be a positive integer.'
        i = 0
        new_dict = {}
        for key, value in self.datalist[idx].items():
            if i == 0:
                new_dict[key] = value
                i += 1
            elif i == interval:
                i = 0
            else:
                i += 1

        if update:
            self.datalist[idx] = new_dict

        return new_dict

    def cut(self, idx:int, start=0, end=-1, update=False):
        """
        :param idx: index of data dictionary
        :param start: cutting start point of data dictionary
        :param end: cutting end point of data dictionary
        :param update: decide whether to update the original data dictionary.
        :return: cut data dictionary
        """
        # assumption: key:value 1 by 1 mapping
        keys = list(self.datalist[idx].keys())
        values = list(self.datalist[idx].values())

        keys = keys[start:end]
        values = values[start:end]

        new_dict = {}

        for key, value in zip(keys, values):
            new_dict[key] = value

        if update:
            self.datalist[idx] = new_dict

        return new_dict

    def data(self, idx:int) -> dict:
        """return data dictionary of the targeted index."""
        return self.datalist[idx]

    def set(self, idx:int) -> SetInfo:
        """return SetInfo of the targeted index."""
        return self.setlist[idx]
