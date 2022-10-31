"""
Build VGG-16 from torchvision
"""
import glob

import torch
import torch.nn as nn

import re
from typing import Union, List, cast
from collections import OrderedDict

# cfgs: Dict[str, List[Union[str, int]]] = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#


class VGG16(nn.Module):
    """
    Edit VGG-16 OrderedDict to get feature maps from maxpooling-layers.
    """
    def __init__(self):
        super(VGG16, self).__init__()
        self.cnn_module1 = self.make_layers(3, [64, 64, 'M'])  # 0.weight, 0.bias / 2.weight, 2.bias
        self.cnn_module2 = self.make_layers(64, [128, 128, 'M'])  # 0.weight, 0.bias / 2.weight, 2.bias
        self.cnn_module3 = self.make_layers(128, [256, 256, 256, 'M'])  # 0 2 4
        self.cnn_module4 = self.make_layers(256, [512, 512, 512, 'M'])  # 0 2 4
        self.cnn_module5 = self.make_layers(512, [512, 512, 512, 'M'])  # 0 2 4
        self.target_name = ['features.0', 'features.0', 'features.2', 'features.2',
                            'features.5', 'features.5', 'features.7', 'features.7',
                            'features.10', 'features.10', 'features.12', 'features.12', 'features.14', 'features.14',
                            'features.17', 'features.17', 'features.19', 'features.19', 'features.21', 'features.21',
                            'features.24', 'features.24', 'features.26', 'features.26', 'features.28', 'features.28']
        self.module_name = ['cnn_module1.0', 'cnn_module1.0', 'cnn_module1.2', 'cnn_module1.2',
                            'cnn_module2.0', 'cnn_module2.0', 'cnn_module2.2', 'cnn_module2.2',
                            'cnn_module3.0', 'cnn_module3.0', 'cnn_module3.2', 'cnn_module3.2',
                            'cnn_module3.4', 'cnn_module3.4',
                            'cnn_module4.0', 'cnn_module4.0', 'cnn_module4.2', 'cnn_module4.2',
                            'cnn_module4.4', 'cnn_module4.4',
                            'cnn_module5.0', 'cnn_module5.0', 'cnn_module5.2', 'cnn_module5.2',
                            'cnn_module5.4', 'cnn_module5.4']
        self.load()

    def load(self):
        """Load pre-trained VGG-16 model."""
        target_file = glob.glob('./loss/vgg/*.pth')[0]
        state_dict = torch.load(target_file)

        # remove classifier load information
        remove_keys = re.compile('classifier.')
        remove_keys = [string for string in state_dict.keys() if re.match(remove_keys, string)]
        for key in remove_keys:
            state_dict.pop(key)

        adaptation = OrderedDict()

        # key-name change
        for target, fix, (key, value) in zip(self.target_name, self.module_name, state_dict.items()):
            replaced = key.replace(target, fix)
            adaptation[replaced] = value

        self.load_state_dict(adaptation, strict=True)
        del state_dict
        del adaptation

    def forward(self, x):
        """Get feature maps from maxpooling-layers."""
        p_d1 = self.cnn_module1(x)
        p_d2 = self.cnn_module2(p_d1)
        p_d3 = self.cnn_module3(p_d2)
        p_d4 = self.cnn_module4(p_d3)
        p_d5 = self.cnn_module5(p_d4)

        return p_d1, p_d2, p_d3, p_d4, p_d5

    @staticmethod
    def make_layers(in_channel, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        """Make layers for VGG-16."""
        layers: List[nn.Module] = []
        in_channels = in_channel
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)