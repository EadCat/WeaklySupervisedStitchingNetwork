"""
A Library from https://github.com/aleju/imgaug
"""
import numpy as np
import random
from PIL import Image
import imgaug.augmenters as iaa


class AugCompose(object):
    """Data Augmentation Class using imgaug"""
    def __init__(self):
        self.gamma = random.uniform(0.8, 1.2)
        self.bright = random.uniform(0.5, 2.0)

    def __call__(self, pilimage: list):
        imgs = []
        gamma_tr = iaa.GammaContrast(gamma=self.gamma)
        bright_tr = iaa.Multiply(mul=self.bright)
        for img in pilimage:
            assert isinstance(img, Image.Image)
            np_img = np.array(img, dtype=np.uint8)
            np_img = gamma_tr.augment_image(np_img)
            np_img = bright_tr.augment_image(np_img)
            img = Image.fromarray(np_img)
            imgs.append(img)

        return imgs
