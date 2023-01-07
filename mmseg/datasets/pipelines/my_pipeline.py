import datetime
import os
import glob

import albumentations as A
import cv2
from PIL import Image
import numpy as np
from ..builder import PIPELINES
from mmcv import use_backend

from ...mylib.seg_utils import get_overlay_from_segmap


@PIPELINES.register_module()
class SaveOverlay:
    def __init__(self, save_root_dir, save_num=100, no_overlay=False):
        self.save_root_dir = save_root_dir
        self.save_num = save_num
        self.num = 0
        self.no_overlay = no_overlay

        self.save_dir = os.path.join(self.save_root_dir, 'overlay')

        if os.path.isdir(self.save_dir):
            time_stamp = datetime.datetime.now().strftime('%y%m%d%H%M%S%f')
            self.save_dir += f'_{time_stamp}'

        os.mkdir(self.save_dir)

    def __call__(self, data):

        self.num += 1

        if self.num <= self.save_num:
            img = data['img']
            mask = data['gt_semantic_seg']

            if self.no_overlay:
                overlay = img
            else:
                overlay = get_overlay_from_segmap(img, mask, alpha=0.5, small_map=False)
            aug = data.get('aug_info', '')

            img = Image.fromarray(overlay)
            w, h = img.size
            save_path = f'{self.save_dir}/{self.num}_{aug}_{h}x{w}.jpg'
            img.save(save_path)

        elif self.num == self.save_num + 1:
            print(f'\033[41mSaved all {self.save_num} images\033[0m')

        return data


@PIPELINES.register_module()
class SaveImg:
    # 毎回インスタンスを作成しているため、SaveOverlayとは異なる実装となっている。

    def __init__(self, save_dir, save_num=100):
        self.save_dir = save_dir
        self.save_num = save_num
        self.num = 0

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def __call__(self, data):
        files = glob.glob(f'{self.save_dir}/*.jpg')

        if len(files) <= self.save_num:
            img = data['img']
            img = Image.fromarray(img)
            w, h = img.size

            time_stamp = datetime.datetime.now().strftime('%y%m%d%H%M%S%f')
            save_path = f'{self.save_dir}/{str(time_stamp)}_{h}x{w}.jpg'
            img.save(save_path)

        return data


# @PIPELINES.register_module()
# class SwitchBackendToPillow:
#     """
#     mmcv/image/io.pyではグローバル変数imread_backend = 'cv2'が定義されている。
#     各画像処理でbackend指定が有効なものについてはbackendが指定されなかった
#     場合にimread_backendの値が使用される。imread_backendの値についてはio.pyの
#     use_backend関数にて設定が可能。しかし、mmcv/image/geometric.pyにて
#     from .io import imread_backendとしており、ここでimread_backendの値は既に
#     固定されてしまっていると思われる。よって、このクラスを用いて柔軟にbackendを変更
#     することは出来ないと思われる。
#     TODO: imread_backendを直接importするのではなく、getterを作成して毎回io.py側の
#     値を確認しに行くようにする。
#     """

#     def __init__(self, imdecode_backend):
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, data):
#         use_backend(self.imdecode_backend)

#         return data


@PIPELINES.register_module()
class Convert2Class1:
    def __call__(self, data):
        mask = data['gt_semantic_seg']

        mask_ignore = (mask == 255)
        mask = np.where((mask >= 1) & (mask < 255), 1, 0)
        mask[mask_ignore] = 255

        data['gt_semantic_seg'] = mask

        return data


@PIPELINES.register_module()
class OrgansDataAug:
    mask_unique_map = {
        'background': [0],
        'kidney': [0, 1],
        'largeintestine': [0, 2],
        'lung': [0, 3],
        'prostate': [0, 4],
        'spleen': [0, 5],
    }

    def __init__(self):

        self.transforms = dict(
            kidney=[],
            largeintestine=[],
            lung=[],
            prostate=[],
            spleen=[]
        )

        downscale = A.Downscale(scale_min=0.08, scale_max=0.4, interpolation=cv2.INTER_LINEAR, p=0.9)
        self.register(downscale, ['prostate'])

        crop = A.Crop(x_min=100, y_min=100, x_max=1900, y_max=1900, p=0.7)
        self.register(crop, ['all'])

        flip = A.Flip(p=1)
        self.register(flip, ['all'])

        rotate_90 = A.RandomRotate90(p=1)
        self.register(rotate_90, ['all'])

        # grid_distortion = A.GridDistortion(p=1)
        # self.register(grid_distortion, ['all'])

        # elastic_transform=A.ElasticTransform(p=1)
        # self.register(elastic_transform, ['all'])

        # gaussian_blur=A.GaussianBlur(blur_limit=(3, 9), p=0.2)
        # self.register(gaussian_blur, ['all'])

        hue_saturation_value = A.HueSaturationValue(hue_shift_limit=(-20, 20),
                                                    sat_shift_limit=(-20, 20),
                                                    val_shift_limit=(-20, 20),
                                                    p=1)
        self.register(hue_saturation_value, ['all'])

        random_gamma = A.RandomGamma(gamma_limit=(25.0, 250.0), p=1)
        random_gamma_strong = A.RandomGamma(gamma_limit=(50.0, 300.0), p=1)
        self.register(random_gamma, ['lung', 'prostate', 'spleen'])
        self.register(random_gamma_strong, ['kidney', 'largeintestine'])

        random_brightness = A.RandomBrightness(limit=(-0.2, 0.1), p=1)
        self.register(random_brightness, ['all'])

        for organ in self.transforms:
            print(f'[{self.__class__.__name__}] organ:{organ} transforms:{self.transforms[organ]}')
            self.transforms[organ] = A.Compose(self.transforms[organ])

    def register(self, T, organ_list):
        if 'all' in organ_list:
            for organ in self.transforms:
                self.transforms[organ].append(T)
        else:
            for organ in self.transforms:
                if organ in organ_list:
                    self.transforms[organ].append(T)

    def __call__(self, data):
        img = data['img']
        mask = data['gt_semantic_seg']

        mask_unique = np.unique(mask)
        organ = ''
        for key, val in self.mask_unique_map.items():
            if val == list(mask_unique):
                organ = key
                break
        if organ == '':
            raise ValueError(f'mask values mismatch. Got {mask_unique}')

        img, mask = self.data_aug(img, mask, organ)
        data['img'] = img
        data['gt_semantic_seg'] = mask

        return data

    def data_aug(self, img, mask, organ):
        augmented = self.transforms[organ](image=img, mask=mask)

        return augmented['image'], augmented['mask']
