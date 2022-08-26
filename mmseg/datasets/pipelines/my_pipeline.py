import datetime
import os

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
            save_path = os.path.join(self.save_dir, str(self.num) + f"{aug}.jpg")
            Image.fromarray(overlay).save(save_path)
        elif self.num == self.save_num + 1:
            print(f'\033[41mSaved all {self.save_num} images\033[0m')

        return data


@PIPELINES.register_module()
class SwitchBackendToPillow:
    """
    mmcv/image/io.pyではグローバル変数imread_backend = 'cv2'が定義されている。
    各画像処理でbackend指定が有効なものについてはbackendが指定されなかった
    場合にimread_backendの値が使用される。imread_backendの値についてはio.pyの
    use_backend関数にて設定が可能。しかし、mmcv/image/geometric.pyにて
    from .io import imread_backendとしており、ここでimread_backendの値は既に
    固定されてしまっていると思われる。よって、このクラスを用いて柔軟にbackendを変更
    することは出来ないということになる。
    TODO: imread_backendを直接importするのではなく、getterを作成して毎回io.py側の
    値を確認しに行くようにする。
    """

    def __init__(self, imdecode_backend):
        self.imdecode_backend = imdecode_backend

    def __call__(self, data):
        use_backend(self.imdecode_backend)

        return data


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
class OrganDownscale:
    mask_unique_map = {
        'background': [0],
        'kidney': [0, 1],
        'largeintestine': [0, 2],
        'lung': [0, 3],
        'prostate': [0, 4],
        'spleen': [0, 5],
    }

    def __init__(self,
                 prostate_scale_min,
                 prostate_scale_max,
                 others_scale_min,
                 others_scale_max,
                 prostate_p=1,
                 others_p=1
                 ):

        self.prostate_downscale = A.Downscale(scale_min=prostate_scale_min,
                                              scale_max=prostate_scale_max,
                                              interpolation=cv2.INTER_LINEAR,
                                              p=prostate_p
                                              )
        self.others_downscale = A.Downscale(scale_min=others_scale_min,
                                            scale_max=others_scale_max,
                                            interpolation=cv2.INTER_LINEAR,
                                            p=others_p
                                            )

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
        if organ == 'prostate':
            augmented = self.prostate_downscale(image=img, mask=mask)
        else:
            augmented = self.others_downscale(image=img, mask=mask)
        return augmented['image'], augmented['mask']


@PIPELINES.register_module()
class HubmapDataAug:
    mask_unique_map = {
        'background': [0],
        'kidney': [0, 1],
        'largeintestine': [0, 2],
        'lung': [0, 3],
        'prostate': [0, 4],
        'spleen': [0, 5],
    }

    def __init__(self,
                 crop=False,
                 flip=True,
                 rotate_90=True,
                 grid_distortion=False,
                 elastic_transform=False,
                 gaussian_blur=False,
                 hue_saturation_value=False,
                 random_gamma=False
                 ):

        transforms = []

        if crop:
            transforms.append(A.Crop(x_min=100, y_min=100, x_max=1900, y_max=1900, p=0.9))
        if flip:
            transforms.append(A.Flip(p=1))
        if rotate_90:
            transforms.append(A.RandomRotate90(p=1))
        if grid_distortion:
            transforms.append(A.GridDistortion(p=1))
        if elastic_transform:
            transforms.append(A.ElasticTransform(p=1))
        if gaussian_blur:
            transforms.append(A.GaussianBlur(blur_limit=(3, 9), p=0.2))
        if hue_saturation_value:
            transforms.append(A.HueSaturationValue(hue_shift_limit=(-40, 40),
                                                   sat_shift_limit=(-20, 40),
                                                   val_shift_limit=(-40, 40),
                                                   p=1)
                              )
        if random_gamma:
            transforms.append(A.RandomGamma(gamma_limit=(80.0, 250.0), p=0.7))

        self.transforms = A.Compose(transforms)

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
        augmented = self.transforms(image=img, mask=mask)

        return augmented['image'], augmented['mask']
