import datetime
import os
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
