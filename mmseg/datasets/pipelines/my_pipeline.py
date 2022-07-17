import datetime
import os
from PIL import Image
import numpy as np
from ..builder import PIPELINES
from ...mylib.seg_utils import get_overlay_from_segmap


@PIPELINES.register_module()
class SaveOverlay:
    def __init__(self, save_root_dir, save_num=100):
        self.save_root_dir = save_root_dir
        self.save_num = save_num
        self.num = 0

        #t = datetime.datetime.now()
        #time_stamp = t.strftime('%y%m%d_%H%M%S')

        self.save_dir = os.path.join(self.save_root_dir, 'overlay')

        try:
            os.mkdir(self.save_dir)
        except FileExistsError as e:
            print(f'\033[41mRemove {self.save_dir} first')
            print(e)
            print('\033[0m')
            raise FileExistsError

    def __call__(self, data):

        self.num += 1

        if self.num <= self.save_num:
            img = data['img']
            mask = data['gt_semantic_seg']

            overlay = get_overlay_from_segmap(img, mask, alpha=0.5, small_map=False)
            aug = data.get('aug_info', '')
            save_path = os.path.join(self.save_dir, str(self.num) + f"{aug}.jpg")
            Image.fromarray(overlay).save(save_path)
        elif self.num == self.save_num + 1:
            print(f'\033[41mSaved all {self.save_num} images\033[0m')

        return data
