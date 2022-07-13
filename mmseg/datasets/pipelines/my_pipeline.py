import os
import cv2

from mmseg.datasets import PIPELINES
from mmseg.mylib.seg_utils import get_overlay_from_segmap


@PIPELINES.register_module()
class SaveOverlay:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.num = 0

    def __call__(self, data):
        img = data['img']
        mask = data['gt_semantic_seg']

        overlay = get_overlay_from_segmap(img, mask, alpha=0.7)
        cv2.imwrite(os.path.join(self.save_dir, str(self.num) + '.jpg'), overlay)

        return data
