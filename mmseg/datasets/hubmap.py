"""
mmsegmentation/mmseg/datasets/__init__.pyにモジュールを追加すること
img_suffix='.png'へ変更
FileNotFoundError: [Errno 2] No such file or directory: '/content/hubmap_512x512/images/18426.jpg'
"""

import os
import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HubmapDataset(CustomDataset):
    CLASSES = ('background', 'cell')
    PALETTE = np.array(
        [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0]],
        dtype=np.uint8
    )
    PALETTE = PALETTE[:len(CLASSES)]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert os.path.exists(self.img_dir) and self.split is not None
