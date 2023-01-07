import os
import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RSNADataset(CustomDataset):
    CLASSES = ('background', 'cancer')
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
        # img_suffix='.png', seg_map_suffix='.png'を削除
        # configのdata.test.img_suffix等で指定すること
        super().__init__(split=split, **kwargs)
        assert os.path.exists(self.img_dir) and self.split is not None
