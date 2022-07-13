
import os
from PIL import Image
import numpy as np
import cv2

small_voc_label_colormap = np.array(
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
     [128, 192, 0],
     [0, 64, 128],
     [128, 64, 128]],
    dtype=np.uint8
)

voc_class_label = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining_table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted_plant',
    'sheep',
    'sofa',
    'train',
    'tv_monitor',
    'void'
]


def get_voc_palette_path():
    abs_dir_path = os.path.dirname(os.path.abspath(__file__))
    voc_palette_img_path = os.path.join(
        abs_dir_path, '../src/voc2012_palette.png')
    if not os.path.isfile(voc_palette_img_path):
        raise FileNotFoundError(
            f'VOC形式カラーパレット取得用のファイルが存在しません: {voc_palette_img_path}')

    return voc_palette_img_path


def convert_rgb_to_palette(pil_img, palette_src):
    """Convert the mode of PIL image from RGB to P(Palette) mode.
       Uses Image.quantize(colors=256, method=None, kmeans=0, palette=None, dither=1)
    Args:
        pil_img: PIL image to convert. Must read from png image.
        palette_src: An example of PIL image that uses required color palette.
    Returns:
        palette_img: palette mode PIL image (.png)
    """

    pil_img = pil_img.convert('RGB')  # to avoid 'L' and RGBA mode
    palette_img = pil_img.quantize(palette=palette_src)

    return palette_img


def convert_rgb_to_voc_palette(pil_img):
    """Convert the mode of PIL image from RGB to P(Palette) mode.
       Uses VOC2012 color palette.
       毎回palette srcを開くため30ms強の遅延が発生する。
       速度を重視する場合はconvert_rgb_to_paletteから実装すること。
    Args:
        pil_img: PIL image to convert. Must read from png image.
    Returns:
        palette_img: palette mode PIL image (.png)
    """
    voc_palette_path = get_voc_palette_path()
    voc_palette_img = Image.open(voc_palette_path)
    assert voc_palette_img.mode == 'P'

    palette_img = convert_rgb_to_palette(
        pil_img=pil_img, palette_src=voc_palette_img)

    return palette_img


def create_voc_label_colormap():
    """Creates a label colormap used in Pascal VOC segmentation benchmark. Cost 1ms.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    index = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((index >> channel) & 1) << shift
        index >>= 3

    return colormap


def segmap_to_segimg(seg_map, colormap):
    """Adds color defined by the dataset colormap to the label.
    Args:
        seg_map: A 2D array with integer type, storing the segmentation label.
    Returns:
        seg_img: A 3D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the voc color map.
    Raises:
        ValueError: If label is not rank 2 or its value is larger than color map maximum entry.
    """

    if seg_map.ndim != 2:
        raise ValueError('Expect 2D input label. Got {}D'.format(seg_map.ndim))

    if np.max(seg_map) >= len(colormap):
        raise ValueError('label value too large')

    seg_img = colormap[seg_map]

    return seg_img


def get_overlay_from_segimg(img, seg_img, alpha=0.7):
    """
    #img: uint8
    #seg_img: uint8
    #overlay: uint8
    # multiplication of numpy array is slow: 30-33ms (cv2.addWeighted: 17-18ms)
    # overlay=seg_img*alpha + img*(1-alpha)
    # overlay=overlay.astype(np.uint8)
    """
    if img.dtype != 'uint8':
        raise ValueError(
            'Arg `img` must be uint8, but got {}.'.format(img.dtype))
    if seg_img.dtype != 'uint8':
        raise ValueError(
            'Arg `seg_img` must be uint8, but got {}.'.format(seg_img.dtype))

    overlay = cv2.addWeighted(seg_img, alpha, img, 1 - alpha, 0)

    return overlay


def get_overlay_from_segmap(img, seg_map, alpha=0.7, small_map=True):
    if small_map:
        colormap = small_voc_label_colormap
    else:
        colormap = create_voc_label_colormap()

    seg_img = segmap_to_segimg(seg_map, colormap)
    overlay = get_overlay_from_segimg(img, seg_img, alpha=alpha)

    return overlay
