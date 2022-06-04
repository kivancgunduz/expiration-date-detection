# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22)
# [GCC 10.2.1 20210110]
# Embedded file name: webcam.py
import cv2, torch, time, sys
from pathlib import Path
from FCOS.configs.configs import fcos_cfgs
from FCOS.fcos_core.predictor import COCODemo
from DMY.dt_and_rec import DetectRecognizeDmy
from DMY.dmy_utils import load_networks
import numpy as np

if getattr(sys, 'frozen', False):
    path = Path(sys._MEIPASS)
else:
    path = Path(__file__).parent
args, cfg_fcos = fcos_cfgs(path)
models = load_networks(path)
detect_rec_dmy = DetectRecognizeDmy(models)
coco_demo = COCODemo(cfg_fcos,
                     confidence_thresholds_for_classes=(args.thresholds_for_classes),
                     min_image_size=(args.min_image_size))

def img_from_request(request_data):
    return cv2.imdecode(
        np.frombuffer(request_data.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

def img_from_path(path):
    return cv2.imread(path)


def main(image):
    "Image must be of the same type as of the return type of cv2.imread"
    start_time=time.time()
    image, date_images, tl_info = coco_demo.run_on_opencv_image(image)
    if date_images is not None:
        image, rec_date, _ = detect_rec_dmy(image, date_images, tl_info)
        t = time.time()-start_time
        return rec_date, t
    else:
        return "The date cannot be detected.", time.time()-start_time


if __name__ == '__main__':
    path_pic = sys.argv[1]
    result, t = main(img_from_path(path_pic))
    print(result)
    print(f"In {t} s")
# okay decompiling webcam.pyc
