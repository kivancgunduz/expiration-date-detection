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

def main():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS)
    else:
        path = Path(__file__).parent
    num_gpus = torch.cuda.device_count()
    print('# of GPUs: {}'.format(num_gpus))
    args, cfg_fcos = fcos_cfgs(path)
    models = load_networks(path)
    detect_rec_dmy = DetectRecognizeDmy(models)
    coco_demo = COCODemo(cfg_fcos,
      confidence_thresholds_for_classes=(args.thresholds_for_classes),
      min_image_size=(args.min_image_size))
    print('##### Webcam Demo Starts #####')
    cam = cv2.VideoCapture(0)
    while 1:
        start_time = time.time()
        ret_val, image = cam.read()
        if image is not None:
            stuff = coco_demo.run_on_opencv_image(image)
            cv2.imshow('Webcam', image)
            image, date_images, tl_info = stuff
            if date_images is not None:
                image, rec_date, _ = detect_rec_dmy(image, date_images, tl_info)
                print('Time: {:.2f} sec/img\tPredicted Date: {}'.format(time.time() - start_time, rec_date))
            cv2.imshow('Webcam', image)
            if cv2.waitKey(1) == 27:
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
# okay decompiling webcam.pyc
