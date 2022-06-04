import cv2, torch, time, sys
from DMY.dt_and_rec import DetectRecognizeDmy
from DMY.dmy_utils import load_networks
from FCOS.fcos_core.predictor import COCODemo
from FCOS.configs.configs import fcos_cfgs
from pathlib import Path

def load():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS)
        print(path)
    else:
        path = Path(__file__).parent
        print(path)
    num_gpus = torch.cuda.device_count()
    print('# of GPUs: {}'.format(num_gpus))
    args, cfg_fcos = fcos_cfgs(path)
    models = load_networks(path)
    detect_rec_dmy = DetectRecognizeDmy(models)
    coco_demo = COCODemo(cfg_fcos,
                        confidence_thresholds_for_classes=(args.thresholds_for_classes),
                        min_image_size=(args.min_image_size))
                        
    return detect_rec_dmy, coco_demo