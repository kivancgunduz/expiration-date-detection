# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/dt_and_rec.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 4229 bytes
from torchvision import transforms as T
from DMY.Evaluation.utils_test import *
from DMY.Evaluation.predictor import PostProcessor
from DAN.utils import Attention_AR_counter
from DMY import cfg

class DetectRecognizeDmy:

    def __init__(self, models):
        super(DetectRecognizeDmy, self).__init__()
        self.model_dmy, self.model_fe, self.model_cam, self.model_dtd = models
        self.get_predictions = PostProcessor()
        self.test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfg.dataset_config['dict_dir'], cfg.dataset_config['case_sensitive'])
        self.transforms = T.Compose([
         T.ToPILImage(),
         T.Resize((64, 256)),
         T.ToTensor()])
        self.transforms_rec = T.Compose([
         T.Resize((32, 128)),
         T.ToTensor()])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def convert2BGR(self, image, width, height):
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (width, height), interpolation=(cv2.INTER_AREA))
        return img

    def __call__(self, orig_image, dmy_images, tl_info):
        """
        First detects day, month, and year components. Then, recognized the characters in the detected components.
        Arguments:
            orig_image: original input image as returned by OpenCV
            dmy_image: detected expiration date region
            tl_info: top-left corner coordinate of the detected expiration date box
            models: dmy detection (model_dmy) and recognition networks (model_fe, model_cam, model_dtd)
        Return:
            final_img: detection and recognition results on original image
            prd_date: recognized expiration date
            labels: day, month, and year labels for recognition results
        """
        collect_images = {}
        collect_prd_date = {}
        collect_box = {}
        for idx, dmy_image in enumerate(dmy_images):
            image = self.transforms(dmy_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cls_pred, bbox_pred, centerness, features = self.model_dmy(image)
            locations = compute_locations(features)
            boxes = self.get_predictions.forward(locations, cls_pred, bbox_pred, centerness, image.size()[2:])
            prediction = [pred.to('cpu') for pred in boxes][0]
            height, width = dmy_image.shape[:2]
            prediction = prediction.resize((width, height))
            top_predictions = sort_predictions(prediction)
            if len(top_predictions.bbox) != 0:
                top_predictions = select_top_one_per_class(top_predictions, (width, height))
                cropped_imgs = crop_boxes(dmy_image, top_predictions, self.transforms_rec).to(self.device)
                with torch.no_grad():
                    features = self.model_fe(cropped_imgs)
                    A = self.model_cam(features)
                    output, out_length = self.model_dtd(features[(-1)], A)
                prd_date, top_predictions = self.test_acc_counter.add_iter(output, out_length, top_predictions)
                dict_date, dict_box = date_dictionary(top_predictions, prd_date)
                collect_images[f"date_{idx + 1}"] = dmy_image
                collect_prd_date[f"date_{idx + 1}"] = dict_date
                collect_box[f"date_{idx + 1}"] = dict_box

        final_img, prd_date, prd_boxes, tl_corner = select_expiration_date(collect_images, collect_prd_date, collect_box, tl_info)
        labels = None
        if final_img is not None:
            orig_image = overlay_boxes(prd_date, prd_boxes, orig_image, tl_corner)
            prd_date, labels = merge_date_labels(prd_date)
        return (
         orig_image, prd_date, labels)
# okay decompiling ./dt_and_rec.pyc
