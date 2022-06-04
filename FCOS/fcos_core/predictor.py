# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/predictor.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 14282 bytes
import cv2, torch, numpy as np, copy
from torchvision import transforms as T
from FCOS.fcos_core.modeling.detector import build_detection_model
from FCOS.fcos_core.utils.checkpoint import DetectronCheckpointer
from FCOS.fcos_core.structures.image_list import to_image_list
from FCOS.fcos_core import layers as L
cnt = 1

class COCODemo(object):
    CATEGORIES = [
     'background',
     'code',
     'due',
     'exp',
     'prod']

    def __init__(self, cfg, confidence_thresholds_for_classes, show_mask_heatmaps=False, masks_per_dim=2, min_image_size=224):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, (self.model), save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()
        self.palette = torch.tensor([33554431, 32767, 2097151])
        self.cpu_device = torch.device('cpu')
        self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
        normalize_transform = T.Normalize(mean=(cfg.INPUT.PIXEL_MEAN),
          std=(cfg.INPUT.PIXEL_STD))
        transform = T.Compose([
         T.ToPILImage(),
         T.Resize(self.min_image_size),
         T.ToTensor(),
         to_bgr_transform,
         normalize_transform])
        return transform

    @staticmethod
    def crop_boxes(img, boxes):
        """
        Crops expiration date region.
        """
        cropped_img = []
        tl_info = {}
        tb, lr = (5, 5)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().cpu().numpy()
            if x1 <= lr or y1 <= tb:
                crop_img = img[y1:y2 + tb, x1:x2 + lr]
            else:
                crop_img = img[y1 - tb:y2 + tb, x1 - lr:x2 + lr]
            cropped_img.append(crop_img)
            tl_info[f"date_{idx + 1}"] = (x1 - lr, y1 - tb)

        return (cropped_img, tl_info)

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        date_images = None
        tl_info = None
        if 3 in top_predictions.extra_fields['labels']:
            exp_idx = torch.nonzero(top_predictions.extra_fields['labels'] == 3).squeeze()
            exp_box = torch.index_select(top_predictions.bbox, 0, exp_idx)
            date_images, tl_info = self.crop_boxes(image.copy(), exp_box)
        image = self.overlay_boxes(image, top_predictions)
        image = self.overlay_class_names(image, top_predictions)
        return (
         image, date_images, tl_info)

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions[0]]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        if prediction.has_field('mask'):
            masks = prediction.get_field('mask')
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field('mask', masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field('scores')
        labels = predictions.get_field('labels')
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field('scores')
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    @staticmethod
    def compute_colors_for_labels(labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        color_list = []
        for label in labels:
            if label == 1:
                color_list.append(np.array([255, 0, 0]).astype('uint8'))
            else:
                if label == 2:
                    color_list.append(np.array([0, 128, 255]).astype('uint8'))
                else:
                    if label == 3:
                        color_list.append(np.array([0, 255, 0]).astype('uint8'))
                    else:
                        color_list.append(np.array([128, 0, 128]).astype('uint8'))

        colors = np.array(color_list).reshape(-1, 3)
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field('labels')
        boxes = predictions.bbox
        colors = self.compute_colors_for_labels(labels).tolist()
        for box, color, label in zip(boxes, colors, labels):
            if label != 3:
                box = box.to(torch.int64)
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field('scores').tolist()
        labels = predictions.get_field('labels').tolist()
        colors = self.compute_colors_for_labels(labels).tolist()
        boxes = predictions.bbox
        size = 0.8
        thick = 2
        for box, score, label, color in zip(boxes, scores, labels, colors):
            if self.CATEGORIES[label] != 'exp':
                x, y = box[:2].numpy().astype('int')
                s = '{}'.format(self.CATEGORIES[label])
                cv2.putText(image, s, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)

        return image
# okay decompiling ./fcos_core/predictor.pyc
