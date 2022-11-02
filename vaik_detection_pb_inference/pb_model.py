from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np

from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.classes = classes

    def inference(self, input_image: np.ndarray, resize_input_shape=(None, None), score_th: float = 0.2,
                  nms_th: float = 0.5) -> Tuple[
        List[Dict], Dict
    ]:
        org_image_shape = input_image.shape
        if None not in resize_input_shape:
            input_image = self.__preprocess(input_image, resize_input_shape)
        raw_pred = self.model(np.expand_dims(input_image, 0))
        filter_pred = self.__filter_score(raw_pred, score_th)
        filter_pred = self.__filter_nms(filter_pred, nms_th)
        objects_dict_list = self.__output_parse(filter_pred, org_image_shape[:-1], score_th)
        return objects_dict_list, raw_pred

    def __preprocess(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, float]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        x_ratio, y_ratio = resize_input_shape[1] / input_image.shape[1], resize_input_shape[0] / input_image.shape[0]
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(input_image.shape[0] * x_ratio))
        else:
            resize_size = (round(input_image.shape[1] * y_ratio), resize_input_shape[0])

        pil_input_image = Image.fromarray(input_image)
        resize_pil_image = pil_input_image.resize(resize_size)
        output_image = np.array(resize_pil_image)

        return output_image

    def __output_parse(self, filter_pred: Dict, input_image_shape: Tuple[int, int]) -> List[Dict]:
        objects_dict_list = []
        for pred_index in range(filter_pred['num_detections']):
            score = float(filter_pred['detection_scores'][pred_index])
            classes_index = int(filter_pred['detection_classes'][pred_index])
            name = str(classes_index) if self.classes is None else self.classes[classes_index - 1]
            ymin = max(0, int((filter_pred['detection_boxes'][pred_index][0] * input_image_shape[0])))
            xmin = max(0, int((filter_pred['detection_boxes'][pred_index][1] * input_image_shape[1])))
            ymax = min(input_image_shape[0] - 1,
                       int((filter_pred['detection_boxes'][pred_index][2] * input_image_shape[0])))
            xmax = min(input_image_shape[1] - 1,
                       int((filter_pred['detection_boxes'][pred_index][3] * input_image_shape[1])))
            object_extend_dict = {'score': score}
            objects_dict = pascal_voc_rw_ex.get_objects_dict_template(name, xmin, ymin, xmax, ymax,
                                                                      object_extend_dict=object_extend_dict)
            objects_dict_list.append(objects_dict)
        return objects_dict_list

    @classmethod
    def __filter_score(self, pred, score_th):
        mask = pred['detection_scores'][0] > score_th
        filter_pred = {}
        filter_pred['detection_anchor_indices'] = tf.boolean_mask(pred['detection_anchor_indices'][0], mask).numpy()
        filter_pred['detection_boxes'] = tf.boolean_mask(pred['detection_boxes'][0], mask).numpy()
        filter_pred['detection_classes'] = tf.boolean_mask(pred['detection_classes'][0], mask).numpy()
        filter_pred['detection_scores'] = tf.boolean_mask(pred['detection_scores'][0], mask).numpy()
        filter_pred['detection_multiclass_scores'] = tf.boolean_mask(pred['detection_multiclass_scores'][0],
                                                                     mask).numpy()
        filter_pred['num_detections'] = int(filter_pred['detection_multiclass_scores'].shape[0])
        return filter_pred

    # Ref. https://python-ai-learn.com/2021/02/14/nmsfast/
    @classmethod
    def __calc_iou(cls, source_array, dist_array, source_area, dist_area):
        x_min = np.maximum(source_array[0], dist_array[:, 0])
        y_min = np.maximum(source_array[1], dist_array[:, 1])
        x_max = np.minimum(source_array[2], dist_array[:, 2])
        y_max = np.minimum(source_array[3], dist_array[:, 3])
        w = np.maximum(0, x_max - x_min + 0.0000001)
        h = np.maximum(0, y_max - y_min + 0.0000001)
        intersect_area = w * h
        iou = intersect_area / (source_area + dist_area - intersect_area)
        return iou

    # Ref. https://python-ai-learn.com/2021/02/14/nmsfast/
    @classmethod
    def __filter_nms(cls, filter_pred, nms_th):
        bboxes = filter_pred['detection_boxes']
        areas = ((bboxes[:, 2] - bboxes[:, 0] + 0.0000001) * (bboxes[:, 3] - bboxes[:, 1] + 0.0000001))
        sort_index = np.argsort(filter_pred['detection_scores'])
        i = -1
        while (len(sort_index) >= 2 - i):
            max_scr_ind = sort_index[i]
            ind_list = sort_index[:i]
            iou = cls.__calc_iou(bboxes[max_scr_ind], bboxes[ind_list], areas[max_scr_ind], areas[ind_list])
            del_index = np.where(iou >= nms_th)
            sort_index = np.delete(sort_index, del_index)
            i -= 1

        filter_pred['detection_anchor_indices'] = filter_pred['detection_anchor_indices'][sort_index][::-1]
        filter_pred['detection_boxes'] = filter_pred['detection_boxes'][sort_index][::-1]
        filter_pred['detection_classes'] = filter_pred['detection_classes'][sort_index][::-1]
        filter_pred['detection_scores'] = filter_pred['detection_scores'][sort_index][::-1]
        filter_pred['detection_multiclass_scores'] = filter_pred['detection_multiclass_scores'][sort_index][::-1]
        filter_pred['num_detections'] = int(filter_pred['detection_multiclass_scores'].shape[0])
        return filter_pred
