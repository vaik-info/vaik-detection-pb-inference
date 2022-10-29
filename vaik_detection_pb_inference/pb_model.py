from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np

from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.classes = classes

    def inference(self, input_image: np.ndarray, resize_input_shape=(None, None), score_th: float = 0.2) -> Tuple[
        List[Dict], Dict
    ]:
        org_image_shape = input_image.shape
        if None not in resize_input_shape:
            input_image = self.__preprocess(input_image, resize_input_shape)
        raw_pred = self.model(np.expand_dims(input_image, 0))
        objects_dict_list = self.__output_parse(raw_pred, org_image_shape[:-1], score_th)
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

    def __output_parse(self, raw_pred: Dict, input_image_shape: Tuple[int, int], score_th: float) -> List[Dict]:
        objects_dict_list = []
        for pred_index in range(int(raw_pred['num_detections'].numpy())):
            score = float(raw_pred['detection_scores'][0][pred_index].numpy())
            if score > score_th:
                classes_index = int(raw_pred['detection_classes'][0][pred_index].numpy())
                name = str(classes_index) if self.classes is None else self.classes[classes_index - 1]
                ymin = int((raw_pred['detection_boxes'][0][pred_index][0] * input_image_shape[0]).numpy())
                xmin = int((raw_pred['detection_boxes'][0][pred_index][1] * input_image_shape[1]).numpy())
                ymax = int((raw_pred['detection_boxes'][0][pred_index][2] * input_image_shape[0]).numpy())
                xmax = int((raw_pred['detection_boxes'][0][pred_index][3] * input_image_shape[1]).numpy())
                object_extend_dict = {'score': score}
                objects_dict = pascal_voc_rw_ex.get_objects_dict_template(name, xmin, ymin, xmax, ymax,
                                                                          object_extend_dict=object_extend_dict)
                objects_dict_list.append(objects_dict)
        return objects_dict_list