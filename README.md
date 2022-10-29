# vaik-detection-inference

Inference with the PB model of the Tensorflow Object Detection API and output the result as a dict in extended Pascal VOC format.

## Example

![vaik-detection-pb-inference](https://user-images.githubusercontent.com/116471878/198853671-a868f67f-7105-4ea8-b10b-4362596728c9.png)

## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-detection-pb-inference.git
```

## Usage

### Example

```python
import os
import numpy as np
from PIL import Image

from vaik_detection_pb_inference.pb_model import PbModel

input_saved_model_dir_path = os.path.expanduser('~/.mnist_detection_model/saved_model')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-detection-dataset/valid/valid_000000000.jpg')).convert('RGB'))

model = PbModel(input_saved_model_dir_path, classes)
objects_dict_list, raw_pred = model.inference(image, resize_input_shape=(512, 512), score_th=0.2, nms_th=0.5)
```

#### Output

- objects_dict_list

```text
[
  {
    'name': 'eight',
    'pose': 'Unspecified',
    'truncated': 0,
    'difficult': 0,
    'bndbox': {
      'xmin': 564,
      'ymin': 100,
      'xmax': 611,
      'ymax': 185
    },
    'score': 0.9445509314537048
  },
  ・・・
  {
    'name': 'four',
    'pose': 'Unspecified',
    'truncated': 0,
    'difficult': 0,
    'bndbox': {
      'xmin': 40,
      'ymin': 376,
      'xmax': 86,
      'ymax': 438
    },
    'score': 0.38432005047798157
  }
]
```

- raw_pred
```
{'detection_boxes': <tf.Tensor: shape=(1, 100, 4), dtype=float32, numpy=
array([[[0.13946737, 0.7805822 , 0.25878623, 0.84517217],
        [0.15312591, 0.40262616, 0.25133595, 0.4656068 ],
        [0.49687555, 0.06259374, 0.58407366, 0.12873867],
        [0.5245404 , 0.05551525, 0.6107743 , 0.11932037],
        [0.15133946, 0.4047003 , 0.2533374 , 0.46617588],
        ・・・
        0.02872352, 0.02868354, 0.02850977, 0.02836256, 0.0283334 ,
        0.02831834, 0.02827285, 0.02827055, 0.02825131, 0.02809727,
        0.02780664, 0.02777159, 0.02760236, 0.02753625, 0.02753502]],
      dtype=float32)>}

```