# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import glob
import json
import numpy as np
import random
import cv2

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'image dir')
flags.DEFINE_string('ann_dir', '',
                    'annotation dir')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def dict_to_tf_example(image_path, annotation, label_map_dict):
  
  full_path = os.path.abspath(image_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  width, height = image.size
  # if image.format != 'PNG':
  #   raise ValueError('Image format not PNG')
  key = hashlib.sha256(encoded_png).hexdigest()
  filename = os.path.basename(image_path)
  print(filename)
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  difficult_obj.append(int(False))

  for char_obj in annotation['objects']:
    c = char_obj['classTitle']
    c_x1 = char_obj['points']['exterior'][0][0]
    c_y1 = char_obj['points']['exterior'][0][1]
    c_x2 = char_obj['points']['exterior'][1][0]
    c_y2 = char_obj['points']['exterior'][1][1]
    xmin.append(float(c_x1) / width)
    ymin.append(float(c_y1) / height)
    xmax.append(float(c_x2) / width)
    ymax.append(float(c_y2) / height)
    classes_text.append(c.encode('utf8'))
    classes.append(label_map_dict[c])
    truncated.append(0)
    poses.append('Unspecified'.encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  logging.basicConfig(level=logging.INFO)
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading plate dataset')
  
  start_indices = [0, 12500, 25000, 37500, 50000, 62500, 75000, 87500,100000,112500,125000,137500,150000,162500,175000,187500,200000,212500,225000,237500,250000,262500,275000,287500,300000,
  312500,325000,337500,350000,362500,375000,387500]
  start_indices = [0, 12500, 25000, 37500,50000]
  count = 0
  for start_idx in start_indices:
    print(f'index range: {start_idx} to {start_idx+12500}')
    for idx in range(start_idx, start_idx + 12500):
      if count % 100 == 0:
        logging.info('On image %d of %d', count, 400000)
      count += 1
      image_path = os.path.join(FLAGS.image_dir, f'{idx}.jpg')
      ann_path = os.path.join(FLAGS.ann_dir,f'{idx}.jpg.json')
      if not os.path.exists(ann_path):
        logging.info(f'{idx}.json doesn\'t exist, skip')
        continue
      if not os.path.exists(image_path):
        logging.info(f'{idx}.jpg doesn\'t exist, skip')
        continue
      with open(ann_path) as json_f:
        annotation = json.load(json_f)

      tf_example = dict_to_tf_example(image_path, annotation, label_map_dict)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
