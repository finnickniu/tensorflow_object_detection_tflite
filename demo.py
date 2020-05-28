import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import skimage
import json
import datetime
import time
import time
import argparse
def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.

  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.

  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)

def main(args):

  # PATH_TO_FROZEN_GRAPH = '/home/apptech/Downloads/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
  PATH_TO_FROZEN_GRAPH=args.PATH_TO_FROZEN_GRAPH

  
  # f = open("/home/apptech/apptech_tf_models/label.json")
  f = open(args.js_file)

  labels=json.loads(f.read())
  # List of the strings that is used to add correct label for each box.
  # labels={1:'heavy_machine'}

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      # cam=cv2.VideoCapture(0)
      cam=cv2.VideoCapture(args.cam_dir)

      # height, width = int(cam.get(3)),int(cam.get(4))
      # fps = cam.get(cv2.CAP_PROP_FPS)
      # fourcc = cv2.VideoWriter_fourcc(*'XVID')
      # out = cv2.VideoWriter(video_save_path, fourcc, fps, 920, 1080))
      fps_time = 0
      cnt=0
      while True:
          success,image = cam.read()
          if not success:
            break
          # image=cv2.resize(image,(1920,1080),interpolation=cv2.INTER_AREA)
          image_h = image.shape[0]
          image_w = image.shape[1]
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
              output_dict['detection_masks'] = output_dict['detection_masks'][0]
            
          image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          for i, score in enumerate(output_dict['detection_scores']):
            if score >0.8:
              classes=labels[str(output_dict['detection_classes'][i])]
              print(classes)
              ymin, xmin, ymax, xmax = tuple(output_dict['detection_boxes'][i].tolist())
              # print(ymin, xmin, ymax, xmax)
              ymin = int(ymin * image_h)
              xmin = int(xmin * image_w)
              ymax = int(ymax * image_h)
              xmax = int(xmax * image_w)
              cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
              # cv2.putText(image, classes, (xmin,ymax+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

          cnt+=1
          # out.write(frame)
          cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          fps_time = time.time()
          show_image = cv2.resize(image, (1920,1080))
          cv2.imshow('win', show_image)
          pressed_key = cv2.waitKey(2)
          if pressed_key == ord('q'):
              break
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('PATH_TO_FROZEN_GRAPH', type=str, help='PATH_TO_FROZEN_GRAPH')
  parser.add_argument('cam_dir', default= 0, help='path to videos/cams')
  parser.add_argument('js_file', type=str, help='path to label.json')
  args = parser.parse_args()
  main(args)


      
