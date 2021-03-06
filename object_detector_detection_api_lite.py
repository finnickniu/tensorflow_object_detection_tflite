import numpy as np
import tensorflow as tf
import cv2

from object_detector_detection_api import ObjectDetectorDetectionAPI, \
    PATH_TO_LABELS, NUM_CLASSES


class ObjectDetectorLite(ObjectDetectorDetectionAPI):
    def __init__(self, model_path='/home/apptech/models/research/object_detection/excavator_mobilenetssd/pre_trained_models/tflite/ssd.tflite'):
        """
            Builds Tensorflow graph, load model and labels
        """

        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.contrib.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, image, threshold=0.1):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """

        # Resize and normalize image for network input
        frame = cv2.resize(image, (320, 320))
        frame = np.expand_dims(frame, axis=0)
        # frame = (2.0 / 255.0) * frame - 1.0
        frame = frame.astype('uint8')

        # run model
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])

        # Find detected boxes coordinates
        return self._boxes_coordinates(image,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]+1).astype(np.uint8),
                            np.squeeze(scores[0]),
                            min_score_thresh=threshold)

    def close(self):
        pass


if __name__ == '__main__':
    detector = ObjectDetectorLite()
    cam=cv2.VideoCapture("/home/apptech/Downloads/WhatsApp Video 2020-05-13 at 12.56.16 PM.mp4")
    while True:
        _,image=cam.read()
        image=cv2.resize(image,(300,300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect(image, 0.2)
        print("result",result)


        for obj in result:
            print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
            format(obj[0], obj[1], obj[3], obj[2]))
            if obj[2]>1: 
                continue
            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                            (obj[0][0], obj[0][1] - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        show_image=cv2.resize(image,(1280,720))

        cv2.imshow('object detection', show_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
