# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import statistics

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #print(boxes, scores, classes, num)
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

import cv2
import subprocess
import json
import math
import statistics
model_path = r'C:\Users\mluser\Documents\Niokr_detection\person_detection_test\faster_rcnn_inception_v2_coco_2018_01_28\frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.6
threshold_siz = 0.9
pm = 0.2
cap = cv2.VideoCapture(r'C:\Users\mluser\Documents\Niokr_detection\person_detection_test\20181120_144115.mp4')

j = 0
r = True
fps = 29.0
#--Параметры сглаживания
countp = 0
while r is True and j<= fps*44:
        r, img = cap.read()

        if j == 0:
                writer = cv2.VideoWriter(
                'vp_out2.avi',
                cv2.VideoWriter_fourcc(*'MJPG'),  # codec
                fps,  # fps
                (img.shape[1], img.shape[0]),  # width, height
                isColor=len(img.shape) > 2)

        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        print ('j = ', j)
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cp = (box[1] + round((box[3] - box[1])/2.0), box[0] + round((box[2] - box[0])/2.0) , round((box[3] - box[1])/2.0),  round((box[2] - box[0])/2.0))
                xmin = box[1]
                xmax = box[3]
                ymin = box[0]
                ymax = box[2]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                if (ymin - 10 > 0):
                          cv2.putText(img,'person:' + str(round(scores[i], 2)), (xmin, ymin - 10),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

                countp = countp + 1
                print('pm=',(ymax-ymin)*(xmax-xmin), img.shape[1]*img.shape[0])
                      # Записываем выделенных людей
                if (((ymax-ymin)*(xmax-xmin)) >= pm * img.shape[1]*img.shape[0] ):
                        crim = img[ymin:ymax, xmin:xmax]
                        cv2.imwrite(r'C:\Users\mluser\Documents\Niokr_detection\pictpers\frame%d.jpg' % countp, crim)
                        # Распознаем и прорисоываем каски
                        commandhelm = r'curl https://172.19.12.53/powerai-vision/api/dlapis/0513f2f9-cdfd-4566-96e4-5b3ac43e338d -k -F files=@C:\Users\mluser\Documents\Niokr_detection\pictpers\frame' + str(countp) + '.jpg'
                        phelm = subprocess.Popen(commandhelm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        outhelm, err = phelm.communicate()
                        print('frame_def ' + str(countp), outhelm)
                        try:
                          jouthelm = json.loads(outhelm)
                          for y in jouthelm['classified']:
                              x1 = y['xmin'] + xmin
                              x2 = y['xmax'] + xmin
                              y1 = y['ymin'] + ymin
                              y2 = y['ymax'] + ymin
                              tag_name = y['label'].replace('_','').replace('true','').replace('false','').replace('red','').replace('blue','').replace('green','').replace('black','').replace('not-','not_').replace('-','')
                              print(y['label'], x1,y1,x2,y2,y['confidence'])
                              if y['confidence'] >= threshold_siz:
                                if 'not' in y['label']:
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    if (y1 - 10 > 0):
                                        cv2.putText(img, tag_name + ':' + str(round(y['confidence'], 2)), (x1, y1 - 10),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                                else:
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 0), 2)
                                    if (y1 - 10 > 0):
                                        cv2.putText(img, tag_name + ':' + str(round(y['confidence'], 2)), (x1, y1 - 10),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 100, 0), 2)
                        except:
                          print('attribute no answer',' outhelm', outhelm)
        writer.write(img)
        j = j + 1
writer.release()
#cv2.destroyAllWindows()