import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import utils
# import cv2
import os
from os.path import splitext, join, isfile, isdir, basename
import argparse
from glob import glob
from scipy import misc, ndimage
import collections
from pymongo import MongoClient
import pandas as pd
import math
import time
from object_detection.utils import label_map_util    ### CWH: Add object_detection path

###########object detection#####################

# Object detection imports

#nas

MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') ### CWH: Add object_detection path

NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    # print("yes")

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--glob_path', type=str, default=None,
                      help='Glob path for multiple images')
    parser.add_argument('-c', '--collection', type=str, default=None,
                      help='name of the mongodb collection')

    parser.add_argument('--id', default="0")

    args = parser.parse_args()

    try:
        client = MongoClient()
    except:
        print("Could not Connect to MongoDB")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    db = client.data_analysis_detection

    

    collection = db[args.collection] 

    # da_06_06_1


    images = glob(args.glob_path)


    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i+1,len(images)))
            img = misc.imread(img_path, mode='RGB')
            cimg = misc.imresize(img, 20)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(cimg, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})


            classes = np.squeeze(classes).astype(np.int32)
            # print("classes",classes)
            scores = np.squeeze(scores)
            # print("scores",scores)

            boxes = np.squeeze(boxes)
            # print("boxes",boxes)
            # numbers = num
            # print("numbers",numbers)

            threshold = 0.20  #CWH: set a minimum score threshold of 50%
            obj_above_thresh = sum(n > threshold for n in scores)
            # print("obj_above_thresh",obj_above_thresh)

            # print("detected %s objects in %s above a %s score" % ( obj_above_thresh, input_image, threshold))
            
            

            objects1 = []
            scores1 = []

            for c in range(0, len(classes)):
              # print("len(classes)",len(classes))
              # print("c",c)
              

              if scores[c] > threshold:

                class_name = category_index[classes[c]]['name']
                # print(" object %s is a %s - score: %s, location: %s" % 
                #     (c, class_name, scores[c], boxes[c]))
                  
                # print("class_name %s,scores[c] %s" % (class_name,scores[c]))
                objects1.append(class_name)
                scores1.append(str(scores[c]))

            # print(results)
            ## the mobilenet is giving 100 predictions for an image, for which four types of 
            ## arrays are given as outputs. classes have the number of categories of all
            ##100 such classes and respectively, all other have scores respective to the 
            ## numbers of the outputs in serial ordering. hence we can give a threshold to
            ## produce an output.

            count = collections.Counter(objects1)
            count = dict(count)

            total_obj = sum(count.values())
            
            # print("count",count)

            results = sorted([(key,value) for (key,value) in count.items()], reverse = True)

            print(results,total_obj)


            collection.insert_one({ "img_name":img_path.split("/")[-1],
                                    "detection":count,
                                    })






