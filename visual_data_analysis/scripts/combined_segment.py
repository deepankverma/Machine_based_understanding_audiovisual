# IMPORTS

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
# from PIL import Image
# import torch
# from torch.autograd import Variable as V
# import torchvision.models as models
# from torchvision import transforms as trn
# from torch.nn import functional as F
from pspnet import *
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import utils
# import cv2
import collections
from pymongo import MongoClient
import datetime
from xml_parse import *
# from time import sleep, perf_counter as pc
import pandas as pd
import math

# colors_file = pd.read_csv("color_list_reduced.csv", header = None)

###########object detection#####################

# Object detection imports

# from object_detection.utils import label_map_util    ### CWH: Add object_detection path

# #nas

# MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') ### CWH: Add object_detection path

# NUM_CLASSES = 90

# # Load a (frozen) Tensorflow model into memory.
# detection_graph = tf.Graph()

# with detection_graph.as_default():
#   od_graph_def = tf.GraphDef()
#   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
#     # print("yes")

# # Loading label map
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)



# # Helper code
# # def load_image_into_numpy_array(image):
# #   # print("yes1")
# #   (im_width, im_height) = image.size
# #   return np.array(image.getdata()).reshape(
# #       (im_height, im_width, 3)).astype(np.uint8)



# # print(type(image))


# #####CLASSIFICATION
# # th architecture to use
# arch = 'resnet18'
# # load the pre-trained weights
# model_file = 'whole_%s_places365_python36.pth.tar' % arch

# useGPU = 1
# if useGPU == 1:
#   model = torch.load(model_file)
# else:
#   model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!



# def img_detection(input_image):



#   with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#       # Definite input and output Tensors for detection_graph
#       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#       # Each box represents a part of the image where a particular object was detected.
#       detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#       # Each score represent how level of confidence for each of the objects.
#       # Score is shown on the result image, together with the class label.
#       detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#       detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#       num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      
#       image = misc.imread(input_image)
#       image = misc.imresize(image,20)
#       # print(image.shape)
#       # the array based representation of the image will be used later in order to prepare the
#       # result image with boxes and labels on it.
#       # image_np = load_image_into_numpy_array(image)
#       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#       image_np_expanded = np.expand_dims(image, axis=0)
#       # Actual detection.
#       (boxes, scores, classes, num) = sess.run(
#           [detection_boxes, detection_scores, detection_classes, num_detections],
#           feed_dict={image_tensor: image_np_expanded})


#       classes = np.squeeze(classes).astype(np.int32)
#       # print("classes",classes)
#       scores = np.squeeze(scores)
#       # print("scores",scores)

#       boxes = np.squeeze(boxes)
#       # print("boxes",boxes)
#       # numbers = num
#       # print("numbers",numbers)

#       threshold = 0.20  #CWH: set a minimum score threshold of 50%
#       obj_above_thresh = sum(n > threshold for n in scores)
#       print("obj_above_thresh",obj_above_thresh)

#       print("detected %s objects in %s above a %s score" % ( obj_above_thresh, input_image, threshold))
      
      

#       objects1 = []
#       scores1 = []

#       for c in range(0, len(classes)):
#         # print("len(classes)",len(classes))
#         # print("c",c)
        

#         if scores[c] > threshold:

#           class_name = category_index[classes[c]]['name']
#           # print(" object %s is a %s - score: %s, location: %s" % 
#           #     (c, class_name, scores[c], boxes[c]))
            
#           # print("class_name %s,scores[c] %s" % (class_name,scores[c]))
#           objects1.append(class_name)
#           # print(objects1)
#           scores1.append(str(scores[c]))


#       # print(results)
#       ## the mobilenet is giving 100 predictions for an image, for which four types of 
#       ## arrays are given as outputs. classes have the number of categories of all
#       ##100 such classes and respectively, all other have scores respective to the 
#       ## numbers of the outputs in serial ordering. hence we can give a threshold to
#       ## produce an output.

#   count = collections.Counter(objects1)
#   count = dict(count)

#   total_obj = sum(count.values())
  
#   # print("count",count)

#   results = sorted([(key,value) for (key,value) in count.items()], reverse = True)



#   return results, total_obj

# ##################places 365 with torch


# def places365(image):


#   model.eval()

#   # load the image transformer
#   centre_crop = trn.Compose([
#         trn.Resize((256,256)),
#         trn.CenterCrop(224),
#         trn.ToTensor(),
#         trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#   ])

#   # load the class label
#   file_name = 'categories_places365.txt'
#   classes = list()
#   with open(file_name) as class_file:
#     for line in class_file:
#         classes.append(line.strip().split(' ')[0][3:])
#   classes = tuple(classes)

#   # load the test image
#   # img_name = '12.jpg'

#   img = Image.open(image)
#   input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

#   # forward pass
#   logit = model.forward(input_img)
#   h_x = F.softmax(logit, 1).data.squeeze()
#   probs, idx = h_x.sort(0, True)

#   # print('RESULT ON ')
#   # output the prediction

#   classes1 = []
#   probability = []

#   for i in range(0, 5):

#     classes1.append(classes[idx[i]])
#     probability.append(probs[i])


      
#       # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
#   results = dict(zip(classes1,probability))

#   results = sorted([(value,key) for (key,value) in results.items()], reverse = True)

#   return results


# ###################### image segmentation

def segment(image):

  # img_path = 'object_detection/test_images/image1.jpg'
  # output_path = "results_trial/cityscapes"
  # id = ""
  model1 = "pspnet50_ade20k"
  multi_scale = 0
  sliding = 0
  flip = 0


  sess = tf.Session()
  K.set_session(sess)

  with sess.as_default():
      # print(args)

      # if "pspnet50" in args.model:
      pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                            weights=model1)
      # elif "pspnet101" in args.model:
      #     if "cityscapes" in args.model:
      #         pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
      #                            weights=args.model)
      #     if "voc2012" in args.model:
      #         pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
      #                            weights=args.model)

      # else:
      #     print("Network architecture not implemented.")

      # if multi_scale == 1:
      # EVALUATION_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # must be all floats!
      # EVALUATION_SCALES = [0.15, 0.25, 0.5]  # must be all floats!

      
      # for i, img_path in enumerate(images):
      # print("Processing image {} / {}".format(i+1,len(images)))
      img = misc.imread(image)
      img = misc.imresize(img,size = 20)
      class_scores = predict_multi_scale(img, pspnet, EVALUATION_SCALES, sliding, flip)
      # print("class_scores",class_scores)

      # print("Combine predictions...")
      class_image = np.argmax(class_scores, axis=2)
      ##class_image returns the lsit of pixel wise classes

      # indexes = np.unique(class_image, return_index=True)[1]
      # c1 = [class_image[index] for index in sorted(indexes)]

      # print("class_image", c1)
      # print("class_image", class_image)

      # pm = np.max(class_scores, axis=2)
      # print("pm", pm)
      colored_class_image, color_names = utils.color_class_image(class_image, model1)
      # print("colored_class_image", colored_class_image.shape)
      

      # indexes = np.unique(color_names, return_index=True)[1]
      # c1 = [color_names[index] for index in sorted(indexes)]
      # print("color_names", color_names)

      count = collections.Counter(color_names)
      count = dict(count)
      

      y = sum(count.values())

      for key, value in count.items():

        # print(value, sum(count.values()))

        count[key] = value/y

      ##to sort in descending order according to the value
      count = sorted([(value,key) for (key,value) in count.items()],reverse = True)


      # colored_class_image is [0.0-1.0] img is [0-255]
      # alpha_blended = 0.5 * colored_class_image + 0.5 * img

      # print("Write result...")
      
      filename, ext = splitext(image)

      ext = ".png"
      image = image.split("/")
      image = image[-1]

      misc.imsave(filename + "_seg" + ext, colored_class_image)
      # print("saved")
      # misc.imsave(img_path + "_probs" + ext, pm)
      # misc.imsave(img_path + "_seg_blended" + ext, alpha_blended)

  return count


# def dominant_colors(image):


#   clusters = 5

# #### will have to resize the image for quck results. It takes a long amount of time

#   t0 = pc()

#   img = misc.imread(image)
#   img = misc.imresize(img, 20)

#   # reshape the image to be a list of pixels
#   image = img.reshape((img.shape[0] * img.shape[1], 3))

#   # cluster the pixel intensities
#   clt = KMeans(n_clusters = clusters)
#   clt.fit(image)

#   print(pc()-t0)

   
#   def centroid_histogram(clt):
#     # grab the number of different clusters and create a histogram
#     # based on the number of pixels assigned to each cluster
#     numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
#     # print(numLabels)
#     (hist, _) = np.histogram(clt.labels_, bins = numLabels)
   
#     # normalize the histogram, such that it sums to one
#     hist = hist.astype("float")
#     hist /= hist.sum()
   
#     # return the histogram
#     return hist

#   def plot_colors(hist, centroids):
#     # initialize the bar chart representing the relative frequency
#     # of each of the colors

#     dom_colors = []
#     percentage = []
#     names = []
   
#     # loop over the percentage of each cluster and the color of
#     # each cluster

#     ##making odds even to match with the color summarizer
#     def odd_even(n):
#       if n % 2 != 0:
#         return n+1
#       else:
#         return n 


#     for (percent, color) in zip(hist, centroids):
#       # plot the relative percentage of each cluster
#       # print(percent)
#       percentage.append(percent)
#       # print(type(color))
#       color1 = list(color)
#       hexa = "#{:02x}{:02x}{:02x}".format(odd_even(int(color1[0])),
#         odd_even(int(color1[1])),odd_even(int(color1[2])))
#       # print(hexa.strip("#").upper())
#       hexa = hexa.strip("#").upper()


#       for hex1,i in zip(colors_file[0],range(len(colors_file))):

#         if hexa.strip("#").upper() == str(hex1):
#           # print(colors_file.loc[i][1])
#           # print(i)

#           z = colors_file.loc[i][1]
#           # print(z)
#           z = z.split(":")
#           # print(z)

#           color_list = []
#           for each in z:
#             a = each.split("[")

#             # print(a)
#             a = a[0]
#             # print(a)
#             color_list.append(a)


#       names.append(color_list) 

#       # print(names)
#       dom_colors.append(hexa)


#     return dom_colors, percentage, names

#   # build a histogram of clusters and then create a figure
#   # representing the number of pixels labeled to each color
#   hist = centroid_histogram(clt)
#   dom_colors, percentage, names = plot_colors(hist, clt.cluster_centers_)

#   clusters1 = dict(zip(percentage,names))

#   clusters1 = sorted([(key,value) for (key,value) in clusters1.items()], reverse = True)

#   clusters = dict(zip(dom_colors,percentage))
#   clusters = sorted([(value,key) for (key,value) in clusters.items()], reverse = True)


#   return clusters, clusters1
   
# def simpson(lists):
#   sum1 = 0
#   sum2 = 0

#   for each in lists:
#     sum1 = sum1 +  (each[1] * (each[1] -1))
#     sum2 = sum2 + each[1]
      
#   if sum2 * (sum2-1) == 0:
#     diverse = 1
#   else:
#     diverse = sum1/(sum2 * (sum2-1))
          
#   return (1 - diverse)



# def entropy(lists):
#   sum1 = 0
#   sum2 = 0
#   l = len(lists)
#   for each in lists:
#     sum1 = (sum1 + each[0]*math.log(each[0])) #entropy
#     sum2 = sum2 + math.pow((100*each[0]),2)  ##hhi 

  
#   if l > 1:
#     ent = sum1/math.log(l)  ##to prevent it from dividing with zero
#   else:
#     ent = sum1/1

#   hhi = sum2/10000

#   return -ent,hhi




def comb(gpx,images):

  

  # t0 = pc()

  # u,u1 = img_detection(images)
  # print(u,u1)
  # # print(type(u))
  # # print(type(u[0]),u[1][1])


  # v = places365(images)
  # print(v)
  
  # divers = entropy(v)
  # print(divers)

  w = segment(images)
  print(w)

  # x,y = dominant_colors(images)
  # print(x,y)
  # print(pc()-t0)

  # print("gpx",gpx,"u",u,"v",v,"w",w,"x",x,"y",y)

  images = images.split("/")
  images = images[-1]

  # d = datetime.datetime.strptime(gpx[-1], "%Y-%m-%dT%H:%M:%S.%fZ")
  # print(list(gpx[1].values()))

# { lng : 55.5 , lat : 42.3 }
  # loc : { type: "Point", coordinates: [ -76.703347, 30.710459 ] },

  collection.insert_one({       "img_name":images,
                                "heading":gpx[0],
                                "loc":gpx[1],
                                # "date":d,
                                # "loc": { "lng": float(list(gpx[1].values())[0]), "lat": float(list(gpx[1].values())[1]) }
                                # "detection":u,
                                # "places":v,
                                "segmentation":w,
                                # "colors":x,
                                # "colors_names":y
                                })





try:
  client = MongoClient()
except:
  print("Could not Connect to MongoDB")


db = client.data_analysis

collection = db.fifth_june_1


image_dir = "/media/deepank/086C617F6C616880/extra_space/data_collection/mapillary/05_6-10-11/2018_06_05_10_02_13_260_+0530/"

xml_list = xml_to_list(image_dir)

idx = reducing_xml(xml_list,image_dir)

print("idx=================", idx)


def new_xml_list(xml_list,idx):

  new_xml_list = []
  for each in idx:
    new_xml_list.append(xml_list[each])

  return new_xml_list

y = new_xml_list(xml_list,idx)

print(len(y))


file_names = sorted(glob.glob(image_dir + '/*.jpg'))
final_xml_list = y


####to slice the list

file_names = file_names[:]
final_xml_list =final_xml_list[:]


for each, i in zip(file_names,range(len(final_xml_list))):
  comb(final_xml_list[i],each)

















