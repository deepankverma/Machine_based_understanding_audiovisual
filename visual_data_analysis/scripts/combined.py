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
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from pspnet import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import cv2
import collections
from pymongo import MongoClient
import datetime
from xml_parse import *
from time import sleep, perf_counter as pc
import pandas as pd
import math
import time


colors_file = pd.read_csv("color_list_reduced.csv", header = None)

###########object detection#####################

# Object detection imports

from object_detection.utils import label_map_util    ### CWH: Add object_detection path

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



# Helper code
# def load_image_into_numpy_array(image):
#   # print("yes1")
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)



# print(type(image))


#####CLASSIFICATION
# th architecture to use
arch = 'resnet18'
# load the pre-trained weights
model_file = 'whole_%s_places365_python36.pth.tar' % arch

useGPU = 1
if useGPU == 1:
  model = torch.load(model_file)
else:
  model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!



def img_detection(input_image):



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

      
      image = misc.imread(input_image)
      image = misc.imresize(image,20)
      # print(image.shape)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      # image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image, axis=0)
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
      print("obj_above_thresh",obj_above_thresh)

      print("detected %s objects in %s above a %s score" % ( obj_above_thresh, input_image, threshold))
      
      

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



  return results, total_obj

##################places 365 with torch


def places365(image):


  model.eval()

  # load the image transformer
  centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # load the class label
  file_name = 'categories_places365.txt'
  classes = list()
  with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
  classes = tuple(classes)

  # load the test image
  # img_name = '12.jpg'

  img = Image.open(image)
  input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

  # forward pass
  logit = model.forward(input_img)
  h_x = F.softmax(logit, 1).data.squeeze()
  probs, idx = h_x.sort(0, True)

  # print('RESULT ON ')
  # output the prediction

  classes1 = []
  probability = []

  for i in range(0, 5):

    classes1.append(classes[idx[i]])
    probability.append(probs[i])


      
      # print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
  results = dict(zip(classes1,probability))

  results = sorted([(value,key) for (key,value) in results.items()], reverse = True)

  return results


###################### image segmentation

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


def dominant_colors(image):


  clusters = 5

#### will have to resize the image for quck results. It takes a long amount of time

  t0 = pc()

  img = misc.imread(image)
  img = misc.imresize(img, 20)

  # reshape the image to be a list of pixels
  image = img.reshape((img.shape[0] * img.shape[1], 3))

  # cluster the pixel intensities
  clt = KMeans(n_clusters = clusters)
  clt.fit(image)

  print(pc()-t0)

   
  def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    # print(numLabels)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
   
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
   
    # return the histogram
    return hist

  def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors

    dom_colors = []
    percentage = []
    names = []
   
    # loop over the percentage of each cluster and the color of
    # each cluster

    ##making odds even to match with the color summarizer
    def odd_even(n):
      if n % 2 != 0:
        return n+1
      else:
        return n 


    for (percent, color) in zip(hist, centroids):
      # plot the relative percentage of each cluster
      # print(percent)
      percentage.append(percent)
      # print(type(color))
      color1 = list(color)
      hexa = "#{:02x}{:02x}{:02x}".format(odd_even(int(color1[0])),
        odd_even(int(color1[1])),odd_even(int(color1[2])))
      # print(hexa.strip("#").upper())
      hexa = hexa.strip("#").upper()


      for hex1,i in zip(colors_file[0],range(len(colors_file))):

        if hexa.strip("#").upper() == str(hex1):
          # print(colors_file.loc[i][1])
          # print(i)

          z = colors_file.loc[i][1]
          # print(z)
          z = z.split(":")
          # print(z)

          color_list = []
          for each in z:
            a = each.split("[")

            # print(a)
            a = a[0]
            # print(a)
            color_list.append(a)


      names.append(color_list) 

      # print(names)
      dom_colors.append(hexa)


    return dom_colors, percentage, names

  # build a histogram of clusters and then create a figure
  # representing the number of pixels labeled to each color
  hist = centroid_histogram(clt)
  dom_colors, percentage, names = plot_colors(hist, clt.cluster_centers_)

  clusters1 = dict(zip(percentage,names))

  clusters1 = sorted([(key,value) for (key,value) in clusters1.items()], reverse = True)

  clusters = dict(zip(dom_colors,percentage))
  clusters = sorted([(value,key) for (key,value) in clusters.items()], reverse = True)


  return clusters, clusters1


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

def hillnumbers(lists):
    
  richness = 0
  sum_ent = 0
  sum_simpson = 0

  ## converting the tuple to make them simiilar
  if len(lists)>1:
    if type(lists[0][1]) == str:
        lists = [(t[1], t[0]) for t in lists]

    for each in lists:
        
      if each[1] > 0:
        z_sum  = sum(i[1] for i in lists)
        p_i = each[1]/z_sum
        p_i_squared = p_i * p_i
        sum_simpson += p_i_squared
        richness += 1
        p_i_log = p_i * np.log(p_i)
        sum_ent += p_i_log
        simpson = 1.0/(float(sum_simpson))
        shannon = np.exp(-sum_ent)
  else:
    richness = 0
    shannon = 0
    simpson = 0

  return richness,shannon,simpson


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


def nature(lists):
  natural = ['sky','tree','grass','earth','mountain','plant','water','sea','sand','river'
        'flower','dirt track','land','waterfall','animal','lake'] 
  sky = ['sky']
  green = ['tree','grass','plant','land']
  # "person" is not included.
  natureperc = 0
  skyperc = 0
  greenperc = 0
  for each in lists:
    if each[1] in natural:
      natureperc+= each[0]
    if each[1] in sky:
      skyperc+= each[0]
    if each[1] in green:
      greenperc+= each[0]
  return natureperc, skyperc, greenperc
    
   

def comb(gpx,images):

  t0 = pc()

  u,u4 = img_detection(images)
  print(u,u4)

  u1, u2, u3 = hillnumbers(u)

  print(u1,u2,u3)


  v = places365(images)

  w = segment(images)
  w1,w2,w3 = hillnumbers(w)
  w4,w5,w6 = nature(w)
  
  print(w1,w2,w3)
  print(w4,w5,w6)



  x,y = dominant_colors(images)
  print(pc()-t0)

  # # print("gpx",gpx,"u",u,"v",v,"w",w,"x",x,"y",y)

  images = images.split("/")
  images = images[-1]

  # d = datetime.datetime.strptime(gpx[-1], "%Y-%m-%dT%H:%M:%S.%fZ")

  d = gpx[-1]

  # t = datetime.datetime(2018, 3, 2, 17, 22, 4, 147000)
  #d1 = d.strftime('%Y-%m-%d %H:%M:%S')
  #d2 = time.strftime("%Y-%m-%d %H:%M:%S", 
              #time.gmtime(time.mktime(time.strptime(d1, 
                                                    #"%Y-%m-%d %H:%M:%S"))))

  d3 = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S.%fZ")

  collection.insert_one({       "img_name":images,
                                "heading":gpx[0],
                                "loc":gpx[1],
                                "date":d3,
                                # "loc": { "lng": float(list(gpx[1].values())[0]), "lat": float(list(gpx[1].values())[1]) }
                                "natural_perc":w4,
                                "sky_perc":w5,
                                "green_perc":w6,
                                "richness_d":u1,
                                "shannon_d":u2,
                                "simpson_d":u3,
                                "richness_s":w1,
                                "shannon_s":w2,
                                "simpson_s":w3,
                                "total_detection":u4,
                                "detection":u,
                                "places":v,
                                "segmentation":w,
                                "colors":x,
                                "colors_names":y
                                })


try:
  client = MongoClient()
except:
  print("Could not Connect to MongoDB")


db = client.street_view

collection = db.d_1_w_1_1



image_dir = "/mnt/disks/disk-2/streetview/d_1/w_1/1"

xml_list = xml_to_list(image_dir)
##manually updating the list as it is returning bad list
idx = [1, 3, 5, 6, 8, 10, 13, 15, 17, 18, 20, 22, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 53, 55, 56, 58, 59, 61, 63, 64, 66, 68, 69, 70, 72, 73, 76, 77, 78, 80, 81, 83, 84, 85, 87, 88, 90, 91, 93, 95, 97, 98, 99, 101, 102, 104, 106, 107, 109, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 127, 129, 130, 132, 133, 135, 136, 138, 139, 140, 142, 143, 144, 146, 147, 149, 150, 151, 153, 154, 156, 157, 159, 161, 162, 164, 166, 167, 168, 171, 172, 174, 175, 177, 178, 179, 181, 182, 184, 185, 186, 188, 189, 191, 193, 194, 196, 197, 198, 199, 201, 203, 204, 206, 208, 209, 210, 212, 213, 215, 216, 218, 219, 221, 222, 224, 225, 227, 228, 229, 230, 232, 233, 234, 236, 237, 239, 240, 242, 243, 245, 246, 247, 249, 250, 251, 253, 254, 256, 257, 258, 260, 262, 263, 264, 266, 268, 269, 271, 272, 274, 275, 277, 279, 281, 282, 283, 285, 286, 288, 289, 290, 292, 294, 295, 297, 298, 299, 300, 302, 303, 305, 306, 308, 309, 311, 312, 313, 315, 316, 318, 319, 321, 322, 324, 325, 327, 329, 330, 331, 333, 334, 336, 337, 339, 340, 342, 343, 345, 346, 348, 349, 350, 352, 353, 355, 356, 358, 360, 361, 362, 364, 366, 367, 368, 370, 372, 373, 375, 377, 379, 380, 382, 384, 385, 387, 389, 390, 392, 393, 395, 397, 398, 399, 401, 404, 405, 408, 410, 412, 413, 414, 416, 417, 419, 420, 422, 423, 424, 426, 427, 429, 430, 432, 433, 434, 436, 437, 439, 440, 441, 443, 444, 446, 447, 448, 450, 451, 452, 454, 455, 457, 458, 459, 461, 462, 464, 465, 466, 468, 469, 471, 473, 474, 476, 477, 479, 480, 482, 484, 485, 487, 488, 490, 491, 493, 494, 496, 497, 499, 500, 502, 503, 505, 507, 508, 510, 512, 513, 515, 516, 518, 519, 521, 522, 524, 526, 528, 529, 530, 532, 533, 535, 537, 538, 539, 541, 542, 544, 545, 547, 548, 549, 551, 552, 554, 555, 557, 558, 559, 560, 562, 563, 565, 566, 567, 569, 570, 572, 573, 576, 577, 578, 580, 581, 583, 584, 585, 587, 588, 589, 591, 593, 594, 596, 597, 599, 600, 601, 603, 605, 606, 608, 609, 611, 612, 614, 615, 616, 618, 620, 621, 623, 624, 626, 627, 629, 630, 632, 633, 635, 637, 639, 640, 642, 643, 645, 646, 648, 649, 651, 653, 654, 655, 656, 658, 660, 662, 663, 665, 666, 667, 668, 670, 672, 674, 675, 677, 678, 680, 681, 682, 684, 685, 687, 689, 690, 692, 693, 695, 696, 698, 699, 701, 702, 704, 705, 707, 708, 710, 712, 713, 715, 716, 718, 719, 720, 722, 724, 725, 727, 729, 731, 732, 734, 736, 738, 739, 740, 742, 744, 745, 746, 749, 751, 752, 754, 756, 757, 758, 760, 761, 763, 764, 766, 767, 770, 772, 773, 775, 776, 778, 779, 781, 782, 784, 785, 787, 789, 791, 792, 794, 795, 796, 798, 800, 801, 803, 804, 805, 807, 809, 810, 812, 814, 815, 817, 818, 820, 822, 824, 826, 828, 829, 831, 832, 834, 835, 837, 838, 839, 841, 843, 844, 845, 847, 848, 850, 851, 853, 854, 856, 857, 859, 860, 862, 863, 865, 867, 868, 870, 873, 874, 876, 878, 879, 881, 882, 884, 885, 887, 888, 890, 891, 893, 894, 895, 897, 898, 900, 901, 902, 904, 905, 907, 909, 910, 911, 913, 915, 916, 918, 920]




#idx = reducing_xml(xml_list,image_dir)

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

file_names = file_names[246:325]
final_xml_list =final_xml_list[246:325]

for each, i in zip(file_names,range(len(final_xml_list))):
  comb(final_xml_list[i],each)














