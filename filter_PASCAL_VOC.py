''' This script reads the annotation txt file containing the bounding box and size information of each image in PASCAL VOC 2007
    and finds images for each class where no ground truth boxes of that class overlap with our patch location.
'''

from __future__ import print_function
import sys
import numpy as np
import math
import os
from tqdm import tqdm
if len(sys.argv) != 3:
    print('Usage:')
    print('python filter_images_get_bbox.py annotations.txt test.txt')
    exit()
# /nfs1/code/aniruddha/detection-patch/VOCdevkit/VOC2007/ImageSets/Main/test.txt

'''
    Input: annotation txt file
    Output: create a boolean matrix of size num_images * num_classes where entry 1 denotes the image has no
            class bbox overlap with patch location
'''


# Patch parameters
patchSize = 100
start_x = 5
start_y = 5


# create the numpy array
num_images = 9963               # PASCAL VOC annotation file - all images but consider only test by intersection
num_classes = 20
filter_matrix = np.zeros((num_images, num_classes), dtype=bool)
filter_matrix_all_test = np.zeros((num_images, num_classes), dtype=bool)
flag = np.zeros((num_images, num_classes), dtype=bool)
test_image_vector = np.zeros(num_images, dtype=bool)

# create a dictionary to index into the numpy array
classes = {"aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, "cat":7, "chair":8, "cow":9, "diningtable":10, \
        "dog":11, "horse":12, "motorbike":13, "person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19}

annotation_file = sys.argv[1]
test_file = sys.argv[2]


# parse the test file to find out which images are in the test set
with open(test_file,'r') as f2:
    lines_test = f2.readlines()

lines_test = [line.rstrip('\n') for line in lines_test]

for x in lines_test:
    test_image_idx = int(x)
    test_image_vector[test_image_idx-1] = True

with open(annotation_file,'r') as f1:
    lines = f1.readlines()

lines = [line.rstrip('\n') for line in lines]
#print(lines)

for line_idx, x in tqdm(enumerate(lines)):
    print(line_idx+1)
    split = x.split()
    # images are indexed from 000001 to 009963 in the annotations
    image_idx = int(split[0])
    if test_image_vector[image_idx-1] == False:             # image not in test set
        # filter_matrix[image_idx-1,:] = False
        # filter_matrix_all_test[image_idx-1,:] = False
        continue
    gt_class = split[1]

    filter_matrix_all_test[image_idx-1,[classes[gt_class]]] = True

    if flag[image_idx-1,[classes[gt_class]]] == False:         # overlap for this class not yet found
        filter_matrix[image_idx-1,[classes[gt_class]]] = True

        # calculating bounding boxes after resizing to YOLO 416x416.
        # The top-left pixel in the image has coordinates (1, 1) in VOC data
        xmin = float(split[2])
        ymin = float(split[3])
        xmax = float(split[4])
        ymax = float(split[5])
        width = float(split[6])
        height = float(split[7])

        xmin_new = math.floor(xmin/width*416)
        ymin_new = math.floor(ymin/height*416)
        xmax_new = math.ceil(xmax/width*416)
        ymax_new = math.ceil(ymax/height*416)


        # find intersection of bounding box with patch
        A = min(xmax_new, start_x+patchSize)
        B = max(xmin_new, start_x)
        C = min(ymax_new, start_y+patchSize)
        D = max(ymin_new, start_y)

        # if overlap unset the corresponding cell in the matrix
        if A>=B and C>=D:
            flag[image_idx-1][classes[gt_class]] = True
            filter_matrix[image_idx-1,[classes[gt_class]]] = False

    else:
        continue

# calculate number of filtered images
print("Filtered images:")
for key, value in sorted(classes.items()):
    print(key + ":" + str(sum(filter_matrix[:,value])) + "/" + str(sum(filter_matrix_all_test[:,value])))


''' create the training files for each class
'''

# open files for each class
fid_list = []
dataset_folder_prefix = '/nfs3/code/aniruddha/UMBCvision/Contextual-Adversarial-Patches/dataset/no_class_overlap_clean_test/'
jpeg_prefix = '/nfs1/code/aniruddha/detection-patch/VOCdevkit/VOC2007/JPEGImages/'

if not os.path.exists(dataset_folder_prefix):
    os.makedirs(dataset_folder_prefix)

for key, value in sorted(classes.items()):
    f = open(dataset_folder_prefix + key + '_test.txt','w')
    fid_list.append(f)
    print(dataset_folder_prefix + key + '_test.txt')

for i in range(num_images):
    for j in range(num_classes):
        if filter_matrix[i][j] == 1:
            fid_list[j].write(jpeg_prefix + str(i+1).zfill(6) + '.jpg' + '\n')

for j in range(num_classes):
    fid_list[j].close()

dataset_folder_prefix = '/nfs3/code/aniruddha/UMBCvision/Contextual-Adversarial-Patches/dataset/no_class_overlap_clean_test/'
# jpeg_prefix = '/nfs1/code/aniruddha/detection-patch/VOCdevkit/VOC2007/JPEGImages/'

fwr = open("/nfs3/code/aniruddha/UMBCvision/Contextual-Adversarial-Patches/dataset/filtered_VOC_test.txt", "w")

for i in range(num_images):
    if sum(filter_matrix[i][:]) > 0:
        fwr.write(jpeg_prefix + str(i+1).zfill(6) + '.jpg' + '\n')

fwr.close()
