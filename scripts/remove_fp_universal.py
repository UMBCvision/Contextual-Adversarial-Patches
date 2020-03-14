# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os,sys
import _pickle as cPickle
import numpy as np
import math
from shutil import copyfile
import pdb


# Patch parameters
# max_epochs = 1
patchSize = 100
# num_iter = 500
start_x = 5
start_y = 5


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        # obj_struct['bbox'] = [int(bbox.find('xmin').text.split('.')[0]),
        #                       int(bbox.find('ymin').text.split('.')[0]),
        #                       int(bbox.find('xmax').text.split('.')[0]),
        #                       int(bbox.find('ymax').text.split('.')[0])]
        objects.append(obj_struct)

    # Added to retrieve width and height
    size_obj = tree.find('size')
    size_struct = {}
    size_struct['width'] = size_obj.find('width').text
    size_struct['height'] = size_obj.find('height').text
    objects.append(size_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def remove_fp(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #     os.makedirs(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # added
    imagenames = [imagename.split('/')[-1].split('.')[0] for imagename in imagenames]
    # print(imagenames)

    # pdb.set_trace()
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        # print 'Saving cached annotations to {:s}'.format(cachefile)
        # with open(cachefile, 'w') as f:
        #     cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename][:-1] if obj['name'] == classname]  # to avoid error
        # Added to retrieve width and height
        # R.append(recs[imagename][-1])
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'size': recs[imagename][-1]}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # Added

    #newdetfile = detfile.replace('no_class_overlap_patch','no_class_overlap_patch_removed_fp')
    newdetfile = sys.argv[4] + classname + '.txt'

    # create dir
    if not os.path.exists(os.path.dirname(newdetfile)):
        os.makedirs(os.path.dirname(newdetfile))

    filewr = open(newdetfile, 'w')

    splitlines = [x.strip().split(' ') for x in lines]

    # Added if detection file is empty
    if len(splitlines) == 0:
        return 0, 0, 0

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # print(BB.size,len(sorted_ind))

    # ani_dict ={}
    # for i, elem in enumerate(sorted_ind):
    #     ani_dict[elem] = i

    # print(sorted_ind[:10])
    # for elem in sorted_ind[:10]:
    #     print ani_dict[elem]

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)     # Change

        width = int(class_recs[image_ids[d]]['size']['width'])
        height = int(class_recs[image_ids[d]]['size']['height'])

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(((BBGT[:, 0])*416)/width, bb[0])             # Change GT annotations to 416*416 size image
            iymin = np.maximum(((BBGT[:, 1])*416)/height, bb[1])            # Change GT annotations to 416*416 size image
            ixmax = np.minimum(((BBGT[:, 2])*416)/width, bb[2])             # Change GT annotations to 416*416 size image
            iymax = np.minimum(((BBGT[:, 3])*416)/height, bb[3])            # Change GT annotations to 416*416 size image
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (((BBGT[:, 2])*416)/width - ((BBGT[:, 0])*416)/width + 1.) *
                   (((BBGT[:, 3])*416)/height - ((BBGT[:, 1])*416)/height + 1.) - inters)

            # ixmin = np.maximum(BBGT[:, 0], bb[0])
            # iymin = np.maximum(BBGT[:, 1], bb[1])
            # ixmax = np.minimum(BBGT[:, 2], bb[2])
            # iymax = np.minimum(BBGT[:, 3], bb[3])
            # iw = np.maximum(ixmax - ixmin + 1., 0.)
            # ih = np.maximum(iymax - iymin + 1., 0.)
            # inters = iw * ih
            #
            # # union
            # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1


                    # if true positive - write into another file
                    filewr.write(lines[sorted_ind[d]])
                else:
                    fp[d] = 1.

                    # if false positive - check overlap with patch location and decide whether to write into another file -
                    # No need to normalize because detections are already in the image size range

                    # bb[0] = math.floor(bb[0]/width*416)
                    # bb[1] = math.floor(bb[1]/height*416)
                    # bb[2] = math.ceil(bb[2]/width*416)
                    # bb[3] = math.ceil(bb[3]/height*416)


                    A = min(bb[2], start_x+patchSize)
                    B = max(bb[0], start_x)
                    C = min(bb[3], start_y+patchSize)
                    D = max(bb[1], start_y)

                    # iw = np.maximum(C - A + 1., 0.)
                    # ih = np.maximum(D - B + 1., 0.)
                    iw = np.maximum(A - B + 1., 0.)                 # Last moment bug-fix
                    ih = np.maximum(C - D + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           ((start_x+patchSize) - (start_x) + 1.) *
                           ((start_y+patchSize) - (start_y) + 1.) - inters)

                    overlaps = inters / uni

                    if overlaps > 0:
                        continue
                    else:
                        filewr.write(lines[sorted_ind[d]])
        else:
            fp[d] = 1.

            # if false positive - check overlap with patch location and decide whether to write into another file
            # xmin_new = math.floor(bb[0]/width*416)
            # ymin_new = math.floor(bb[1]/height*416)
            # xmax_new = math.ceil(bb[2]/width*416)
            # ymax_new = math.ceil(bb[3]/height*416)


            # find intersection of bounding box with patch
            A = min(bb[2], start_x+patchSize)
            B = max(bb[0], start_x)
            C = min(bb[3], start_y+patchSize)
            D = max(bb[1], start_y)

            # iw = np.maximum(C - A + 1., 0.)
            # ih = np.maximum(D - B + 1., 0.)
            iw = np.maximum(A - B + 1., 0.)                 # Last moment bug-fix
            ih = np.maximum(C - D + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   ((start_x+patchSize) - (start_x) + 1.) *
                   ((start_y+patchSize) - (start_y) + 1.) - inters)

            overlaps = inters / uni

            if overlaps > 0:
                continue
            else:
                filewr.write(lines[sorted_ind[d]])


    # # compute precision recall
    # fp = np.cumsum(fp)
    # tp = np.cumsum(tp)
    # rec = tp / float(npos)
    # # avoid divide by zero in case the first detection matches a difficult
    # # ground truth
    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    filewr.close()



def _do_python_eval(res_prefix, output_dir = 'output'):
    _devkit_path = '<devkit_root>/detection-patch/VOCdevkit'          # Changed
    # _devkit_path = '/VOCdevkit'                                                 # add your own devkit path
    _year = '2007'
    _classes = ('__background__', # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

    #filename = '/data/hongji/darknet/results/comp4_det_test_{:s}.txt'
    filename = res_prefix + '{:s}.txt'
    annopath = os.path.join(
        _devkit_path,
        'VOC' + _year,
        'Annotations',
        '{:s}.xml')


    # Changed

    # imagesetfile = os.path.join(
    #     _devkit_path,
    #     'VOC' + _year,
    #     'ImageSets',
    #     'Main',
    #     'test.txt')
    #cachedir = os.path.join(_devkit_path, 'annotations_cache')


    cachedir = sys.argv[3]
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(_year) < 2010 else False
    use_07_metric = True
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)


    imagesetfile = sys.argv[2]
    print(imagesetfile)
    # Changed
    for i, cls in enumerate(_classes):
        if cls != sys.argv[5]:                  # Change
            if cls != '__background__':
                copyfile(res_prefix + cls + '.txt', sys.argv[4] + cls + '.txt')
            continue

        # CHANGE - remove for all classes

        # if cls == "__background__":
        #     continue

        # print(filename, cls)
        remove_fp(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
    #     aps += [ap]
    #     print('AP for {} = {:.4f}'.format(cls, ap))
    #     # Changed
    #     # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
    #     #     cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')

    # for i, cls in enumerate(_classes):
    #     if cls == '__background__':
    #         continue

    #     print(imagesetfile)
    #     rec, prec, ap = voc_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #             use_07_metric=use_07_metric)

    #     print('AP for {} = {:.4f}'.format(cls, ap))
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    # print('-- Thanks, The Management')
    # print('--------------------------------------------------------------')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 6:
        res_prefix = sys.argv[1]
        if not os.path.exists(os.path.dirname(sys.argv[4])):
            os.makedirs(os.path.dirname(sys.argv[4]))
        _do_python_eval(res_prefix, output_dir = 'output')
    else:
        print('Usage:')
        print(' python remove_fp_patch.py result_prefix imagesetfile cachedir new_result_prefix classname')