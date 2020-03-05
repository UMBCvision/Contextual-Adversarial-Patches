# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os,sys
import cPickle
import numpy as np

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
        objects.append(obj_struct)
    # Added by Aniruddha to retrieve width and height
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

def voc_eval(detpath,
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
    #     os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # added by Aniruddha
    imagenames = [imagename.split('/')[-1].split('.')[0] for imagename in imagenames]
    # print(imagenames)

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save - TARGETED only
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
        R = [obj for obj in recs[imagename][:-1] if obj['name'] == classname]
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




    splitlines = [x.strip().split(' ') for x in lines]

    # # Added by Aniruddha - if detection file is empty
    # if len(splitlines) == 0:
    #     return 0, 0, 0

    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    # print(BB.size,len(sorted_ind))


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
        BBGT = R['bbox'].astype(float)            # Change Aniruddha

        width = int(class_recs[image_ids[d]]['size']['width'])
        height = int(class_recs[image_ids[d]]['size']['height'])

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            # print(lenBBGT)
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

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap



def _do_python_eval(res_prefix, output_dir = 'output'):
    #_devkit_path = '/data/xiaohang/pytorch-yolo2/VOCdevkit'
    _devkit_path = '/mirror_nfs1/code/aniruddha/detection-patch/VOCdevkit'          # Changed by Aniruddha
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


    # Changed by Aniruddha
    # imagesetfile = '/mirror_nfs1/code/aniruddha/detection-patch/dataset/no_class_overlap/aeroplane_test.txt'

    # imagesetfile = os.path.join(
    #     _devkit_path,
    #     'VOC' + _year,
    #     'ImageSets',
    #     'Main',
    #     'test.txt')
    #cachedir = os.path.join(_devkit_path, 'annotations_cache')

    cachedir = sys.argv[3] #Hamed
#    cachedir = '/mirror_nfs1/code/aniruddha/detection-patch/annotations/KITTI/train_clean_car_half2' # + res_prefix.split('/')[-2]
    # cachedir = '/mirror_nfs1/code/aniruddha/detection-patch/annotations/experiment7_evaluation/2007_test/'
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(_year) < 2010 else False
    use_07_metric = True
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    imagesetfile = sys.argv[2] #Hamed
#    imagesetfile = '/mirror_nfs1/data/aniruddha/detection-patch/KITTI/dataset/car_train_half2.txt' # + res_prefix.split('/')[-2] + '_test_half_2.txt'
    # imagesetfile = '/mirror_nfs1/code/aniruddha/detection-patch/2007_test.txt'
    print(imagesetfile)
    # Changed by Aniruddha
    for i, cls in enumerate(_classes):
        if cls == '__background__':
            continue

        if cls != sys.argv[4]:                  # Change Aniruddha
            continue

        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        # Changed by Aniruddha
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
        #     cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
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

    #     imagesetfile = '/mirror_nfs1/code/aniruddha/detection-patch/dataset/no_class_overlap/' + cls + '_test.txt'
    #     print(imagesetfile)
    #     rec, prec, ap = voc_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #             use_07_metric=use_07_metric)

    #     print('AP for {} = {:.4f}'.format(cls, ap))
    print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    # print('-- Thanks, The Management')
    # print('--------------------------------------------------------------')


if __name__ == '__main__':
    #res_prefix = '/data/hongji/darknet/project/voc/results/comp4_det_test_'
    #classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    import sys
    if len(sys.argv) == 5: #Hamed
        res_prefix = sys.argv[1]
        _do_python_eval(res_prefix, output_dir = 'output')
    else:
        print('Usage:')
        print(' python voc_eval_patch.py result_prefix imagesetfile cachedir classname')


