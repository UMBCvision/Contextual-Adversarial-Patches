# Contextual-Adversarial-Patches

Official Implementation of the paper Adversarial Patches Exploiting Contextual Reasoning in Object Detection

The utilization of spatial context to improve accuracy in most fast object detection algorithms is well known. Detectors increase inference speed by doing a single forward pass per image which means they implicitly use contextual reasoning for their predictions. It can been shown that an adversary can design contextual adversarial patches - patches which do not overlap with any objects of interest in the scene - and exploit contextual reasoning to fool standard detectors. In this paper, we also study methods to fix this vulnerability. We design category specific adversarial patches which make a widely used object detector like YOLO blind to an attacker chosen object category and show that limiting the use of contextual reasoning during object detector training acts as a form of defense. We believe defending against context based adversarial attacks is not an easy task. We take a step towards that direction and urge the research community to give attention to this vulnerability.

![alt text][teaser]

## Requirements
Tested with pytorch 1.1 and python 3.6 \
An environment file has been provided.

## Dataset creation

Download PASCAL VOC data. You can follow the steps here https://github.com/marvis/pytorch-yolo2. Insert appropriate Devkit path. Look for occurrences of <devkit_root> in the project.
```python
python filter_PASCAL_VOC.py PASCAL_VOC_annotations.txt <devkit_root>/VOCdevkit/VOC2007/ImageSets/Main/test.txt
```

This script reads the annotation txt file containing the bounding box and size information of each image in PASCAL VOC 2007 and finds images for each class where no ground truth boxes of that class overlap with our patch location.

## Download Pretrained Weights
Download darknet weights
```
cd weights
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
Download YOLO VOC weights
```
wget http://pjreddie.com/media/files/yolo-voc.weights
```

## Per Image Patch
```bash
bash run_pipeline_per_image_patch.sh
```

This script trains contextual adversarial patch per image of a chosen category and evaluates YOLO on the patched images. Please change the VOC category name and the category index to run for desired category.

## Universal Patch
```bash
bash run_pipeline_universal_patch.sh
```

This script trains universal contextual adversarial patch for a chosen category and evaluates YOLO on the held out images patched with the universal patch. Please change the VOC category name and the category index to run for desired category.


## Train Grad Defense
```python
python train_defense.py cfg/voc.data cfg/yolo-defense.cfg weights/darknet19_448.conv.23 backupdir voc_train.txt
```

## Acknowledgement

This repository was built with help from https://github.com/marvis/pytorch-yolo2.

## Citation
If you find our paper or code useful, please cite us using
```bib
@article{saha2019adversarial,
  title={Adversarial Patches Exploiting Contextual Reasoning in Object Detection},
  author={Saha, Aniruddha and Subramanya, Akshayvarun and Patil, Koninika and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:1910.00068},
  year={2019}
}
```

[teaser]: https://github.com/UMBCvision/Contextual-Adversarial-Patches/blob/master/Teaser_Contextual_Reasoning.PNG