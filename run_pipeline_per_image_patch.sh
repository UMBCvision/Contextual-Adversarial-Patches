#!/bin/bash

# train
python train_per_image_patch.py cfg/voc.data cfg/yolo-voc.cfg weights/yolo-voc.weights dataset/no_class_overlap_clean_test/cow_test.txt output/perimagepatch/backup/cow output/perimagepatch/patched_images/cow 9 0 logs/perimagepatch/cow.log

# create filelist
cd output/perimagepatch/patched_images/cow
ls $PWD/* > ../../../../dataset/perimagepatch/cow_test.txt
cd ../../../..

# validation
python valid_patch.py cfg/voc.data cfg/yolo-voc.cfg weights/yolo-voc.weights dataset/perimagepatch/cow_test.txt cow results/perimagepatch/with_fp/cow 0

# remove fp
python scripts/remove_fp_patch.py results/perimagepatch/with_fp/cow/comp4_det_test_ dataset/perimagepatch/cow_test.txt dummyfolder results/perimagepatch/removed_fp/cow/comp4_det_test_ cow

# Run VOC evaluation with_fp
python scripts/voc_eval_patch.py results/perimagepatch/with_fp/cow/comp4_det_test_ dataset/perimagepatch/cow_test.txt dummyfolder cow

# Run VOC evaluation removed_fp
python scripts/voc_eval_patch.py results/perimagepatch/removed_fp/cow/comp4_det_test_ dataset/perimagepatch/cow_test.txt dummyfolder cow