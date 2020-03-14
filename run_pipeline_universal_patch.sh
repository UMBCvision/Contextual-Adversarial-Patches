#!/bin/bash

# train
python train_universal_patch.py cfg/voc.data cfg/yolo-voc.cfg weights/yolo-voc.weights dataset/no_class_overlap_clean_test/train_test_half_1.txt dummyfolder output/universalpatch/noise/train/batch_size_1 18 0 logs/universalpatch/train/batch_size_1/train.log 1 100

# validation
python valid_npy_array_universal.py cfg/voc.data cfg/yolo-voc.cfg weights/yolo-voc.weights dataset/no_class_overlap_clean_test/train_test_half_2.txt train output/universalpatch/noise/train/batch_size_1/epoch_100_universal_patch.npy results/universalpatch/train/with_fp/batch_size_1 0

# remove fp
python scripts/remove_fp_universal.py results/universalpatch/train/with_fp/batch_size_1/comp4_det_test_ dataset/no_class_overlap_clean_test/train_test_half_2.txt dummyfolder results/universalpatch/train/removed_fp/batch_size_1/comp4_det_test_ train

# Run VOC evaluation with_fp
python scripts/voc_eval_universal.py results/universalpatch/train/with_fp/batch_size_1/comp4_det_test_ dataset/no_class_overlap_clean_test/train_test_half_2.txt dummyfolder train

# Run VOC evaluation removed_fp
python scripts/voc_eval_universal.py results/universalpatch/train/removed_fp/batch_size_1/comp4_det_test_ dataset/no_class_overlap_clean_test/train_test_half_2.txt dummyfolder train