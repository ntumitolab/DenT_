#!/bin/bash
#PBS -l select=1:ncpus=16:ngpus=2
#PBS -q v100

# training parameters are mentioned in paper ( B. Implementation Details )
# 
# --target_image "" \    # predict objects: target, target_dna
# --epoch 200 \          # The number of training iterations is approximately 10,000 (200 epochs).
# --train_batch 2 \      # During the training progress, the batch size is 2 unless otherwise specified.
# --lr 2e-4 \            # The models are trained using the Adam optimizer with a learning rate of 2e-4 and weight decay of 5e-4.
# --weight_decay 5e-4 \  # The models are trained using the Adam optimizer with a learning rate of 2e-4 and weight decay of 5e-4.

source activate DenT-tensorboard
cd "/home/guest/gst23-16/DenT/ShangRu_202307_Test/pbs"

python "../../train.py" \
    --data_path "../../data/{DataSet}_DenT/" \
    --target_image "target" \
    --source_image "source" \
    --gpu 0 \
    --epoch 200 \
    --val_epoch 1 \
    --train_batch 2 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --model "DenT" \
    --log_dir "./logs/" \
    --checkpoints "./checkpoints/" \
    --random_seed 123 \
    
    # --deep_supervision \
    # --patch \