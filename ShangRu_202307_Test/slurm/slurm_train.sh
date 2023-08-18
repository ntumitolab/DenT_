#!/bin/bash
#SBATCH --job-name=DenT_test    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --cpus-per-task=8       ## 該 task 索取 32 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 8 GPUs
#SBATCH --time=00:3:00          ## 最長跑 10 分鐘 (測試完這邊記得改掉，或是直接刪除該行)
#SBATCH --account=MST112202     ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gtest        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)

# training parameters are mentioned in paper ( B. Implementation Details )
# 
# --target_image "" \    # predict objects: target, target_dna
# --epoch 200 \          # The number of training iterations is approximately 10,000 (200 epochs).
# --train_batch 2 \      # During the training progress, the batch size is 2 unless otherwise specified.
# --lr 2e-4 \            # The models are trained using the Adam optimizer with a learning rate of 2e-4 and weight decay of 5e-4.
# --weight_decay 5e-4 \  # The models are trained using the Adam optimizer with a learning rate of 2e-4 and weight decay of 5e-4.

module purge
module load miniconda3
conda activate "DenT-tensorboard"
cd "/home/twsqzqy988/DenT/ShangRu_202307_Test/slurm"

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