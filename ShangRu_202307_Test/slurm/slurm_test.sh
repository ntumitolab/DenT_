#!/bin/bash
#SBATCH --job-name=DenT-target-test    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       ## 該 task 索取 32 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 8 GPUs
# SBATCH --time=00:2:00          ## 最長跑 10 分鐘 (測試完這邊記得改掉，或是直接刪除該行)
#SBATCH --account=MST112202     ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gtest        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)
#SBATCH --mail-type=ALL
#SBATCH --mail-user="rime97410000@gmail.com"

# training parameters are mentioned in paper ( B. Implementation Details )
# 
# --target_image "" \    # predict objects: target, target_dna

module purge
module load miniconda3
conda activate "DenT-tensorboard"
cd "/home/twsqzqy988/DenT/ShangRu_202307_Test/slurm"

python "../../test.py" \
    --data_path "../../data/{DataSet}_DenT/" \
    --target_image "target" \
    --source_image "source" \
    --gpu 0 \
    --test_batch 1 \
    --model "DenT" \
    --checkpoints "./checkpoints/[xxxxxx]" \
    --random_seed 123 \
    --seg_dir "./seg_results/[xxxxxx]" \

    # --patch \ # KiUnet parameters