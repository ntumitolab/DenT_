#!/bin/bash

# training parameters are mentioned in paper ( B. Implementation Details )
#
# --target_image "" \  # predict objects: target, target_dna


python "../../test.py" \
    --data_path "../../data/{DataSet}_DenT/" \
    --target_image "target" \
    --source_image "source" \
    --gpu 0 \
    --test_batch 1 \
    --model "DenT" \
    --checkpoints "./checkpoints/[xxxxxx]" \
    --random_seed 123 \
    --seg_dir "./seg_results/" \

    # --patch \ # KiUnet parameters