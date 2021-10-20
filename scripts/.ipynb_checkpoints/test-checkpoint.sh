# TODO: create shell script for running the testing code of the baseline model
#!/bin/bash -x
python test.py --data_path $1 --target_image $2 --seg_dir $3 --model $4 --image_type $5
