#!/bin/bash -x
DIR=${1:-Train_info_Unet_15725}
PATH="logs/${DIR}"
tensorboard --logdir ${PATH}