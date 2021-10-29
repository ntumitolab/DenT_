# DenT: Dense Transformers for 3Dbiomedical image Segmentation

![Image name]()

## Requirements

All of the processes are trained on Taiwan Computing Cloud (TWCC) and with an NVIDIA graphics card (NVIDIA V100). It is recommended to install latest drivers and use the GPU with 24+ GB of RAM.

## Environment setup
- Recommend: Install [Miniconda](https://conda.io/miniconda.html) for environment control.
- Create a Conda environment for the platform: python = 3.6 or up, pytorch = 20.02 or up.

## Data
- Data is currently available [here](https://drive.google.com/drive/folders/10LJxPudM3GWgYvN6Rz-DhLWFRMt4WJLD?usp=sharing). Download to the `./data/` directory (for example, you should see data inside `./data/DenT`). 
- If you want to train with your own data, please put your dataset `***` inside `./data/`. target dataset and source dataset should be named as `./data/***/train/target`and`./data/***/train/source`.
- Create your own testing set and validation set by running the following script:
```shell
python split_data.py
```
you should now see 3 folders inside the `./data/***` (train, test, val), each consists of the folder `target` and `source`.

## Train
Start training a model with:
- data: YOUR_OWN_DATA (e.g., Mito)
- source: SOURCE_IMAGE_OF_DATA (e.g., source)
- target: TARGET_IMAGE_OF_DATA (e.g., target)
- model: YOUR_MODEL (e.g., DenT)
```shell
./scripts/train.sh Mito source target DenT
```
You can set your own **batch size** and **checkpoint saved per * epoches** inside `train.py`.
The model will be saved at `./checkpoints` and the log will be saved at `./logs`.

## Predict the structures with the trained model
Run Segmentation test by:
- data: YOUR_OWN_DATA (e.g., Mito)
- source: SOURCE_IMAGE_OF_DATA (e.g., source)
- target: TARGET_IMAGE_OF_DATA (e.g., target)
- segmentation_directory: YOUR_RESULT (e.g., seg_result)
- model: YOUR_MODEL (e.g., DenT)
```shell
./scripts/test.sh Mito source target seg_result DenT
```
You may get the segmentation result inside `./seg_result`.

## Reference
- [Google ViT](https://github.com/google-research/vision_transformer)
- [TransUNet](https://github.com/Beckschen/TransUNet)

## Citations

```bibtex
```

