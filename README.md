# DenT
## Requirements

All of the processes are trained on Linux (Ubuntu 18.04) and with an NVIDIA graphics card (NVIDIA Geforce 1080Ti). It is recommended to install latest drivers and use the GPU with 12+ GB of RAM.

## Installation
### Environment setup
- Install [Miniconda](https://conda.io/miniconda.html) for environment control.
- Create a Conda environment for the platform:
```shell
conda env create -f environment.yml
```
- Activate the environment:
```shell
conda activate fnet
```
- Test the environment by executing the script:
```shell
./scripts/test_run.sh
```
The installation will done if the script executes without errors.

## Data
Data is currently available [here](https://drive.google.com/drive/folders/1-YL5Byg2Wh6uiI5w_bSb0eDY7fh1zSrc?usp=sharing). Download to the `./data/` directory (for example, you should see data inside `./data/z_stack_confocal_HR`).

## Train a model with provided data
Start training a model with:
- data: YOUR_OWN_DATA (e.g., dna)
- GPU: WHICH_GPU_TO_USE (e.g., 0) 
```shell
./scripts/train_model.sh dna 0
```
The first time this is run, the DNA dataset will be split into 25% test and 75% training images. A model will be trained using the training images. The model will be stored in directory `saved_models/dna`.

`losses.csv` file is inside the `saved_models/dna` like this:
```shell
997,0.2585
998,0.2647
999,0.2837
1000,0.3116
```

## Predict the structures with the trained model
```
./scripts/predict.sh dna 0
```
Predicted outputs will be in directories `results/dna/test` and `results/dna/train` corresponding to predictions on the training set and on the test set respectively. For example, there are 20 prediction images inside the directory `results/dna/test`:
```
$ ls results/3d/dna/test
00  01  02  03  04  05  06  07  08  09  10  11  12  13  14  15  16  17  18  19  predict_options.json  predictions.csv
```
Each number represents a single dataset with source image (bright-field), predicted image, and target image (fluorescence image):
```
$ ls results/3d/dna/test/00
prediction_dna.tiff  signal.tiff  target.tiff
```
