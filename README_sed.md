
#ARCHIVE CONTENTS
HMS_train_v2                 : code for train 

#HARDWARE: (The following specs were used to create the original solution)
Ubuntu 18.04.5 LTS (200 GB boot disk)
Intel(R) Xeon(R) CPU @ 2.30GHz (8 vCPUs, 51 GB memory)
1 x NVIDIA NVIDIA GeForce RTX 4090

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10.12
CUDA 12.2
nvidia drivers v.535.171.04

#DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# traindataset
mkdir -p data/input/birdclef-2022/
cd data/input/birdclef-2022/
kaggle competitions download -c BIRDCLEF2022

# backgroundnoise_for_augmentation, annotation data ,PL
set this dataset to './data/input/' and unzip.
https://www.kaggle.com/datasets/asaliquid1011/bird3-train-data

# for inference
set trained model to './models'
https://www.kaggle.com/datasets/asaliquid1011/bird3-infrence-data

#MODEL BUILD: There are three options to produce the solution.
1) train models
    a) runs in 4-12 hours
shell command to run each build is below
1) train models
python ./train_SED_bird3.py --config <configfilename>

