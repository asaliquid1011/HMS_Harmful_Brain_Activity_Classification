Below you can find a outline of how to reproduce my solution for the <BIRDCLEF2022> competition.
If you run into any trouble with the setup/code or have any questions please contact me at <jananesenoodleudon@gmail.com>

# ARCHIVE CONTENTS
train_code                   : code for train 
predict_code                 : code for prediction

# HARDWARE: (The following specs were used to create the original solution)
Google colab Pro +
Ubuntu 18.04.5 LTS (200 GB boot disk)
Intel(R) Xeon(R) CPU @ 2.30GHz (8 vCPUs, 51 GB memory)
1 x NVIDIA Tesla P100

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7.13
CUDA 11.2
cuddn 8.0.5
nvidia drivers v.460.32.03

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
# below are the shell commands used in each step, as run from the top level directory
mkdir -p data/input/birdclef-2022/
cd data/input/birdclef-2022/
kaggle competitions download -c BIRDCLEF2022

# MODEL BUILD: There are three options to produce the solution.
1) train models
    a) runs in 4-5 hours
2) prediction
    a) runs in 1-2 minutes

shell command to run each build is below
1) train models
python ./train_SED_bird3.py --config <configfilename>

2) prediction
python ./inference_SED_bird3.py --config <configfilename> --num-workers 4 --batch-size 32 --coeff-thresh 1.0 --mode-ens Average
