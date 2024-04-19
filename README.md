5th place solution(UEMU's part) in [kaggle HMS competiton](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

# ARCHIVE CONTENTS
1)train_code                   : code for train \
predict_code                 : code for prediction

## HARDWARE:
```
os         : Ubuntu 18.04.5 LTS (200 GB boot disk)
cpu        : Intel(R) Xeon(R) CPU @ 2.30GHz (8 vCPUs, 51 GB memory)
memory     : 256gb
gpu        : 1 x NVIDIA Tesla P100
```

## SOFTWARE:
python packages are detailed separately in `requirements.txt`
```
Python     : 3.10.12
CUDA     12.1
nvidia drivers v.460.32.03
```

## BUILD DOCKER:
```
docker compose up -d
```

## DATA SETUP
assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed
```
mkdir -p data/input/birdclef-2022/
cd data/input/birdclef-2022/
kaggle competitions download -c BIRDCLEF2022
```

## MODEL BUILD: There are three options to produce the solution.
1) train models
    a) runs in 4-5 hours
2) prediction
    a) runs in 1-2 minutes

shell command to run each build is below
1) train models
python ./train_SED_bird3.py --config <configfilename>

2) prediction
python ./inference_SED_bird3.py --config <configfilename> --num-workers 4 --batch-size 32 --coeff-thresh 1.0 --mode-ens Average
