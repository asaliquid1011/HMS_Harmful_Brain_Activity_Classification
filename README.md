5th place solution(UEMU's part) in [kaggle HMS competiton](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).

## SOLUTION
[Details](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492652)\
[Inference notebook](https://www.kaggle.com/code/asaliquid1011/hms-team-inference-ktmud)

## ARCHIVE CONTENTS
```
1)train_code      : code for train models(UEMU's part)
2)ensemble_code   : code for ensemble models(UEMU&kazumax&TH's part)
```

## HARDWARE:
```
os         : Ubuntu 22.04
cpu        : Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz (8 core)
memory     : 256gb
gpu        : 1 x NVIDIA GeForce RTX 4090
```

## SOFTWARE:
python packages are detailed separately in `requirements.txt`.
```
Python     : 3.10.12
CUDA       : 12.1
nvidia drivers : 535.171.04
```

## BUILD DOCKER:
```
docker compose up -d
```

## DATA SETUP
assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed
```
mkdir -p data/input/hms-harmful-brain-activity-classification/
cd data/input/hms-harmful-brain-activity-classification/
kaggle competitions download -c hms-harmful-brain-activity-classification
```

## MODEL BUILD train models(UEMU's part).
```
python ./train_SED_bird3.py --config <configfilename>
```

## MODEL BUILD ensemble models(UEMU&kazumax&TH's part).

2) prediction
python ./inference_SED_bird3.py --config <configfilename> --num-workers 4 --batch-size 32 --coeff-thresh 1.0 --mode-ens Average
