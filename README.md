# HMS - Harmful Brain Activity Classification

[Report](/report.pdf)

## Environment Setup

Create python or conda environment
```
conda create --name hbac_env python=3.9
```
```
python -m venv hbac_env
```

Activate environemtn
```
conda activate hbac_env
```
```
source hbac_env/bin/activate
```

Install [pytorch 2.0.0](https://pytorch.org/get-started/previous-versions/)

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install required packages
```
python -m pip install -r requirements.txt
```

## Dataset

The HBAC dataset can be obtained from the [HMS](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) Kaggle Website. Place the unzipped dataset in a data directory with the following structure:
```
.
└── hbac
    ├── example_figures
    ├── models
    ├── test_eegs
    ├── test_spectrograms
    ├── train_eegs
    └── train_spectrograms
```

## Training

The main dataloading and training pipeline can be found in [hms-pipeline](/hms-pipeline.ipynb). More information on the trianing process can be found in the notebook.


Training can be monitored with Tensorboard using the following 

```
tensorboard --logdir <log directory>
```

Example of training graphs are shown below

![training plots in tensorboard](/images/training_plots.png)

## Results

Results of CNN architectures using only spectrogram data as well as the contrastive CNN architecture using both spectrogram and EEG data are shown below.

| Model  | Test Accuracy | Test F-1 Score | Test KL-Loss |
| --- | --- | --- | --- |
| EfficientNet-b0 | 0.5962 | 0.6252 | 0.8949 |
| EfficientNetV2 | 0.5972 | 0.6334 | 0.871 |
| ConvNext | 0.6171 | 0.6441 | 0.9095 |
| Contrastive | 0.6128 | 0.6411 | 0.8432 |
