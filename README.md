# HMS - Harmful Brain Activity Classification

## Environment Setup

Create python or conda environment
```
conda create --name hbac_env
```
```
python -m venv hbac_env
```

Install [pytorch](https://pytorch.org/)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install required packages
```
pip install -r requirements.txt
```

## Training

The main dataloading and training pipeline can be found in [hms-pipeline](hms-pipeline.ipynb)


Training can be monitored with Tensorboard using the following 

```
tensorboard --logdir <log directory>
```

Example of training graphs are shown below

## Results

