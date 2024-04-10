#!/bin/bash

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge hdbscan
conda install -c conda-forge opencv
conda install toml matplotlib tqdm
conda install tensorboard
conda install psutil
conda install numpy=1.24
