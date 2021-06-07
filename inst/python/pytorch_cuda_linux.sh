#!/bin/bash

# uninstall PyTorch
conda uninstall pytorch
pip uninstall torch
pip uninstall torch # run this command twice

# install PyTorch coda version 11.1 (Linux)

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
