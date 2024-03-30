#!/usr/bin/env bash

# install requirements
pip install torch==1.11.0 torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.11.0)
pip install lmdb
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install torch-geometric==1.7.2
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install performer-pytorch
pip install tensorboard
pip install setuptools==59.5.0
pip install dpu-utils
pip install fairseq

