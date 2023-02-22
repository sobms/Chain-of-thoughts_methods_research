#!/bin/bash
# BEFORE RUNNING THIS SCRIPT PLEASE INSTALL ANACONDA3

# install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
# generation of a Container Device Interface (CDI) specification that refers to all devices
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# check the names of the generated devices
grep "  name:" /etc/cdi/nvidia.yaml

# install pytorch with cuda supporting, petals and tqdm
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U petals
pip install tqdm
# clone GSM8k dataset into project directory
git clone https://github.com/openai/grade-school-math.git