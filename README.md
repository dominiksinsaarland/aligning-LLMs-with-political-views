# Aligning Large Language Models with Diverse Political Viewpoints

This repo contains data and replication code for the paper [(Stammbach et al., 2024)](https://arxiv.org/abs/2406.14155).

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:

```shell
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

