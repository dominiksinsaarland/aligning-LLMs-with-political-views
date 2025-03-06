# Aligning Large Language Models with Diverse Political Viewpoints

This repo contains data and replication code for the paper [(Stammbach et al., 2024)](https://arxiv.org/abs/2406.14155).

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:

```shell
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes transformers
```

## SFT fine-tuning
```shell
# assuming slurm
sbatch --time=1240 --mem-per-cpu=25000 --gpus=1 --gres=gpumem:24G --wrap="python src/train_script_supervised_finetuning.py --infile data/smartvote_dataset_trainset.json --model_name unsloth/llama-3-8b-Instruct-bnb-4bit --output_dir llama-3-instruct-SFT"
```

## ORPO alignment
```shell
# assuming slurm
sbatch --time=2880 --mem-per-cpu=25000 --gpus=1 --gres=gpumem:24G --wrap="python src/train_script_orpo_new_params.py --filename data/smartvote_dataset_trainset.json --model_name unsloth/llama-3-8b-Instruct-bnb-4bit --save_path llama-3-instruct-ORPO"
```

## Generate with model
```shell
# assuming slurm
sbatch --time=300 --mem-per-cpu=25000 --gpus=1 --gres=gpumem:24G --wrap="python src/inference.py \
    --model_pth <path/to/checkpoint> \
	--query "<your query>" \
	--party "<a political party>" \
	--language "<choose a language>"
```

## Evaluation
- Diversity scores
- MAUVE scores

## Plain smartovte dataset


