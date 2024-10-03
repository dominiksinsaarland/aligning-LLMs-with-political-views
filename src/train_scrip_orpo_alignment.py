import torch
from transformers import TrainingArguments
from trl import DPOTrainer, ORPOTrainer, ORPOTrainer, ORPOConfig
from unsloth import FastLanguageModel
import json
from datasets import Dataset
import argparse
from peft import PeftModel

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='unsloth/zephyr-sft')
	parser.add_argument('--save_path', type=str, default='sample-outpath')
	parser.add_argument('--filename', type=str, default="data/smartvote_dataset_trainset.json")

	args = parser.parse_args()
	max_seq_length = 1024 # Supports automatic RoPE Scaling, so choose any number.

	orpo_config = ORPOConfig(
		beta=0.1, # the lambda/alpha hyperparameter in the paper/code
		per_device_train_batch_size = 4,
		gradient_accumulation_steps = 4,
		warmup_ratio = 0.1,
		num_train_epochs = 10,
		fp16 = not torch.cuda.is_bf16_supported(),
		bf16 = torch.cuda.is_bf16_supported(),
		logging_steps = 1,
		optim = "adamw_8bit",
		seed = 42,
		output_dir = args.save_path,
		learning_rate= 2e-5,
		lr_scheduler_type = "cosine",
	)

	model, tokenizer = FastLanguageModel.from_pretrained(
	    model_name = args.model_name,
	    max_seq_length = max_seq_length,
	    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
	    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
	    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
	    cache_dir = "../../transformer_models/"
	)


	model = FastLanguageModel.get_peft_model(
	    model,
	    r = 64,
	    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
		              "gate_proj", "up_proj", "down_proj",],
	    lora_alpha = 64,
	    lora_dropout = 0, # Supports any, but = 0 is optimized
	    bias = "none",    # Supports any, but = "none" is optimized
	    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
	    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
	    random_state = 3407,
	    max_seq_length = max_seq_length,
	)

	with open(args.filename) as f:
		train_dataset = json.load(f)
	train_dataset = Dataset.from_dict(train_dataset)


	dpo_trainer = ORPOTrainer(
	    model = model,
	    args = orpo_config,
	    train_dataset = train_dataset,
	    tokenizer = tokenizer,
	)
	dpo_trainer.train()
