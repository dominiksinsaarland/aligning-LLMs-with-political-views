from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import DPOTrainer, SFTTrainer
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", default="unsloth/mistral-7b-bnb-4bit", type=str, help="")
	parser.add_argument("--output_dir", required=True, type=str, help="")
	parser.add_argument("--infile", required=True, type=str, help="")

	args = parser.parse_args()
	max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number.
	use_unsloth = True
	if use_unsloth:
		model, tokenizer = FastLanguageModel.from_pretrained(
		    model_name = args.model_name,
		    max_seq_length = max_seq_length,
		    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
		    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
		    cache_dir = "../../transformer_models/"
		)

		model = FastLanguageModel.get_peft_model(
		    model,
		    r = 16,
		    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
				      "gate_proj", "up_proj", "down_proj",],				      
		    lora_alpha = 64,
		    lora_dropout = 0, # Dropout = 0 is currently optimized
		    bias = "none",    # Bias = "none" is currently optimized
		    use_gradient_checkpointing = True,
		    random_state = 3407,
		)
	else:
		model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
		tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

	training_args = TrainingArguments(output_dir="./output")
	with open(args.infile) as f:
		train_dataset = json.load(f)
		# dict_keys(['prompt', 'chosen', 'rejected'])	
		# Instruction Format: {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

	del train_dataset['rejected']
	train_dataset["completion"] = train_dataset.pop("chosen")
	train_dataset = Dataset.from_dict(train_dataset)


	sft_trainer = SFTTrainer(
	    model = model,
	    args = TrainingArguments(
		per_device_train_batch_size = 4,
		gradient_accumulation_steps = 8,
		warmup_ratio = 0.1,
		num_train_epochs = 1,
		fp16 = not torch.cuda.is_bf16_supported(),
		bf16 = torch.cuda.is_bf16_supported(),
		logging_steps = 1,
		optim = "adamw_8bit",
		seed = 42,
		learning_rate=2e-5,
		output_dir = args.output_dir,
		lr_scheduler_type = "cosine",
	    ),
	    train_dataset = train_dataset,
	    tokenizer = tokenizer,
	    max_seq_length = 1024,
	)
	sft_trainer.train()
