import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer

def format_prompt(args):
    """Format the prompt to match the fine-tuning data structure."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful Swiss policy advisor. Below you are asked a policy issue or question. "
        f"You are in the political party {args.party} and you reply in {args.language}.\n"
        f"{args.query}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    )

def generate_response(model, tokenizer, args, max_length=512):
    """Generate a response from the model given a prompt."""
    formated_prompt = format_prompt(args)
    inputs = tokenizer(formated_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    response = response[len(formated_prompt):].strip()

    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the fine-tuned model.")
    parser.add_argument("--query", required=True, type=str, help="The query you want answered")
    parser.add_argument("--party", required=True, type=str, help="The party you want the model to represent")
    parser.add_argument("--language", required=True, type=str, help="The language you want the model to respond in")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    response = generate_response(model, tokenizer, args)
    print("Generated Response:\n", response)

if __name__ == "__main__":
    main()

