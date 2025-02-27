import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def load_model(model_path):
    """Load the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to the fine-tuned model.")
    parser.add_argument("--prompt", required=True, type=str, help="Input prompt for inference.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    response = generate_response(model, tokenizer, args.prompt)
    print("Generated Response:\n", response)

if __name__ == "__main__":
    main()

