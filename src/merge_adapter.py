import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model")
    args = parser.parse_args()
    
    print(f"Loading base model {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu" # Merge on CPU to save VRAM
    )
    
    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    print("Merging adapter and unloading...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {args.output_path}...")
    merged_model.save_pretrained(args.output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("Merge complete! The model is ready to be loaded directly without PEFT.")

if __name__ == "__main__":
    main()
